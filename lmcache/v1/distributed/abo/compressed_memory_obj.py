# SPDX-License-Identifier: Apache-2.0

"""
CompressedMemoryObj: ABO-compression-aware MemoryObj subclass.

Extends TensorMemoryObj with staging buffer management, async
compress/decompress state tracking, and compression-aware
.tensor / .byte_array property behaviour.

Lifecycle (STORE path):
  allocate → [staging buffer + raw_data (compressed space, fixed size)]
    │
    ▼ GPU→CPU D2H copy → data written into staging buffer
    │
    ▼ _compress_task (async thread):
    │   staging → codec.compress → raw_data
    │   release staging buffer
    │
    ▼ L2 store:
        byte_array → memoryview of raw_data
        (fixed size, codec self-describes boundaries)

Lifecycle (RETRIEVE path):
  L2 load → raw_data (compressed data)
    │
    ▼ acquire staging buffer + decompress (async CUDA stream):
    │   raw_data → codec.decompress → staging buffer
    │   record _decompress_event
    │
    ▼ CPU→GPU H2D copy ← staging buffer (.tensor property wait_event)
    │
    ▼ release staging buffer
"""

# Standard
from typing import TYPE_CHECKING, Optional
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import (
    MemoryObjMetadata,
    TensorMemoryObj,
    get_size_bytes,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.abo.staging_pool import StagingPool
    from lmcache.v1.memory_management import MemoryAllocatorInterface

logger = init_logger(__name__)


class CompressedMemoryObj(TensorMemoryObj):
    """ABO-compression-aware MemoryObj subclass.

    Extends TensorMemoryObj with:
    - Staging buffer management (for D2H/H2D transfer intermediary)
    - Async compress/decompress state tracking
    - Multi-state .tensor property behaviour
    - Compression-aware .byte_array property behaviour
    """

    # Class-level staging pool shared by all instances.
    # Injected once via set_staging_pool() after StagingPool is created.
    _cls_staging_pool: Optional["StagingPool"] = None

    @classmethod
    def set_staging_pool(cls, pool: "StagingPool") -> None:
        """Inject the global StagingPool reference (called once at init).

        Args:
            pool: StagingPool instance shared by all CompressedMemoryObj.
        """
        cls._cls_staging_pool = pool

    def __init__(
        self,
        raw_data: torch.Tensor,
        metadata: MemoryObjMetadata,
        parent_allocator: Optional["MemoryAllocatorInterface"],
        staging_tensor: Optional[torch.Tensor],
        original_shapes: list[torch.Size],
        original_dtypes: list[torch.dtype],
    ):
        """Initialize CompressedMemoryObj.

        Args:
            raw_data: Compressed-size space allocated from L1 pool (uint8 flat tensor).
            metadata: Memory object metadata.
            parent_allocator: Parent allocator (LazyMemoryAllocator).
            staging_tensor: Pinned buffer borrowed from StagingPool
                (optional, None for RETRIEVE).
            original_shapes: Shapes of the original uncompressed tensors (per group).
            original_dtypes: Dtypes of the original uncompressed tensors (per group).
        """
        super().__init__(raw_data, metadata, parent_allocator)

        # Staging buffer related
        self._staging_tensor: Optional[torch.Tensor] = staging_tensor
        self._original_shapes: list[torch.Size] = original_shapes
        self._original_dtypes: list[torch.dtype] = original_dtypes

        # Compress state (STORE path)
        self._compress_failed: bool = (
            False  # True if compress failed (graceful degradation)
        )
        self._compress_event: Optional[torch.cuda.Event] = None

        # Decompress state (RETRIEVE path)
        self._need_decompress: bool = False
        self._decompress_failed: bool = (
            False  # True if decompress failed (graceful degradation)
        )
        self._decompress_event: Optional[torch.cuda.Event] = None
        self._h2d_event: Optional[torch.cuda.Event] = (
            None  # CUDA event for H2D completion (for per-chunk release)
        )

        # Backup of meta.address (set to 0 during staging phase, restored
        # after compress)
        self._original_address: Optional[int] = None

        # Staging reference count: tracks pending operations using staging.
        # +1 on acquire / ref_acquire, -1 on release_staging.
        # Staging is only truly released when ref_count reaches 0.
        self._staging_ref_count: int = 0

        # Lock protecting staging state (has_staging check + acquire/release).
        # In MLA mode, multiple TP workers may concurrently access the same
        # CompressedMemoryObj from different RETRIEVE threads.
        self._staging_lock = threading.Lock()

        # If holding staging, set up memcpy adaptation immediately
        if self._staging_tensor is not None:
            self._setup_staging_memcpy_meta()

    def _setup_staging_memcpy_meta(self):
        """Modify meta during staging phase so lmcache_memcpy_async works correctly.

        Sets meta.address to 0 because the staging buffer is an independent
        pinned memory region. Combined with PIN_CHUNK_SIZE being set to
        >= kv_chunk_size, this makes lmcache_memcpy_async issue a single
        cudaMemcpyAsync for the staging buffer (no segmentation).
        """
        self._original_address = self.meta.address
        self.meta.address = 0

    def _restore_raw_data_meta(self):
        """Restore meta to raw_data's address after compress is done."""
        if self._original_address is not None:
            self.meta.address = self._original_address
            self._original_address = None

    def get_size(self) -> int:
        """Get the effective byte size for memcpy operations.

        During staging phase (holding staging buffer), returns the
        uncompressed size so that lmcache_memcpy_async copies the
        full KV data. After staging is released, falls back to the
        base class which returns the compressed (raw_data) size.
        """
        if self._staging_tensor is not None:
            return get_size_bytes(self._original_shapes, self._original_dtypes)
        return super().get_size()

    def _resolve_effective_buffer(self) -> Optional[torch.Tensor]:
        """Resolve the effective host buffer with validation and decompress sync.

        Shared logic for .raw_tensor / .tensor / .data_ptr:
        1. Invalid / failed → None
        2. Wait decompress event on staging (RETRIEVE path)
        3. Return staging tensor or raw_data
        """
        if not self.valid:
            logger.warning("Trying to access an invalidated CompressedMemoryObj")
            return None

        if self._compress_failed or self._decompress_failed:
            return None

        if self._staging_tensor is not None:
            # Wait for decompress event (idempotent, multi-device safe).
            # In MLA mode each device's stream must independently wait_event.
            if self._decompress_event is not None:
                torch.cuda.current_stream().wait_event(self._decompress_event)
            return self._staging_tensor

        return self.raw_data

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        """Flat uint8 host buffer for memcpy operations."""
        return self._resolve_effective_buffer()

    @property
    def data_ptr(self) -> Optional[int]:  # type: ignore[override]
        """Host-side data pointer for memcpy operations."""
        buf = self._resolve_effective_buffer()
        return buf.data_ptr() if buf is not None else None

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        """Shaped tensor view (staging → typed view, otherwise raw_data view)."""
        buf = self._resolve_effective_buffer()
        if buf is None:
            return None

        if buf is self._staging_tensor:
            # Return staging buffer view (first group shape/dtype)
            original_bytes = (
                self.meta.shapes[0].numel()  # type: ignore[index]
                * self.meta.dtypes[0].itemsize  # type: ignore[index]
            )
            return (
                buf[:original_bytes]
                .view(self.meta.dtypes[0])  # type: ignore[index]
                .view(self.meta.shapes[0])  # type: ignore[index]
            )

        # Staging released, return raw_data view
        return super().tensor

    @property
    def byte_array(self) -> Optional[memoryview]:  # type: ignore[override]
        """Get binary data view.

        Compression-aware behaviour:
        - If _compress_failed=True: return None (graceful degradation)
        - If a compress task has been submitted (_compress_event is set) and
          has not failed, wait for compress_event to complete. This blocks
          the caller (StoreController's background thread) until compressed
          data is ready in raw_data, without blocking the main thread.
        - Returns memoryview of full raw_data (fixed size buffer).
          The codec embeds block metadata so the decompressor
          self-describes boundaries; no need to trim to actual
          compressed size.
        """
        # Compress failed: no valid compressed data available
        if self._compress_failed:
            return None

        # Compression still in progress: wait for it to complete
        if self._compress_event is not None:
            self._compress_event.synchronize()

        return super().byte_array

    def set_compress_event(self, event: torch.cuda.Event):
        """Set the CUDA event for compress completion (stream mode).

        Args:
            event: Compress-done event recorded on compress_stream.
        """
        self._compress_event = event

    def mark_compress_failed(self):
        """Mark compress as failed (graceful degradation).

        The object will not have valid compressed data, so L2 store
        should skip it. Sets _compress_failed=True so that .tensor
        returns None.
        """
        self._compress_failed = True

    def mark_decompress_failed(self):
        """Mark decompress as failed (graceful degradation).

        Sets _decompress_failed=True so that .tensor returns None,
        which triggers an error in the H2D path and falls into
        the retrieve failure handling.
        """
        self._decompress_failed = True
        self._decompress_event = None

    def try_acquire_staging(self, extra_count: int = 0) -> bool:
        """Acquire a staging buffer from the class-level StagingPool.

        On success, sets self._staging_tensor and calls
        _setup_staging_memcpy_meta().

        Thread-safe: protected by _staging_lock.

        Args:
            extra_count: Extra number of ref counts to add (default 0).
                For prefetch/retrieve paths, pass tp_size so that
                each TP worker can independently release_staging.

        Returns:
            True if staging buffer was acquired, False otherwise.
        """
        pool = CompressedMemoryObj._cls_staging_pool
        if pool is None:
            logger.error("StagingPool not initialized (set_staging_pool not called)")
            return False
        with self._staging_lock:
            # Double-check: another thread may have acquired staging already
            if self._staging_tensor is not None:
                self._staging_ref_count += 1 + extra_count
                return True
            staging = pool.try_acquire()
            if staging is None:
                return False
            self._staging_tensor = staging
            self._staging_ref_count += 1 + extra_count
            self._setup_staging_memcpy_meta()
            return True

    def setup_decompress(
        self,
        decompress_event: torch.cuda.Event,
    ):
        """Set up decompress state (used in RETRIEVE path).

        Staging buffer must already be acquired via try_acquire_staging()
        before calling this method.

        Args:
            decompress_event: CUDA event for decompress completion.
        """
        assert self._staging_tensor is not None, (
            "setup_decompress requires staging buffer (call try_acquire_staging first)"
        )
        self._decompress_event = decompress_event
        self._need_decompress = True

    def mark_decompress_done(self):
        """Mark decompress as done.

        Clears decompress state after decompressed data has been consumed.
        """
        self._need_decompress = False
        self._decompress_event = None

    def staging_ref_acquire(self, extra_count: int = 0):
        """Increment staging ref count (for reusing existing staging).

        Called when RETRIEVE path skips decompress because staging
        already holds uncompressed data (STORE path in progress).

        Thread-safe: protected by _staging_lock.

        Args:
            extra_count: Number of ref counts to add (default 0).
        """
        with self._staging_lock:
            self._staging_ref_count += 1 + extra_count

    def mark_staging_released(self):
        """Mark staging buffer as released.

        Clears staging reference and restores meta.address.
        The actual buffer return to pool should be done by the caller.

        Returns:
            The staging tensor that was held (or None if already released).
        """
        staging = self._staging_tensor
        self._staging_tensor = None
        self._restore_raw_data_meta()
        self._need_decompress = False
        self._decompress_failed = False
        self._decompress_event = None
        self._h2d_event = None
        return staging

    def release_staging(self, extra_count: int = 0):
        """Decrement staging ref count and release when it reaches 0.

        Each acquire / staging_ref_acquire increments the count;
        each release_staging decrements it. The staging buffer is
        only truly returned to the pool when the count reaches 0.
        This prevents the STORE compress callback from releasing
        staging while a concurrent RETRIEVE H2D is still using it.

        Thread-safe: protected by _staging_lock.

        Args:
            extra_count: Number of ref counts to subtract (default 0).
        """
        staging_to_release = None
        with self._staging_lock:
            if self._staging_tensor is None:
                return

            self._staging_ref_count -= 1 + extra_count
            if self._staging_ref_count > 0:
                return

            staging_to_release = self.mark_staging_released()

        # Return staging buffer to pool (outside lock to avoid holding
        # lock during pool operations)
        if staging_to_release is not None:
            pool = CompressedMemoryObj._cls_staging_pool
            if pool is not None:
                pool.release(staging_to_release)

    @property
    def has_staging(self) -> bool:
        """Whether currently holding a staging buffer."""
        return self._staging_tensor is not None

    def __del__(self):
        """Destructor: ensure staging buffer is returned to pool."""
        if self._staging_tensor is not None:
            try:
                self.release_staging()
            except Exception:
                pass
        # Call parent destructor
        super().__del__()

    def __repr__(self) -> str:
        return (
            f"CompressedMemoryObj("
            f"address={self.meta.address}, "
            f"phy_size={self.meta.phy_size}, "
            f"original_shapes={self._original_shapes}, "
            f"original_dtypes={self._original_dtypes}, "
            f"has_staging={self.has_staging}, "
            f"compress_failed={self._compress_failed}, "
            f"need_decompress={self._need_decompress}, "
            f"decompress_failed={self._decompress_failed})"
        )
