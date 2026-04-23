# SPDX-License-Identifier: Apache-2.0
"""
ABO-aware StorageManager subclass.

Overrides ``_create_l1_manager`` to return ABOL1Manager,
and adds all ABO-specific behaviour: compress/decompress managers,
staging pool, and hooks into reserve_write / submit_prefetch_task /
_on_prefetch_l1_hits / close.
"""

# Standard
from typing import Literal
import threading

# Third Party
import cupy
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.abo.abo_codec import (
    _DEFAULT_RATIO,
    ABOCodecFactory,
    resolve_abo_dtype,
)
from lmcache.v1.distributed.abo.abo_l1_manager import ABOL1Manager
from lmcache.v1.distributed.abo.compress_manager import ABOCompressManager
from lmcache.v1.distributed.abo.compressed_memory_obj import CompressedMemoryObj
from lmcache.v1.distributed.abo.decompress_manager import ABODecompressManager
from lmcache.v1.distributed.abo.staging_pool import StagingPool
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj, get_size_bytes

logger = init_logger(__name__)


class ABOStorageManager(StorageManager):
    """StorageManager with ABO KV compression support.

    Overrides ``_create_l1_manager`` to return ABOL1Manager and adds
    all ABO-specific components (codec, compress/decompress managers,
    staging pool, release stream).
    """

    def _create_l1_manager(self, config: StorageManagerConfig) -> L1Manager:
        """Override: create ABOL1Manager with abo config."""
        return ABOL1Manager(config.l1_manager_config, abo_config=config.abo_config)

    def __init__(self, config: StorageManagerConfig):
        super().__init__(config)
        self._init_abo(config)

    def _init_abo(self, config: StorageManagerConfig) -> None:
        """Initialize ABO compression components."""
        abo = config.abo_config
        logger.info(
            "Initializing ABO compression: "
            "staging_size=%.1fGB, ratio=%s, codec=%s, threads=%d",
            abo.staging_size_gb,
            abo.ratio,
            abo.codec,
            abo.num_threads,
        )

        # 1. Create codec
        self._abo_codec = ABOCodecFactory.create_codec(abo.codec, abo)
        self._abo_config = abo

        # 2. Create ABOCompressManager with on_compress_failed callback
        def _on_compress_failed(obj_key: ObjectKey) -> None:
            try:
                result = self._l1_manager.abort_write([obj_key])  # type: ignore[attr-defined]  # noqa: E501,F821
                logger.warning(
                    "ABO compress failed for key %s, abort_write result: %s",
                    obj_key,
                    result,
                )
            except Exception as e:
                logger.error(
                    "Failed to abort_write after compress failure (key=%s): %s",
                    obj_key,
                    e,
                )

        self._abo_compress_manager = ABOCompressManager(
            codec=self._abo_codec,
            on_compress_failed=_on_compress_failed,
        )

        # 3. Create ABODecompressManager
        self._abo_decompress_manager = ABODecompressManager(
            codec=self._abo_codec,
        )

        # 4. Record staging config for lazy initialization
        self._staging_size_gb = abo.staging_size_gb
        self._staging_pool: StagingPool | None = None
        self._staging_init_lock = threading.Lock()

        # 5. Release stream for async staging release (lazy init)
        self._release_stream: torch.cuda.Stream | None = None
        self._cupy_release_stream: cupy.cuda.ExternalStream | None = None

        logger.info(
            "ABO compression initialized "
            "(StagingPool will be created on register_kv_cache or first reserve_write)"
        )

    def init_staging_pool(self, layout_desc: MemoryLayoutDesc) -> None:
        """Initialize StagingPool and ABO params (thread-safe, idempotent).

        Called from server.register_kv_cache() to eagerly initialize,
        or from reserve_write() as fallback.
        """
        if self._staging_pool is not None:
            return
        with self._staging_init_lock:
            if self._staging_pool is not None:
                return

            kv_chunk_bytes = get_size_bytes(layout_desc.shapes, layout_desc.dtypes)
            logger.info(
                "StagingPool init: kv_chunk_bytes=%d (%.2f MB), "
                "shapes=%s, dtypes=%s, staging_size=%.1f GB, "
                "estimated pool_size=%d",
                kv_chunk_bytes,
                kv_chunk_bytes / (1 << 20),
                layout_desc.shapes,
                layout_desc.dtypes,
                self._staging_size_gb,
                int(self._staging_size_gb * (1 << 30)) // kv_chunk_bytes,
            )

            # Adjust PIN_CHUNK_SIZE
            old_pin = LazyMemoryAllocator.PIN_CHUNK_SIZE
            new_pin = max(old_pin, kv_chunk_bytes)
            if new_pin != old_pin:
                LazyMemoryAllocator.PIN_CHUNK_SIZE = new_pin
                logger.info(
                    "Adjusted LazyMemoryAllocator.PIN_CHUNK_SIZE: %d -> %d "
                    "(kv_chunk_bytes=%d)",
                    old_pin,
                    new_pin,
                    kv_chunk_bytes,
                )

            staging_size_bytes = int(self._staging_size_gb * (1 << 30))
            pool_size = staging_size_bytes // kv_chunk_bytes
            if pool_size < 32:
                logger.warning(
                    "StagingPool pool_size=%d is too small (recommend >= 32), "
                    "staging_size=%.1fGB, kv_chunk_bytes=%d",
                    pool_size,
                    self._staging_size_gb,
                    kv_chunk_bytes,
                )
                pool_size = max(pool_size, 32)

            self._staging_pool = StagingPool(
                pool_size=pool_size,
                buffer_bytes=kv_chunk_bytes,
            )
            CompressedMemoryObj.set_staging_pool(self._staging_pool)

            # Resolve ABO dtype and ratio
            kv_dtype = layout_desc.dtypes[0]
            abo_dtype = resolve_abo_dtype(kv_dtype)
            ratio = self._abo_config.ratio
            if ratio is None:
                ratio = _DEFAULT_RATIO[abo_dtype]

            chunk_size = layout_desc.shapes[0][2]
            block_size = kv_chunk_bytes // (chunk_size * 2)
            self._abo_compress_manager.set_abo_params(abo_dtype, ratio, block_size)

            logger.info(
                "StagingPool created and ABO params set: "
                "abo_dtype=%s, ratio=%d, block_size=%d, kv_dtype=%s",
                abo_dtype,
                ratio,
                block_size,
                kv_dtype,
            )

    def _get_release_stream(
        self,
    ) -> tuple[torch.cuda.Stream, cupy.cuda.ExternalStream]:
        """Get or lazily create the release stream for async staging release.

        Separate from H2D stream to avoid blocking H2D with host callbacks.
        """
        if self._release_stream is None:
            device = torch.cuda.current_device()
            self._release_stream = torch.cuda.Stream(device=device)
            self._cupy_release_stream = cupy.cuda.ExternalStream(
                self._release_stream.cuda_stream
            )
            logger.info("Created ABO release_stream for async staging release")
        return self._release_stream, self._cupy_release_stream

    def submit_release_tasks(
        self,
        h2d_event: torch.cuda.Event,
        memory_objs: list[MemoryObj],
    ) -> None:
        """Schedule async staging release after H2D completes.

        Records an event on h2d_stream, waits for it on release_stream,
        then launches release_staging as host callback on release_stream.
        This avoids blocking h2d_stream with host callbacks.

        Args:
            memory_obj: CompressedMemoryObj whose staging to release.
            h2d_stream: The stream where H2D copy was issued.
        """
        release_stream, cupy_release_stream = self._get_release_stream()

        if h2d_event is not None:
            release_stream.wait_event(h2d_event)

        with torch.cuda.device(release_stream.device):
            for memory_obj in memory_objs:
                if not isinstance(memory_obj, CompressedMemoryObj):
                    continue
                cupy_release_stream.launch_host_func(memory_obj.release_staging, 0)

    @property
    def abo_compress_manager(self) -> ABOCompressManager:
        return self._abo_compress_manager

    @property
    def abo_decompress_manager(self) -> ABODecompressManager:
        return self._abo_decompress_manager

    # --- Overridden methods ---

    def reserve_write(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"],
    ) -> dict[ObjectKey, MemoryObj]:
        """Override: after base reserve_write allocates CompressedMemoryObj,
        eagerly acquire a staging buffer (num_readers=1) for each STORE-path
        obj. If the pool is exhausted for a key, abort that write and drop
        it from the returned dict so the caller will skip D2H for it.

        Keeping staging acquisition here (instead of in ABOMemoryManager.
        allocate) avoids leaking the STORE/L2-load distinction into the L1
        layer: L2->L1 loads go through a different path and never pass
        through this reserve_write.
        """
        result = super().reserve_write(keys, layout_desc, mode)

        failed_keys: list[ObjectKey] = []
        for k, obj in list(result.items()):
            if not isinstance(obj, CompressedMemoryObj):
                continue

            if not obj.try_acquire_staging():
                failed_keys.append(k)
                result.pop(k)

        if failed_keys:
            logger.warning(
                "StagingPool exhausted for %d keys; aborting their writes",
                len(failed_keys),
            )
            self._l1_manager.abort_write(failed_keys)  # type: ignore[attr-defined]

        return result

    def _on_prefetch_l1_hits(
        self,
        l1_read_result: dict[ObjectKey, tuple[L1Error, MemoryObj | None]],
        l1_hit_keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> list[ObjectKey]:
        """Override: trigger early decompress for L1-hit objects."""
        skipped: list[ObjectKey] = []
        to_decompress: list[CompressedMemoryObj] = []
        for i, k in enumerate(l1_hit_keys):
            entry = l1_read_result.get(k)
            obj = entry[1] if entry is not None else None
            if not isinstance(obj, CompressedMemoryObj):
                continue

            if obj.has_staging:
                # Reuse STORE-in-progress staging: no decompress needed
                obj.staging_ref_acquire(extra_count)
                continue

            if not obj.try_acquire_staging(extra_count=extra_count):
                skipped = list(l1_hit_keys[i:])
                break
            to_decompress.append(obj)

        self._abo_decompress_manager.submit_decompress_tasks(to_decompress)  # type: ignore[arg-type]

        return skipped

    def release_prefetch_staging(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> None:
        self._l1_manager.release_prefetch_staging(keys, extra_count=extra_count)  # type: ignore[attr-defined]

    def close(self) -> None:
        """Override: close ABO managers before base close."""
        if self._abo_compress_manager is not None:
            self._abo_compress_manager.close()
        if self._abo_decompress_manager is not None:
            self._abo_decompress_manager.close()
        # Sync release stream before closing
        if self._release_stream is not None:
            self._release_stream.synchronize()
        super().close()
