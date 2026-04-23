# SPDX-License-Identifier: Apache-2.0

"""
ABO decompress manager (CUDA stream + launch_host_func mode).

Manages async decompression in the PREFETCH path:
1. Batch-submit decompress tasks via cupy launch_host_func on _decompress_stream
2. Each decompress callback runs sync decompress (codec.decompress with numpy buffers)
3. After decompress, record event for GPU-side sync (current_stream.wait_event)
4. After H2D, release staging buffer via launch_host_func callback

Key constraint:
  cudaLaunchHostFunc callbacks need GIL. Never call stream.synchronize()
  or event.synchronize() from the same thread that holds GIL while
  host callbacks are pending on that stream.

Timing design:
  _decompress_stream:
    launch_host_func(decomp0) → decomp_event0 → launch_host_func(decomp1) → ...
  high_priority stream (H2D):
    wait(decomp_event0) → H2D chunk0 → launch_host_func(release_staging0) → ...
    wait(decomp_event1) → H2D chunk1 → launch_host_func(release_staging1) → ...
"""

# Standard
from typing import TYPE_CHECKING, Optional

# Third Party
import cupy
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.abo.compressed_memory_obj import CompressedMemoryObj

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


class ABODecompressManager:
    """Manages async decompression in the RETRIEVE path (stream mode).

    Uses a dedicated CUDA stream + cupy launch_host_func to schedule
    sync decompress as host callbacks. The callback executes in the
    CUDA driver's callback thread (which acquires GIL automatically).

    H2D loop uses .tensor property's wait_event for GPU-side sync.
    After H2D, staging release is also done via launch_host_func.
    """

    def __init__(
        self,
        codec,
        device: Optional[torch.device] = None,
    ):
        """Initialize decompress manager.

        Args:
            codec: abokvpress.HuffmanCodec instance (from ABOCodecFactory).
            device: CUDA device (None = current device).
        """
        self._codec = codec

        # Lazily created CUDA stream + cupy wrapper
        self._decompress_stream: Optional[torch.cuda.Stream] = None
        self._cupy_decompress_stream: Optional[cupy.cuda.ExternalStream] = None
        self._device = device

    def _get_decompress_stream(
        self,
    ) -> tuple[torch.cuda.Stream, cupy.cuda.ExternalStream]:
        """Get or create the dedicated decompress CUDA stream + cupy wrapper."""
        if self._decompress_stream is None:
            device = self._device or torch.cuda.current_device()
            self._decompress_stream = torch.cuda.Stream(device=device)
            self._cupy_decompress_stream = cupy.cuda.ExternalStream(
                self._decompress_stream.cuda_stream
            )
        return self._decompress_stream, self._cupy_decompress_stream

    def submit_decompress_tasks(
        self,
        memory_objs: list["MemoryObj"],
    ) -> None:
        """Batch-submit async decompress tasks for a list of MemoryObj.

        Used by the prefetch path. The caller (e.g.
        ``ABOStorageManager._on_prefetch_l1_hits``) is responsible for
        acquiring a fresh staging buffer for each obj before passing it
        here; objs that already hold staging (STORE compress-in-progress,
        reuse case) MUST be filtered out by the caller.
        """
        decompress_stream, cupy_decompress_stream = self._get_decompress_stream()

        for obj in memory_objs:
            if not isinstance(obj, CompressedMemoryObj):
                continue

            # Capture references for the closure
            _obj = obj
            _codec = self._codec
            _staging = obj._staging_tensor

            def _decompress_callback(_arg, _o=_obj, _c=_codec, _s=_staging):
                """Host callback: sync decompress raw_data → staging.

                Runs in CUDA driver's callback thread (GIL acquired by cupy).
                Args:
                    _arg: User data from cupy launch_host_func (unused).
                """
                if not _o.has_staging:
                    return
                try:
                    # codec.decompress(dst_np, dst_size, src_np, src_size)
                    # Pass full raw_data; codec self-describes block boundaries
                    dst_np = _s.numpy()
                    src_np = _o.raw_data.numpy()
                    result = _c.decompress(dst_np, dst_np.nbytes, src_np, src_np.nbytes)
                    if not result.success:
                        logger.warning("ABO decompress failed: %s", result.error_msg)
                        _o.mark_decompress_failed()
                    else:
                        _o.mark_decompress_done()
                except Exception as e:
                    logger.warning("Decompress callback failed: %s", e)
                    _o.mark_decompress_failed()

            # Switch to decompress_stream's device context to ensure
            # launch_host_func and event.record run in the correct CUDA context
            # (MLA: any TP worker's thread may call this, but stream is fixed)
            with torch.cuda.device(decompress_stream.device):
                # Register host callback on decompress_stream
                cupy_decompress_stream.launch_host_func(_decompress_callback, None)

                # Record decompress done event on decompress_stream
                decompress_event = torch.cuda.Event()
                decompress_event.record(decompress_stream)
            obj.setup_decompress(decompress_event)

    def close(self) -> None:
        """Clean up resources.

        NOTE: Do NOT call decompress_stream.synchronize() here — it would
        deadlock if any host callbacks are pending.
        """
        pass
