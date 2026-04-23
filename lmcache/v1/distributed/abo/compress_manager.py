# SPDX-License-Identifier: Apache-2.0

"""
ABO compress task manager (CUDA stream + launch_host_func mode).

Manages per-chunk async compression in the STORE path:
1. cupy_stream.launch_host_func(compress_fn) — register sync compress
   as host callback on compress_stream
2. compress_done_event.record(compress_stream) — mark compress done

The host callback runs in CUDA driver's callback thread (GIL acquired
by cupy). D2H completion is ensured by compress_stream.wait_event(d2h_event)
before the callback is registered, so the callback can directly execute
sync compress, mark done, and release staging.

Timing design:
  main_stream:       D2H chunk0 → d2h_event0.record → D2H chunk1 → ...
  compress_stream:   wait(d2h_event0) → launch_host_func(compress0)
                     → compress_done_event0.record → ...
  (callback thread): compress0() → mark_done + release staging
"""

# Standard
from typing import TYPE_CHECKING, Callable, Optional

# Third Party
import cupy
import torch

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.abo.compressed_memory_obj import CompressedMemoryObj
    from lmcache.v1.distributed.api import ObjectKey

logger = init_logger(__name__)


class ABOCompressManager:
    """Manages async compression in the STORE path (stream mode).

    Uses a dedicated CUDA stream + cupy launch_host_func to schedule
    sync compress as a host callback. The callback executes in the
    CUDA driver's callback thread (which acquires GIL automatically).

    All operations (compress, staging release, mark done) are registered
    on the same compress_stream via launch_host_func, so FIFO ordering
    guarantees correct sequencing without extra synchronization primitives.
    """

    def __init__(
        self,
        codec,
        device: Optional[torch.device] = None,
        on_compress_failed: Optional[Callable[["ObjectKey"], None]] = None,
    ):
        """Initialize compress manager.

        Args:
            codec: abokvpress.HuffmanCodec instance (from ABOCodecFactory).
            device: CUDA device (None = current device).
            on_compress_failed: Callback on compress failure.
        """
        self._codec = codec
        self._abo_dtype: Optional[str] = None
        self._ratio: Optional[int] = None
        self._block_size: int = 128 * 1024
        self._device = device
        self._on_compress_failed = on_compress_failed

        # Lazily created CUDA stream (needs CUDA context)
        self._compress_stream: Optional[torch.cuda.Stream] = None
        self._cupy_compress_stream: Optional[cupy.cuda.ExternalStream] = None

    def set_abo_params(
        self, abo_dtype: str, ratio: int, block_size: int = 128 * 1024
    ) -> None:
        """Set ABO dtype and compression ratio.

        Called from StorageManager._ensure_staging_pool once the KV dtype
        is known from layout_desc.

        Args:
            abo_dtype: ABO dtype string (e.g. "bf16", "fp8e4m3").
            ratio: Compression ratio (20-32).
            block_size: Block size for abokvpress compress (bytes).
        """
        self._abo_dtype = abo_dtype
        self._ratio = ratio
        self._block_size = block_size

    def _get_compress_stream(
        self,
    ) -> tuple[torch.cuda.Stream, cupy.cuda.ExternalStream]:
        """Get or create the dedicated compress CUDA stream + cupy wrapper."""
        if self._compress_stream is None:
            device = self._device or torch.cuda.current_device()
            self._compress_stream = torch.cuda.Stream(device=device)
            self._cupy_compress_stream = cupy.cuda.ExternalStream(
                self._compress_stream.cuda_stream
            )
        return self._compress_stream, self._cupy_compress_stream

    def submit_per_chunk_compress(
        self,
        d2h_event: torch.cuda.Event,
        obj_key: "ObjectKey",
        compressed_obj: "CompressedMemoryObj",
    ) -> torch.cuda.Event:
        """Submit per-chunk compression on the compress stream.

        Called in _cb_store_gpu_copy's for loop, after each chunk D2H.
        Does not block the main thread.

        Flow:
        1. cupy_stream.launch_host_func(compress_fn) — register sync compress
           as host callback on compress_stream
        2. compress_done_event.record(compress_stream) — mark compress done

        The host callback (compress_fn) runs in CUDA driver's callback thread:
        - Executes sync compress (codec.compress with numpy buffers)
        - Marks compress done
        - Releases staging buffer

        D2H completion is ensured by compress_stream.wait_event(d2h_event)
        before the callback is registered. The main thread is never blocked.

        Args:
            d2h_event: CUDA event recorded after this chunk's D2H.
            obj_key: Object key (for error logging).
            compressed_obj: CompressedMemoryObj instance.

        Returns:
            compress_done_event: CUDA event recorded after compress callback.
                Use current_stream.wait_event() for GPU-side sync.
        """
        assert self._abo_dtype is not None and self._ratio is not None, (
            "abo_dtype and ratio must be set via set_abo_params before compress"
        )
        assert compressed_obj._staging_tensor is not None, (
            "Compress requires staging_tensor != None"
        )
        assert compressed_obj.raw_data is not None, "Compress requires raw_data != None"

        compress_stream, cupy_compress_stream = self._get_compress_stream()

        # Capture references for the closure
        _obj_key = obj_key
        _compressed_obj = compressed_obj
        _on_compress_failed = self._on_compress_failed
        _codec = self._codec
        _abo_dtype = self._abo_dtype
        _ratio = self._ratio
        _block_size = self._block_size

        def _compress_callback(_arg):
            """Host callback: sync compress → mark done → release staging.

            Runs in CUDA driver's callback thread (GIL acquired by cupy).
            D2H completion is guaranteed by compress_stream.wait_event(d2h_event)
            before this callback is registered.
            Args:
                _arg: User data from cupy launch_host_func (unused).
            """
            try:
                staging_tensor = _compressed_obj._staging_tensor
                raw_data = _compressed_obj.raw_data

                if staging_tensor is None or raw_data is None:
                    logger.error(
                        "Compress aborted: staging_tensor or raw_data is None (key=%s)",
                        _obj_key,
                    )
                    _compressed_obj.mark_compress_failed()
                    if _on_compress_failed is not None:
                        try:
                            _on_compress_failed(_obj_key)
                        except Exception:
                            pass
                    return

                # Sync compress: codec.compress(dst, capacity, src, size, ratio, dtype)
                dst_np = raw_data.numpy()
                src_np = staging_tensor.numpy()
                result = _codec.compress(
                    dst_np,
                    dst_np.nbytes,
                    src_np,
                    src_np.nbytes,
                    _ratio,
                    _abo_dtype,
                    # block_size=_block_size,
                )

                if not result.success:
                    logger.error(
                        "ABO compress failed (key=%s): %s",
                        _obj_key,
                        result.error_msg,
                    )
                    _compressed_obj.mark_compress_failed()
                    if _on_compress_failed is not None:
                        try:
                            _on_compress_failed(_obj_key)
                        except Exception:
                            pass
                    return

                logger.debug(
                    "Compress done: key=%s, actual_compressed_size=%d bytes, "
                    "buffer_size=%d bytes",
                    _obj_key,
                    result.result_size,
                    _compressed_obj.raw_data.nbytes,
                )
            except Exception as e:
                logger.error("Compress callback failed (key=%s): %s", _obj_key, e)
                _compressed_obj.mark_compress_failed()
                if _on_compress_failed is not None:
                    try:
                        _on_compress_failed(_obj_key)
                    except Exception:
                        pass
            finally:
                # Release staging via ref-counted release_staging.
                # If a concurrent RETRIEVE holds a ref, staging won't
                # be freed until that ref is also released.
                _compressed_obj.release_staging()

        # Ensure compress_stream waits for D2H to finish before callback.
        # This replaces d2h_event.synchronize() inside the callback,
        # avoiding CPU-side blocking in the callback thread.
        with torch.cuda.device(compress_stream.device):
            compress_stream.wait_event(d2h_event)

            # Register host callback on compress_stream
            cupy_compress_stream.launch_host_func(_compress_callback, None)

            # Record compress done event on compress_stream
            # (for GPU-side sync via current_stream.wait_event)
            compress_done_event = torch.cuda.Event()
            compress_done_event.record(compress_stream)

        # Set the CUDA event on compressed_obj for GPU-side sync
        compressed_obj.set_compress_event(compress_done_event)

        return compress_done_event

    def close(self) -> None:
        """Clean up resources.

        NOTE: Do NOT call compress_stream.synchronize() here — it would
        deadlock if any host callbacks are pending. The server process
        will terminate and clean up automatically.
        """
        pass
