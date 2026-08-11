# SPDX-License-Identifier: Apache-2.0
"""KLX_XPU host-memory pinning backed by the Kunlun runtime (libxpurt).

Delegates to :mod:`lmcache.klx_xpu_cuda_compat.mem_alloc`, which registers host
buffers via ``xpu_host_register`` (falling back to ``cudaHostRegister`` via
torch cudart when libxpurt is absent), including the 512 GiB segmentation
required to sidestep the Kunlun driver's 32-bit SGL size overflow.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend

logger = init_logger(__name__)


class KlxXpuPinMemoryBackend(PinMemoryBackend):
    """Pin host memory for KLX_XPU DMA via the Kunlun host-registration API."""

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Register a host region for KLX_XPU DMA.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.
            size: Size in bytes of the region to pin.
            flags: Unused; kept for signature compatibility with the base.

        Returns:
            True if registration succeeded, False otherwise.
        """
        del flags
        try:
            # First Party
            from lmcache.klx_xpu_cuda_compat.mem_alloc import host_register

            host_register(ptr, size)
            return True
        except Exception as exc:
            logger.warning(
                "KLX_XPU host_register failed for ptr=%#x size=%d: %s",
                ptr,
                size,
                exc,
            )
            return False

    def unpin_memory(self, ptr: int) -> bool:
        """Unregister a previously pinned host region.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.

        Returns:
            True if unregistration succeeded, False otherwise.
        """
        try:
            # First Party
            from lmcache.klx_xpu_cuda_compat.mem_alloc import host_unregister

            host_unregister(ptr)
            return True
        except Exception as exc:
            logger.warning("KLX_XPU host_unregister failed for ptr=%#x: %s", ptr, exc)
            return False

    @property
    def is_pin_supported(self) -> bool:
        """Whether KLX_XPU host pinning is supported on this system.

        True when either the Kunlun runtime (libxpurt) was loaded or the CUDA
        cudart fallback is importable.
        """
        try:
            # First Party
            from lmcache.klx_xpu_cuda_compat.mem_alloc import has_klx_xpu_runtime

            if has_klx_xpu_runtime():
                return True
            # Third Party
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False
