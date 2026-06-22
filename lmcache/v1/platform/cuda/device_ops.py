# SPDX-License-Identifier: Apache-2.0
"""CUDA ops backend: bulk-bind the compiled ``lmcache.c_ops`` extension.

:class:`CudaDeviceOps` calls :meth:`bind_native` in :meth:`ensure_native`
to layer the compiled CUDA extension on top of the torch baseline.  If the
extension is missing, a warning is logged and the instance stays on the
torch fallback (soft-fail, same as XPU).

On Kunlun XPU the CUDA API is emulated by ``torch_xmlir``, so this spec is
the one that gets selected; there the pure-Python ``lmcache.xpu_cuda_compat``
shim is bound in place of the compiled extension.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps

logger = init_logger(__name__)


class CudaDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "cuda"

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        self._native_bound = True  # set early to prevent repeated attempts

        # Kunlun XPU reports as CUDA (torch_xmlir supplies the CUDA API
        # compatibility layer), but the CUDA-compiled ``.so`` cannot run on
        # XPU hardware.  Bind the pure-Python compat shim instead: it
        # overrides only the H2D/D2H memcpy and the hugepage pinned
        # allocator, leaving every other op on the torch baseline.
        # First Party
        from lmcache import is_kunlun_xpu

        if is_kunlun_xpu():
            # First Party
            import lmcache.xpu_cuda_compat as compat

            self.bind_native(compat)
            return

        try:
            # First Party
            import lmcache.c_ops as native
        except ImportError:
            logger.warning(
                "lmcache.c_ops compiled extension not found; "
                "CudaDeviceOps stays on the torch baseline for all ops."
            )
            return
        self.bind_native(native)
