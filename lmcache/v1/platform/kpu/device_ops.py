# SPDX-License-Identifier: Apache-2.0
"""KPU ops backend: torch baseline plus Kunlun-specific memory overrides.

On Kunlun KPU the CUDA-compiled ``lmcache.c_ops`` ``.so`` cannot run, so
:class:`KpuDeviceOps` deliberately stays on the pure torch baseline
(:mod:`lmcache.v1.platform.torch_ops`), whose tensor ops route through
xmlir's CUDA compatibility layer to the KPU hardware.  It overrides only the
ops that need Kunlun-specific handling:

* Hugepage-pinned host allocation (THP + ``xpu_host_register``).
* Pointer-mode H2D / D2H memcpy (PyTorch ``copy_`` instead of ``cudaMemcpy``).

Unlike :class:`~lmcache.v1.platform.cuda.device_ops.CudaDeviceOps`,
:meth:`ensure_native` is a no-op: binding the CUDA extension would pull in
``libcudart`` symbols that are unavailable on Kunlun.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps

logger = init_logger(__name__)


class KpuDeviceOps(DeviceOps):
    """DeviceOps for Kunlun KPU: torch baseline + Kunlun memory overrides."""

    device_type: ClassVar[str] = "kpu"

    def ensure_native(self) -> None:
        """No-op: Kunlun KPU cannot load the CUDA-compiled ``lmcache.c_ops``.

        The torch baseline (routed through xmlir) plus the Kunlun-specific
        overrides below provide the full ``lmcache.c_ops`` surface.
        """
        if self._native_bound:
            return
        self._native_bound = True
        logger.info(
            "KpuDeviceOps: using torch baseline (xmlir) with Kunlun memory "
            "overrides; CUDA native extension is intentionally not bound."
        )

    # ── Ops: memory alloc / free (Kunlun THP + host registration) ──────

    def alloc_hugepage_pinned_ptr(self, size, device_id=0):
        # First Party
        from lmcache.kpu_cuda_compat import mem_alloc

        return mem_alloc.alloc_hugepage_pinned_ptr(size, device_id)

    def free_hugepage_pinned_ptr(self, ptr, size=0):
        # First Party
        from lmcache.kpu_cuda_compat import mem_alloc

        return mem_alloc.free_hugepage_pinned_ptr(ptr, size)

    def alloc_hugepage_pinned_numa_ptr(self, size, numa_id=0):
        # First Party
        from lmcache.kpu_cuda_compat import mem_alloc

        return mem_alloc.alloc_hugepage_pinned_numa_ptr(size, numa_id)

    def free_hugepage_pinned_numa_ptr(self, ptr, size=0):
        # First Party
        from lmcache.kpu_cuda_compat import mem_alloc

        return mem_alloc.free_hugepage_pinned_numa_ptr(ptr, size)

    # ── Ops: pointer-mode memcpy (xmlir copy_ instead of cudaMemcpy) ───

    def lmcache_memcpy_async(self, *args, **kwargs):
        # First Party
        from lmcache.kpu_cuda_compat import mem_ops

        return mem_ops.lmcache_memcpy_async(*args, **kwargs)
