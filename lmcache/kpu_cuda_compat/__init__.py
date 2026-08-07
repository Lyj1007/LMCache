# SPDX-License-Identifier: Apache-2.0
"""KPU-compatible helpers backing the ``lmcache.c_ops`` surface.

On Kunlun KPU with xmlir, the CUDA-compiled ``.so`` (``lmcache.c_ops``)
cannot run, so :class:`~lmcache.v1.platform.kpu.device_ops.KpuDeviceOps`
overrides only the ops that need KPU-specific handling:

* H2D / D2H pointer-mode memcpy (uses PyTorch tensor ``copy_`` instead of
  ``cudaMemcpy``) -- see :mod:`lmcache.kpu_cuda_compat.mem_ops`.
* Hugepage-pinned host allocation backed by THP + ``xpu_host_register``
  -- see :mod:`lmcache.kpu_cuda_compat.mem_alloc`.

Everything else falls through to the torch baseline
(:mod:`lmcache.v1.platform.torch_ops`), whose tensor ops work via xmlir's
CUDA compatibility layer.
"""

# First Party
from lmcache.kpu_cuda_compat.enums import GPUKVFormat, TransferDirection
from lmcache.kpu_cuda_compat.mem_alloc import (
    alloc_hugepage_pinned_numa_ptr,
    alloc_hugepage_pinned_ptr,
    free_hugepage_pinned_numa_ptr,
    free_hugepage_pinned_ptr,
    has_kpu_runtime,
    host_register,
    host_unregister,
)
from lmcache.kpu_cuda_compat.mem_ops import lmcache_memcpy_async

__all__ = [
    "GPUKVFormat",
    "TransferDirection",
    "alloc_hugepage_pinned_numa_ptr",
    "alloc_hugepage_pinned_ptr",
    "free_hugepage_pinned_numa_ptr",
    "free_hugepage_pinned_ptr",
    "has_kpu_runtime",
    "host_register",
    "host_unregister",
    "lmcache_memcpy_async",
]
