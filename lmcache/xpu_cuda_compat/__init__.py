# SPDX-License-Identifier: Apache-2.0
"""XPU-compatible backend providing the same API as ``lmcache.c_ops``.

On Kunlun XPU with xmlir, the CUDA-compiled ``.so`` (``lmcache.c_ops``) cannot
run, so this module overrides only the ops that need XPU-specific handling:

* H2D / D2H pointer-mode memcpy (uses PyTorch tensor ``copy_`` instead of
  ``cudaMemcpy``).
* Hugepage-pinned host allocation backed by THP + ``xpu_host_register``.

Everything else falls through to ``python_ops_fallback`` (PyTorch tensor ops
that work via xmlir's CUDA compatibility layer).
"""

# First Party
from lmcache.xpu_cuda_compat.enums import GPUKVFormat, TransferDirection
from lmcache.xpu_cuda_compat.mem_alloc import (
    alloc_hugepage_pinned_numa_ptr,
    alloc_hugepage_pinned_ptr,
    free_hugepage_pinned_numa_ptr,
    free_hugepage_pinned_ptr,
)
from lmcache.xpu_cuda_compat.mem_ops import lmcache_memcpy_async

try:
    # First Party
    from lmcache.c_ops import PageBufferShapeDesc, multi_layer_block_kv_transfer
except ImportError:
    pass
