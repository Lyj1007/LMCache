# SPDX-License-Identifier: Apache-2.0
"""Hugepage pinned host allocation helpers for Kunlun XPU.

Kunlun XPU uses CUDA-compatible host registration APIs for fast H2D/D2H
transfers, but the default Python fallback does not request huge pages.
This module provides THP-backed pinned host allocation for LMCache CPU
buffers when ``local_cpu_use_hugepages`` is enabled.

The L1 SHM pool used by the XPU offload v2 design (see
``docs/design/v1/multiprocess/xpu_offload_v2_design.md`` §10) is registered
via the same ``_pin_host_memory`` helper in 4 GiB segments at server
startup.
"""

# Standard
from typing import Optional
import ctypes
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_HUGE_PAGE_SIZE = 2 * 1024 * 1024
_MADV_HUGEPAGE = 14
_PROT_READ_WRITE = 0x3
_MAP_PRIVATE_ANONYMOUS = 0x22
_SYS_MBIND = 237
_MPOL_BIND = 2

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p

# Two well-known install paths for libxpurt; first-found wins. Empty on
# non-XPU hosts, in which case ``_pin_host_memory`` falls back to
# ``cudaHostRegister`` via torch.cuda.cudart().
_xpurt_paths = [
    "/root/miniconda/envs/python310_torch29_cuda/xcudart/lib/libxpurt.so",
    "/root/miniconda/envs/python310_torch29_cuda/lib/python3.10/"
    "site-packages/torch_xmlir/xre/so/libxpurt.so",
]
_libxpurt: Optional[ctypes.CDLL] = None
for _path in _xpurt_paths:
    if os.path.exists(_path):
        _libxpurt = ctypes.CDLL(_path, use_errno=True)
        break

# Tracks size used at allocation time so ``free`` can ``munmap`` the same
# region without forcing callers to remember the aligned size.
_allocation_sizes: dict[int, int] = {}


def _align_hugepage(size: int) -> int:
    """Round a byte size up to the THP hugepage boundary.

    Args:
        size: Requested allocation size in bytes.

    Returns:
        Size aligned up to the next 2 MiB hugepage boundary.
    """
    return (size + _HUGE_PAGE_SIZE - 1) & ~(_HUGE_PAGE_SIZE - 1)


def _pin_host_memory(ptr: int, size: int) -> None:
    """Register a host allocation for XPU or CUDA DMA.

    Args:
        ptr: Host memory pointer returned by ``mmap``.
        size: Allocation size in bytes to register.

    Raises:
        RuntimeError: If the host registration API reports an error.
    """
    if _libxpurt is not None:
        ret = _libxpurt.xpu_host_register(
            ctypes.c_void_p(ptr), ctypes.c_size_t(size), ctypes.c_uint(0)
        )
        if ret != 0:
            raise RuntimeError(f"xpu_host_register failed: {ret}")
        return

    ret = torch.cuda.cudart().cudaHostRegister(ptr, size, 0)
    if ret != 0:
        raise RuntimeError(f"cudaHostRegister failed: {ret}")


def _unpin_host_memory(ptr: int) -> None:
    """Unregister a pinned host allocation.

    Args:
        ptr: Host memory pointer previously passed to ``_pin_host_memory``.

    Raises:
        RuntimeError: If the host unregister API reports an error.
    """
    if _libxpurt is not None:
        ret = _libxpurt.xpu_host_unregister(ctypes.c_void_p(ptr))
        if ret != 0:
            raise RuntimeError(f"xpu_host_unregister failed: {ret}")
        return

    ret = torch.cuda.cudart().cudaHostUnregister(ptr)
    if ret != 0:
        raise RuntimeError(f"cudaHostUnregister failed: {ret}")


def _bind_numa(ptr: int, size: int, numa_id: int) -> None:
    """Best-effort bind an anonymous mapping to one NUMA node.

    Args:
        ptr: Start address of the memory mapping.
        size: Mapping size in bytes.
        numa_id: NUMA node id to bind; negative values disable binding.
    """
    if numa_id < 0:
        return
    nodemask = ctypes.c_ulong(1 << numa_id)
    _libc.syscall(
        _SYS_MBIND,
        ctypes.c_void_p(ptr),
        ctypes.c_size_t(size),
        _MPOL_BIND,
        ctypes.byref(nodemask),
        ctypes.c_ulong(64),
        0,
    )


def _alloc_thp_pinned(size: int, numa_id: Optional[int] = None) -> int:
    """Allocate THP-advised anonymous memory and pin it for device DMA.

    Args:
        size: Requested allocation size in bytes.
        numa_id: Optional NUMA node id for best-effort binding before first
            touch.

    Returns:
        Pointer to the pinned host allocation.

    Raises:
        RuntimeError: If mapping, registration, or cleanup fails.
    """
    aligned_size = _align_hugepage(size)
    ptr = _libc.mmap(
        None,
        aligned_size,
        _PROT_READ_WRITE,
        _MAP_PRIVATE_ANONYMOUS,
        -1,
        0,
    )
    if ptr == ctypes.c_void_p(-1).value:
        raise RuntimeError(f"mmap failed: {ctypes.get_errno()}")

    pinned = False
    try:
        if numa_id is not None:
            _bind_numa(ptr, aligned_size, numa_id)
        _libc.madvise(ctypes.c_void_p(ptr), aligned_size, _MADV_HUGEPAGE)
        # Touch one byte per hugepage to fault the THP allocation in before
        # registering it for DMA. ``cudaHostRegister`` on a non-faulted
        # page can silently degrade to small-page mappings.
        for offset in range(0, aligned_size, _HUGE_PAGE_SIZE):
            ctypes.memset(ptr + offset, 0, 1)
        _pin_host_memory(ptr, aligned_size)
        pinned = True
    except Exception:
        if pinned:
            try:
                _unpin_host_memory(ptr)
            except Exception:
                logger.exception(
                    "xpu_host_unregister failed during rollback at 0x%x", ptr
                )
        _libc.munmap(ctypes.c_void_p(ptr), aligned_size)
        raise

    _allocation_sizes[ptr] = aligned_size
    logger.info(
        "XPU THP pinned host allocation succeeded "
        "(ptr=0x%x, requested_size=%d, aligned_size=%d, numa_id=%s, "
        "pin_backend=%s)",
        ptr,
        size,
        aligned_size,
        numa_id,
        "xpurt" if _libxpurt is not None else "cuda_cudart",
    )
    return ptr


def _free_thp_pinned(ptr: int, size: int = 0) -> None:
    """Unpin and unmap a THP-backed host allocation.

    Args:
        ptr: Pointer returned by ``_alloc_thp_pinned``.
        size: Original requested size in bytes, used only if the pointer was
            not tracked in this process.

    Raises:
        RuntimeError: If host unregister or ``munmap`` fails.
    """
    aligned_size = _allocation_sizes.pop(ptr, _align_hugepage(size))
    try:
        _unpin_host_memory(ptr)
    finally:
        ret = _libc.munmap(ctypes.c_void_p(ptr), aligned_size)
        if ret != 0:
            raise RuntimeError(f"munmap failed: {ctypes.get_errno()}")


def alloc_hugepage_pinned_ptr(size: int, device_id: int = 0) -> int:
    """Allocate THP-backed pinned host memory and return its pointer.

    Args:
        size: Requested allocation size in bytes.
        device_id: Kept for API compatibility with CUDA c_ops.

    Returns:
        Pointer to the start of the allocation.
    """
    del device_id
    return _alloc_thp_pinned(size)


def free_hugepage_pinned_ptr(ptr: int, size: int = 0) -> None:
    """Free memory allocated by ``alloc_hugepage_pinned_ptr``.

    Args:
        ptr: Pointer returned by ``alloc_hugepage_pinned_ptr``.
        size: Original requested allocation size in bytes.
    """
    _free_thp_pinned(ptr, size)


def alloc_hugepage_pinned_numa_ptr(size: int, numa_id: int = 0) -> int:
    """Allocate THP-backed pinned host memory with best-effort NUMA binding.

    Args:
        size: Requested allocation size in bytes.
        numa_id: NUMA node to bind the allocation to when supported.

    Returns:
        Pointer to the start of the allocation.
    """
    return _alloc_thp_pinned(size, numa_id)


def free_hugepage_pinned_numa_ptr(ptr: int, size: int = 0) -> None:
    """Free memory allocated by ``alloc_hugepage_pinned_numa_ptr``.

    Args:
        ptr: Pointer returned by ``alloc_hugepage_pinned_numa_ptr``.
        size: Original requested allocation size in bytes.
    """
    _free_thp_pinned(ptr, size)


def host_register(ptr: int, size: int) -> None:
    """Register an externally-mapped host buffer for device DMA.

    Used by the L1 SHM pool host-registration code (XPU offload v2 §10) to
    register POSIX-SHM-backed segments in 4 GiB chunks at server startup.

    Args:
        ptr: Host memory pointer to register.
        size: Allocation size in bytes.

    Raises:
        RuntimeError: If the host registration API reports an error.
    """
    _pin_host_memory(ptr, size)


def host_unregister(ptr: int) -> None:
    """Unregister a host buffer previously registered with ``host_register``.

    Args:
        ptr: Host memory pointer previously passed to ``host_register``.

    Raises:
        RuntimeError: If the host unregister API reports an error.
    """
    _unpin_host_memory(ptr)


def has_xpu_runtime() -> bool:
    """Return True if libxpurt was located and loaded at import time."""
    return _libxpurt is not None
