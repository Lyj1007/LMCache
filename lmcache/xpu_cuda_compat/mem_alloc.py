# SPDX-License-Identifier: Apache-2.0
"""Hugepage pinned host allocation helpers for Kunlun XPU.

Kunlun XPU uses CUDA-compatible host registration APIs for fast H2D/D2H
transfers, but the default Python fallback does not request huge pages.
This module provides THP-backed pinned host allocation for LMCache CPU
buffers when ``local_cpu_use_hugepages`` is enabled.

Host registration is performed in 4 GiB segments (see
``_PIN_SEGMENT_BYTES``) to avoid the Kunlun XPU driver's 32-bit SGL size
overflow in ``memdescCreate`` (``mem_desc.c:282``).  This allows
registering buffers larger than 512 GiB in a single ``_pin_host_memory``
call.
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
try:
    _SMALL_PAGE_SIZE = int(os.sysconf("SC_PAGESIZE"))
except (ValueError, OSError):
    _SMALL_PAGE_SIZE = 4096
_MADV_HUGEPAGE = 14
_PROT_READ_WRITE = 0x3
_MAP_PRIVATE_ANONYMOUS = 0x22
_MAP_HUGETLB = 0x40000
_MAP_HUGE_2MB = 21 << 26  # MAP_HUGE_SHIFT = 26
_SYS_MBIND = 237
_MPOL_BIND = 2

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p,   # addr
    ctypes.c_size_t,   # length
    ctypes.c_int,      # prot
    ctypes.c_int,      # flags
    ctypes.c_int,      # fd
    ctypes.c_long,     # offset
]
_libc.madvise.restype = ctypes.c_int
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.munmap.restype = ctypes.c_int
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

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

# Maximum bytes per single ``xpu_host_register`` / ``cudaHostRegister`` call.
# The Kunlun XPU driver asserts ``SglSize <= 0xffffffffULL`` in
# ``memdescCreate`` (mem_desc.c:282); registering 1024 GiB in a single call
# overflows an internal 32-bit field, while 512 GiB is known to work.  128 GiB
# segments give a 4× safety margin below the verified 512 GiB ceiling and
# keep the number of ioctl round-trips small (8 segments for 1 TiB).
_PIN_SEGMENT_BYTES: int = 128 * 1024 * 1024 * 1024

# Tracks registered segments per base pointer so ``_unpin_host_memory`` can
# unregister each segment individually.  Key is the original base pointer
# passed to ``_pin_host_memory``; value is a list of (segment_ptr, seg_size).
_pinned_segments: dict[int, list[tuple[int, int]]] = {}


def _align_hugepage(size: int) -> int:
    """Round a byte size up to the THP hugepage boundary.

    Args:
        size: Requested allocation size in bytes.

    Returns:
        Size aligned up to the next 2 MiB hugepage boundary.
    """
    return (size + _HUGE_PAGE_SIZE - 1) & ~(_HUGE_PAGE_SIZE - 1)


def _align_smallpage(size: int) -> int:
    """Round a byte size up to the system page size boundary.

    Args:
        size: Requested allocation size in bytes.

    Returns:
        Size aligned up to the next system page boundary.
    """
    ps = _SMALL_PAGE_SIZE
    return (size + ps - 1) & ~(ps - 1)


def _pin_single(ptr: int, size: int) -> None:
    """Register a single contiguous host memory segment for device DMA.

    Args:
        ptr: Host memory pointer (already offset to segment start).
        size: Segment size in bytes.

    Raises:
        RuntimeError: If the host registration API reports an error.
    """
    if _libxpurt is not None:
        ret = _libxpurt.xpu_host_register(
            ctypes.c_void_p(ptr), ctypes.c_size_t(size), ctypes.c_uint(0)
        )
        if ret != 0:
            raise RuntimeError(
                f"xpu_host_register failed: ret={ret}, "
                f"ptr=0x{ptr:x}, size={size}"
            )
        return

    ret = torch.cuda.cudart().cudaHostRegister(ptr, size, 0)
    if ret != 0:
        raise RuntimeError(
            f"cudaHostRegister failed: ret={ret}, "
            f"ptr=0x{ptr:x}, size={size}"
        )


def _unpin_single(ptr: int) -> None:
    """Unregister a single host memory segment.

    Args:
        ptr: Segment pointer previously passed to ``_pin_single``.

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


def _pin_host_memory(ptr: int, size: int) -> None:
    """Register a host allocation for XPU or CUDA DMA.

    Buffers larger than ``_PIN_SEGMENT_BYTES`` are registered in 128 GiB
    segments to avoid the Kunlun XPU driver's 32-bit SGL size overflow
    (``SglSize <= 0xffffffffULL`` assertion in ``memdescCreate``).  Each
    segment is tracked in ``_pinned_segments`` so the corresponding
    ``_unpin_host_memory`` call can unregister them individually.

    Args:
        ptr: Host memory pointer returned by ``mmap``.
        size: Allocation size in bytes to register.

    Raises:
        RuntimeError: If the host registration API reports an error.  Any
            segments already registered before the failure are rolled back.
    """
    if size <= _PIN_SEGMENT_BYTES:
        _pin_single(ptr, size)
        _pinned_segments[ptr] = [(ptr, size)]
        logger.info(
            "Host registration OK: ptr=0x%x, size=%d (%.2f GB), segments=1",
            ptr, size, size / (1024**3),
        )
        return

    segments: list[tuple[int, int]] = []
    offset = 0
    num_segments = (size + _PIN_SEGMENT_BYTES - 1) // _PIN_SEGMENT_BYTES
    logger.info(
        "Host registration started: ptr=0x%x, size=%d (%.2f GB), "
        "segments=%d (%d GiB each)",
        ptr, size, size / (1024**3), num_segments,
        _PIN_SEGMENT_BYTES // (1024**3),
    )
    while offset < size:
        seg_size = min(_PIN_SEGMENT_BYTES, size - offset)
        seg_ptr = ptr + offset
        try:
            _pin_single(seg_ptr, seg_size)
        except RuntimeError:
            logger.error(
                "Host registration failed at offset %d / %d, "
                "rolling back %d segments",
                offset, size, len(segments),
            )
            for sp, _ss in reversed(segments):
                try:
                    _unpin_single(sp)
                except Exception:
                    logger.exception(
                        "Rollback: unregister failed at 0x%x", sp
                    )
            raise
        segments.append((seg_ptr, seg_size))
        offset += seg_size
        if offset % (64 * 1024**3) < _PIN_SEGMENT_BYTES or offset == size:
            logger.info(
                "Host registration progress: %d / %d (%.1f%%)",
                offset, size, offset / size * 100,
            )

    _pinned_segments[ptr] = segments
    logger.info(
        "Host registration complete: ptr=0x%x, size=%d (%.2f GB), "
        "segments=%d",
        ptr, size, size / (1024**3), len(segments),
    )


def _unpin_host_memory(ptr: int) -> None:
    """Unregister a pinned host allocation.

    If the allocation was registered in segments by ``_pin_host_memory``,
    each segment is unregistered individually.  Falls back to a single
    unregister call for pointers not found in ``_pinned_segments``.

    Args:
        ptr: Host memory pointer previously passed to ``_pin_host_memory``.

    Raises:
        RuntimeError: If the host unregister API reports an error.
    """
    segments = _pinned_segments.pop(ptr, None)
    if segments is not None:
        for seg_ptr, _ss in segments:
            _unpin_single(seg_ptr)
        logger.info(
            "Host unregistration complete: ptr=0x%x, segments=%d",
            ptr, len(segments),
        )
        return

    _unpin_single(ptr)


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
    """Allocate hugepage-backed anonymous memory and pin it for device DMA.

    Allocation proceeds in three tiers; the first that succeeds wins:

    1. Explicit hugetlb (``MAP_HUGETLB | MAP_HUGE_2MB``) -- best TLB hit
       rate but requires reserved hugepages in the kernel pool.
    2. Transparent huge pages (THP) via ``madvise(MADV_HUGEPAGE)`` -- relies
       on the khugepaged kernel thread; works without a reserved pool.
    3. Regular small pages -- last-resort fallback so the server can still
       start when the hugepage pool is exhausted and THP cannot deliver
       2 MiB pages due to memory fragmentation. A ``WARN`` is logged and
       the allocation is tracked so callers can detect the degradation.

    Args:
        size: Requested allocation size in bytes.
        numa_id: Optional NUMA node id for best-effort binding before first
            touch.

    Returns:
        Pointer to the pinned host allocation.

    Raises:
        RuntimeError: If all three allocation paths fail, or pin fails.
    """
    hp_size = _align_hugepage(size)

    # Tier 1: explicit hugetlb
    ptr = _libc.mmap(
        None,
        hp_size,
        _PROT_READ_WRITE,
        _MAP_PRIVATE_ANONYMOUS | _MAP_HUGETLB | _MAP_HUGE_2MB,
        -1,
        0,
    )
    if ptr != ctypes.c_void_p(-1).value:
        mmap_size = hp_size
        fault_stride = _HUGE_PAGE_SIZE
        mode = "hugetlb"
        logger.info(
            "Explicit hugetlb mmap OK: size=%d (%.2f GB)",
            mmap_size, mmap_size / (1024**3),
        )
    else:
        hp_errno = ctypes.get_errno()
        logger.info(
            "Explicit hugetlb mmap failed (errno=%d), falling back to THP",
            hp_errno,
        )
        # Tier 2: THP via madvise(MADV_HUGEPAGE)
        ptr = _libc.mmap(
            None,
            hp_size,
            _PROT_READ_WRITE,
            _MAP_PRIVATE_ANONYMOUS,
            -1,
            0,
        )
        if ptr != ctypes.c_void_p(-1).value:
            mmap_size = hp_size
            fault_stride = _HUGE_PAGE_SIZE
            mode = "THP"
        else:
            thp_errno = ctypes.get_errno()
            # Tier 3: regular small pages
            logger.warning(
                "THP mmap failed (errno=%d); falling back to regular "
                "small pages. KV-transfer host buffers will use %d KiB "
                "pages, which increases TLB pressure and may reduce "
                "H2D/D2H throughput.",
                thp_errno, _SMALL_PAGE_SIZE // 1024,
            )
            mmap_size = _align_smallpage(size)
            ptr = _libc.mmap(
                None,
                mmap_size,
                _PROT_READ_WRITE,
                _MAP_PRIVATE_ANONYMOUS,
                -1,
                0,
            )
            if ptr == ctypes.c_void_p(-1).value:
                raise RuntimeError(
                    f"mmap failed at all tiers "
                    f"(hugetlb errno={hp_errno}, "
                    f"THP errno={thp_errno}, "
                    f"smallpage errno={ctypes.get_errno()})"
                )
            fault_stride = _SMALL_PAGE_SIZE
            mode = "smallpage"

    pinned = False
    try:
        if numa_id is not None:
            _bind_numa(ptr, mmap_size, numa_id)
        if mode == "THP":
            _libc.madvise(ctypes.c_void_p(ptr), mmap_size, _MADV_HUGEPAGE)
        # Touch one byte per page (huge or small) to fault the allocation
        # in before registering it for DMA.
        for offset in range(0, mmap_size, fault_stride):
            ctypes.memset(ptr + offset, 0, 1)
        _pin_host_memory(ptr, mmap_size)
        pinned = True
    except Exception:
        if pinned:
            try:
                _unpin_host_memory(ptr)
            except Exception:
                logger.exception(
                    "xpu_host_unregister failed during rollback at 0x%x", ptr
                )
        _libc.munmap(ctypes.c_void_p(ptr), mmap_size)
        raise

    _allocation_sizes[ptr] = mmap_size
    logger.info(
        "XPU pinned host allocation succeeded "
        "(ptr=0x%x, requested_size=%d, mmap_size=%d, numa_id=%s, "
        "hugepage=%s, pin_backend=%s)",
        ptr,
        size,
        mmap_size,
        numa_id,
        mode,
        "xpurt" if _libxpurt is not None else "cuda_cudart",
    )
    if mode == "smallpage":
        logger.warning(
            "Host buffer at 0x%x is backed by small pages (no hugepages "
            "available); expect reduced H2D/D2H throughput.",
            ptr,
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

    Used by the L1 SHM pool host-registration code (XPU offload v2 §10).
    Segmentation is handled internally by ``_pin_host_memory``.

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
