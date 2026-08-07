# SPDX-License-Identifier: Apache-2.0
"""Async memory copy operations for LMCache on Kunlun KPU.

Overrides the torch baseline (:mod:`lmcache.v1.platform.torch_ops`) version to
avoid the ``libcudart`` dependency. On Kunlun KPU with xmlir, ``libcudart.so``
is not available, so the baseline pointer-mode path that uses ``cudaMemcpy``
cannot work. This implementation converts raw pointers to tensor views via
:func:`_tensor_from_ptr` and performs copies using PyTorch tensor ops
(``.copy_()``), which route through xmlir's CUDA compatibility layer to the
KPU hardware.
"""

# Third Party
import torch

# First Party
from lmcache.kpu_cuda_compat.enums import TransferDirection
from lmcache.v1.platform.torch_ops import _tensor_from_ptr


def lmcache_memcpy_async(
    dest: int,
    src: int,
    nbytes: int,
    direction: int,
    host_buffer_offset: int = 0,
    host_buffer_alignments: int = 0,
) -> None:
    """Async memory copy for Kunlun KPU.

    Handles H2D / D2H pointer-based transfers by converting raw pointers to
    tensor views via :func:`_tensor_from_ptr` and using PyTorch's ``.copy_()``
    which routes through xmlir's CUDA compatibility layer.

    Uses ``torch.uint8`` for byte-level copies, matching the semantics of the
    CUDA c_ops ``cudaMemcpy`` implementation.

    Args:
        dest: Destination pointer address.
        src: Source pointer address.
        nbytes: Number of bytes to copy.
        direction: ``TransferDirection`` value (H2D=0, D2H=1).
        host_buffer_offset: Offset into the host buffer.
        host_buffer_alignments: Alignment requirement (must be a power of two).

    Raises:
        ValueError: If ``host_buffer_alignments`` is not a positive power of
            two, or if ``direction`` is unsupported.
    """
    if host_buffer_alignments <= 0 or (
        host_buffer_alignments & (host_buffer_alignments - 1) != 0
    ):
        raise ValueError("host_buffer_alignments must be power of two")

    # Byte-level copy using uint8 -- matches CUDA c_ops cudaMemcpy semantics.
    num_elements = nbytes

    if int(direction) == int(TransferDirection.H2D):
        # Host -> Device: src is CPU pointer, dest is KPU (cuda) pointer.
        # Offset is applied to the host (src) pointer per CUDA c_ops semantics.
        src_view = _tensor_from_ptr(
            src + host_buffer_offset, (num_elements,), torch.uint8, "cpu"
        )
        dest_view = _tensor_from_ptr(dest, (num_elements,), torch.uint8, "cuda")

        # Use a CUDA stream for async behavior (maps to KPU stream via xmlir).
        with torch.cuda.stream(torch.cuda.Stream()):
            dest_view.copy_(src_view, non_blocking=True)

    elif int(direction) == int(TransferDirection.D2H):
        # Device -> Host: src is KPU (cuda) pointer, dest is CPU pointer.
        # Offset is applied to the host (dest) pointer per CUDA c_ops semantics.
        src_view = _tensor_from_ptr(src, (num_elements,), torch.uint8, "cuda")
        dest_view = _tensor_from_ptr(
            dest + host_buffer_offset, (num_elements,), torch.uint8, "cpu"
        )

        with torch.cuda.stream(torch.cuda.Stream()):
            dest_view.copy_(src_view, non_blocking=True)

    else:
        raise ValueError(f"Unsupported direction: {direction}")
