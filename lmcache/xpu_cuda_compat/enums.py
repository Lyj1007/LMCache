# SPDX-License-Identifier: Apache-2.0
"""Enumerations matching lmcache.c_ops (mem_kernels.cuh).

Values must be identical to the CUDA version so that LMCache's GPU connector
code works without modification.
"""

# Standard
from enum import IntEnum


class TransferDirection(IntEnum):
    """Direction of KV cache transfer."""

    H2D = 0
    D2H = 1


class GPUKVFormat(IntEnum):
    """Physical memory layout of GPU KV cache.

    Symbol reference:
        NL: number of layers
        NB: number of blocks/pages
        BS: block/page size
        NBBS: block/page buffer size = NB * BS
        NH: number of heads
        HS: head size
        TWO: 2
        ONE: 1

    ``_`` means a dimension within the same tensor.
    ``_X_`` means a dimension across a list.
    """

    NB_NL_TWO_BS_NH_HS = 0
    NL_X_TWO_NB_BS_NH_HS = 1
    NL_X_NB_TWO_BS_NH_HS = 2
    NL_X_NB_BS_HS = 3
    TWO_X_NL_X_NBBS_NH_HS = 4
    NL_X_NBBS_ONE_HS = 5
    NL_X_TWO_NB_NH_BS_HS = 6
    NL_X_NB_TWO_NH_BS_HS = 7
    NB_NL_TWO_NH_BS_HS = 8
    TWO_X_NL_X_NB_BS_NH_HS = 9
