# SPDX-License-Identifier: Apache-2.0
"""
ABO-aware L1MemoryManager subclass.

Overrides ``allocate`` to produce CompressedMemoryObj and
``free`` to release staging buffers before freeing.
"""

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.abo.abo_codec import (
    ABOCodecFactory,
    ABOConfig,
    estimate_compressed_bytes,
)
from lmcache.v1.distributed.abo.compressed_memory_obj import CompressedMemoryObj
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


class ABOMemoryManager(L1MemoryManager):
    """L1MemoryManager with ABO compression support.

    Overrides ``allocate`` to allocate compressed-size space and wrap
    as CompressedMemoryObj.  Overrides ``free`` to release staging
    buffers before freeing underlying memory.
    """

    def __init__(
        self,
        config: L1MemoryManagerConfig,
        abo_config: ABOConfig,
    ):
        super().__init__(config)
        self._abo_config = abo_config
        self._abo_codec = ABOCodecFactory.create_codec(abo_config.codec, abo_config)
        logger.info("ABOMemoryManager initialized with abo_config=%s", abo_config)

    def allocate(
        self,
        layout_desc: MemoryLayoutDesc,
        count: int,
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Allocate compressed MemoryObj (ABO mode).

        Only allocates compressed-size L1 raw_data and wraps as
        CompressedMemoryObj. Staging buffer acquisition is the caller's
        responsibility (e.g. ABOStorageManager.reserve_write for STORE,
        _on_prefetch_l1_hits for RETRIEVE); this method is path-agnostic.
        """
        ratio = self._abo_config.ratio
        compressed_bytes = estimate_compressed_bytes(
            layout_desc.shapes, layout_desc.dtypes, ratio=ratio
        )

        # Build compressed-size layout_desc
        compressed_shape = torch.Size([compressed_bytes])
        compressed_layout = MemoryLayoutDesc(
            shapes=[compressed_shape],
            dtypes=[torch.uint8],
        )

        # Allocate compressed-size space from L1 memory pool
        raw_objects = self._allocator.batched_allocate(
            compressed_layout.shapes, compressed_layout.dtypes, count
        )
        if raw_objects is None:
            return L1Error.OUT_OF_MEMORY, []

        # Wrap as CompressedMemoryObj
        compressed_objs: list[MemoryObj] = []
        for raw_obj in raw_objects:
            compressed_obj = CompressedMemoryObj(
                raw_data=raw_obj.raw_data,
                metadata=raw_obj.meta,
                parent_allocator=raw_obj.parent_allocator,  # type: ignore[attr-defined]  # noqa: E501,F821
                staging_tensor=None,
                original_shapes=layout_desc.shapes,
                original_dtypes=layout_desc.dtypes,
            )
            compressed_obj.meta.shape = layout_desc.shapes[0]
            compressed_obj.meta.dtype = layout_desc.dtypes[0]
            compressed_obj.meta.shapes = layout_desc.shapes
            compressed_obj.meta.dtypes = layout_desc.dtypes
            compressed_objs.append(compressed_obj)

            # Invalidate original raw_obj to avoid double-free
            raw_obj.parent_allocator = None  # type: ignore[attr-defined]

        return L1Error.SUCCESS, compressed_objs

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """Free memory objects, releasing staging buffers first."""
        for obj in mem_objs:
            if isinstance(obj, CompressedMemoryObj) and obj.has_staging:
                obj.release_staging()
        return super().free(mem_objs)
