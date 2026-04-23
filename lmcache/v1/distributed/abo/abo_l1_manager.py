# SPDX-License-Identifier: Apache-2.0
"""
ABO-aware L1Manager subclass.

Overrides ``_create_memory_manager`` to return ABOMemoryManager,
and adds ``abort_write`` for compress failure recovery.
"""

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.abo.abo_codec import ABOConfig
from lmcache.v1.distributed.abo.abo_memory_manager import ABOMemoryManager
from lmcache.v1.distributed.abo.compressed_memory_obj import CompressedMemoryObj
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager, l1_mgr_synchronized
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


class ABOL1Manager(L1Manager):
    """L1Manager with ABO compression support.

    Overrides ``_create_memory_manager`` to produce ABOMemoryManager.
    Adds ``abort_write`` for compress failure recovery.
    """

    def __init__(self, config: L1ManagerConfig, abo_config: ABOConfig):
        self._abo_config = abo_config
        super().__init__(config)

    def _create_memory_manager(self, config: L1ManagerConfig) -> L1MemoryManager:
        """Override: create ABOMemoryManager with codec config."""
        return ABOMemoryManager(config.memory_config, abo_config=self._abo_config)

    def release_prefetch_staging(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> None:
        with self._lock:
            for k in keys:
                entry = self._objects.get(k)
                if entry is None:
                    continue
                obj = entry.memory_obj
                if isinstance(obj, CompressedMemoryObj) and obj.has_staging:
                    obj.release_staging(extra_count)

    @l1_mgr_synchronized
    def abort_write(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1Error]:
        """Abort a write: unlock write lock, remove key, free memory.

        Used when ABO compress fails after reserve_write but before finish_write.
        """
        need_to_free: list[MemoryObj] = []
        ret: dict[ObjectKey, L1Error] = {}
        aborted_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = L1Error.KEY_NOT_EXIST
                continue

            if not entry.write_lock.is_locked():
                logger.warning(
                    "ABOL1Manager: abort_write on non-write-locked key %s",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

            entry.write_lock.unlock()
            need_to_free.append(entry.memory_obj)
            del self._objects[key]
            ret[key] = L1Error.SUCCESS
            aborted_keys.append(key)

        self._memory_manager.free(need_to_free)

        for listener in self._registered_listeners:
            listener.on_l1_keys_deleted_by_manager(aborted_keys)

        if aborted_keys:
            logger.info(
                "ABOL1Manager: aborted write for %d keys",
                len(aborted_keys),
            )

        return ret
