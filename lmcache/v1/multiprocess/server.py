# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from functools import partial
from itertools import islice
from typing import Callable, Generator
import argparse
import threading
import time

# Third Party
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.distributed.abo.abo_storage_manager import ABOStorageManager
from lmcache.v1.distributed.abo.compressed_memory_obj import CompressedMemoryObj
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.storage_manager import PrefetchHandle, StorageManager
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.config import (
    ObservabilityConfig,
    add_observability_args,
    init_observability,
    parse_args_to_observability_config,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    add_mp_server_args,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.gpu_context import GPUCacheContext
from lmcache.v1.multiprocess.mp_runtime_plugin_launcher import (
    MPRuntimePluginLauncher,
)
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.periodic_thread import (
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)
from lmcache.v1.platform.cache_context import create_cache_context
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


# Helper functions
def compute_extra_count(
    tp_size: int,
    world_size: int,
) -> int:
    """Compute extra count for MLA multi-reader locking.

    Non-MLA: each TP worker owns a distinct KV shard,
      so each ObjectKey is retrieved by exactly 1
      worker -> extra_count = 0.
    MLA: TP does not split KV caches, all TP workers
      share the same object. vLLM passes world_size
      already divided by tp_size (e.g. world_size=1
      for TP=4 PP=1), so ipc_keys_to_object_keys
      only produces 1 ObjectKey per chunk.  All TP
      workers retrieve that same ObjectKey, hence
      extra_count = tp_size - 1.

    Detection: tp > world_size means MLA (world_size
    was divided by tp on the vLLM side).

    Fallback: old vLLM (<= 0.8.5) does not send
    tp_size (defaults to 1); we fall back to
    world_size which gives extra_count = 0
    (safe but may under-lock for MLA).

    TODO: world_size currently carries an overloaded
    meaning (total ranks for non-MLA vs total/tp for
    MLA). Consider a dedicated field in the future.

    Args:
        tp_size: Tensor-parallel size from the client.
        world_size: World size from the cache key.

    Returns:
        Number of extra count (0 for non-MLA).
    """
    tp = tp_size if tp_size > 1 else world_size
    return tp - 1 if tp > world_size else 0


def get_layout_desc(gpu_context: GPUCacheContext, num_tokens: int) -> MemoryLayoutDesc:
    """Get the memory layout description for a given GPU context and number of tokens.

    Supports multiple KV layer groups with different shapes and dtypes.

    Args:
        gpu_context: The GPU cache context containing the KV cache information.
        num_tokens: The number of tokens to determine the layout for.

    Returns:
        MemoryLayoutDesc: The memory layout description containing shapes and dtypes.
    """
    num_groups = gpu_context.kv_layer_groups_manager.num_groups
    shapes = [
        gpu_context.get_kv_buffer_shape(num_tokens, group_idx)
        for group_idx in range(num_groups)
    ]
    dtypes = [
        gpu_context.kv_layer_groups_manager.kv_layer_groups[group_idx].dtype
        for group_idx in range(num_groups)
    ]
    return MemoryLayoutDesc(shapes=shapes, dtypes=dtypes)


def batched_iteration(lst: list, batch_size: int) -> Generator[tuple, None, None]:
    """Utility function to iterate over a list in batches.

    Args:
        lst: The list to iterate over.
        batch_size: The size of each batch.

    Yields:
        Batches of the list as tuples.
    """
    if batch_size < 1:
        raise ValueError("batch size must be at least one")
    it = iter(lst)
    while batch := tuple(islice(it, batch_size)):
        yield batch


@dataclass
class _PrefetchJob:
    handle: PrefetchHandle
    world_size: int
    request_id: str


@dataclass
class _PendingLookupState:
    """State saved between SYNC_LOOKUP and RETRIEVE.

    SYNC_LOOKUP stores the ``job_id`` here so that the worker adapter
    can poll ``QUERY_PREFETCH_STATUS_WITH_REQ_ID(request_id)`` to
    wait for L2-to-L1 data transfer to complete before sending
    RETRIEVE.
    """

    job_id: int
    """Prefetch job ID returned by ``lookup()``."""

    created_at: float = 0.0
    """Monotonic timestamp when this state was created."""


# Main class for the mp cache engine
class MPCacheEngine:
    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
    ):
        # GPU ID -> KV cache tensors
        self.gpu_contexts: dict[int, GPUCacheContext] = {}

        # GPU ID -> (model name, world size) as metadata
        # NOTE: This is mainly for determining the layout desc during prefetch
        # We assume that if the (model name, world size) is the same, then
        # the layout desc returned by the gpu context is the same.
        self.gpu_context_meta: dict[int, tuple[str, int]] = {}

        # chunk size
        self.chunk_size = chunk_size

        # Lock for clear() to avoid concurrent storage manager mutations
        self.lock = threading.Lock()

        # storage manager
        self._abo_enabled = storage_manager_config.abo_config.enable
        self.storage_manager = self._create_storage_manager(storage_manager_config)

        # Token hasher and session manager for token-based operations
        self.token_hasher = TokenHasher(
            chunk_size=chunk_size, hash_algorithm=hash_algorithm
        )
        self.session_manager = SessionManager(self.token_hasher)

        # EventBus for observability
        self._event_bus = get_event_bus()

        # Prefetch job tracking for two-phase lookup
        # TODO: implement periodic cleanup of stale _prefetch_jobs entries
        # for crash resilience (e.g., client calls lookup but never queries)
        self._prefetch_jobs: dict[int, _PrefetchJob] = {}
        self._request_to_job_id: dict[str, int] = {}
        self._next_prefetch_job_id: int = 0
        self._prefetch_job_lock = threading.Lock()

        self._setup_metrics()

        # Pending lookup state for SYNC_LOOKUP -> RETRIEVE handoff.
        # Keyed by request_id; written by sync_lookup(), consumed by
        # retrieve().
        self._pending_lookups: dict[str, _PendingLookupState] = {}
        self._pending_lookups_lock = threading.Lock()

        # Background TTL scan thread (PeriodicThread)
        self._stale_cleaners: list[tuple[str, Callable[[], int]]] = []
        self._scan_thread = create_periodic_thread(
            name="mp-engine-ttl-scan",
            interval=self._SCAN_INTERVAL,
            execute_fn=self._ttl_scan_execute,
            level=ThreadLevel.MEDIUM,
            init_wait=self._SCAN_INTERVAL,
        )
        self._scan_thread.start()

    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
        layout_hints: LayoutHints,
    ) -> None:
        """
        Registers the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id (int): The GPU instance ID (such as PID).
            kv_caches (KVCache): The KV cache tensor wrappers from vLLM.
            model_name (str): The name of the model associated with this KV cache.
            world_size (int): The world size associated with this KV cache.
            layout_hints: See :class:`LayoutHints`.  Forwarded to
                :class:`GPUCacheContext` for GPU KV format detection.
        """
        gpu_context = create_cache_context(
            kv_caches,
            self.chunk_size,
            layout_hints=layout_hints or None,
        )
        self.gpu_contexts[instance_id] = gpu_context
        self.gpu_context_meta[instance_id] = (model_name, world_size)
        logger.info(
            "Registered KV cache for GPU ID %d: "
            "layers=%d, block_size=%d, num_blocks=%d, "
            "is_mla=%s, format=%s, dtype=%s, model=%s, "
            "world_size=%d",
            instance_id,
            gpu_context.num_layers,
            gpu_context.block_size,
            gpu_context.num_blocks,
            gpu_context.is_mla,
            gpu_context.gpu_kv_format_name,
            gpu_context.dtype,
            model_name,
            world_size,
        )

        # Eagerly initialize ABO staging pool on first register
        if self._abo_enabled and isinstance(self.storage_manager, ABOStorageManager):
            layout_desc = get_layout_desc(gpu_context, self.chunk_size)
            self.storage_manager.init_staging_pool(layout_desc)

    def unregister_kv_cache(self, instance_id: int) -> None:
        """
        Unregisters the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id (int): The GPU instance ID (such as PID).
        """
        if instance_id in self.gpu_contexts:
            del self.gpu_contexts[instance_id]
            del self.gpu_context_meta[instance_id]
            logger.info("Unregistered KV cache for GPU ID %d", instance_id)
            torch.cuda.empty_cache()
        else:
            logger.warning("No KV cache found for GPU ID %d to unregister", instance_id)

    @_lmcache_nvtx_annotate
    def store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """
        Stores the GPU KV cache blocks to CPU.

        Args:
            key (IPCCacheEngineKey): The IPC key for the KV cache blocks.
                Must have worker_id != None (worker store operation).
            instance_id (int): The GPU instance ID (such as PID).
            gpu_block_ids (list[int]): The GPU block IDs to store.
            event_ipc_handle (bytes): The IPC handle of the event to wait on.

        Returns:
            tuple[bytes, bool]: The first element is the IPC handle of the event
                that signals the completion of the store operation. The second
                element indicates whether the store operation was successful.
        """
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        chunk_hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(key.start, key.end)
        ]

        st = time.perf_counter()

        assert key.worker_id is not None, "Must store with worker_id != None"
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        assert instance_id in self.gpu_contexts, (
            f"KV cache not registered for GPU ID {instance_id}"
        )
        gpu_context = self.gpu_contexts[instance_id]

        blocks_per_chunk = self.chunk_size // gpu_context.block_size

        with (
            torch.cuda.device(gpu_context.device),
            torch.cuda.stream(gpu_context.stream),
        ):
            event = torch.cuda.Event(interprocess=True)

            # Stage all block_ids to GPU once before the loop
            all_block_ids_gpu = gpu_context.stage_block_ids(gpu_block_ids)

            # Wait for vLLM to finish
            vllm_event = torch.cuda.Event.from_ipc_handle(
                gpu_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=gpu_context.stream)

            self._event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={"device": str(gpu_context.device)},
                ),
            )

            layout_desc = get_layout_desc(gpu_context, self.chunk_size)
            reserved_dict = self.storage_manager.reserve_write(
                obj_keys, layout_desc, "new"
            )

            # NOTE: Store is not batched because some obj_keys may be
            # skipped (not in reserved_dict), making block_ids
            # non-contiguous. Batching would require torch.cat to
            # reassemble block_ids, negating the benefit.
            num_groups = gpu_context.kv_layer_groups_manager.num_groups
            for idx, obj_key in enumerate(obj_keys):
                if obj_key in reserved_dict:
                    memory_obj = reserved_dict[obj_key]
                else:
                    continue

                chunk_block_ids_gpu = all_block_ids_gpu[
                    idx * blocks_per_chunk : (idx + 1) * blocks_per_chunk
                ]

                # Copy from GPU paged buffer to tmp buffer, then to CPU — per group
                for group_idx in range(num_groups):
                    tmp_buffer = gpu_context.get_tmp_chunk_gpu_buffer(group_idx)
                    group_kv_pointers = gpu_context.get_group_kv_pointers(group_idx)
                    lmc_ops.multi_layer_block_kv_transfer(
                        group_kv_pointers,
                        [tmp_buffer.data_ptr()],
                        chunk_block_ids_gpu,
                        gpu_context.device,
                        lmc_ops.TransferDirection.D2H,
                        gpu_context.get_shape_desc(group_idx),
                        self.chunk_size,
                        gpu_context.gpu_kv_format_,
                        0,
                    )

                # Store is not batched, so we always use chunk_idx=0 (single slot)
                lmcache_memcpy_async_d2h(
                    gpu_context.get_tmp_gpu_buffer_flat(chunk_idx=0), memory_obj
                )

                # ABO: per-chunk event + async compress trigger
                if self._abo_enabled and isinstance(memory_obj, CompressedMemoryObj):
                    d2h_event = torch.cuda.Event()
                    d2h_event.record()
                    self.storage_manager.abo_compress_manager.submit_per_chunk_compress(  # type: ignore[attr-defined]  # noqa: E501,F821
                        d2h_event, obj_key, memory_obj
                    )

            event.record()

        self.gpu_contexts[instance_id].cupy_stream.launch_host_func(
            self.storage_manager.finish_write,
            list(reserved_dict.keys()),
        )

        self._event_bus.publish_on_stream(
            self.gpu_contexts[instance_id].cupy_stream,
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=key.request_id,
                metadata={
                    "stored_count": len(reserved_dict),
                    "device": str(gpu_context.device),
                },
            ),
        )

        ed = time.perf_counter()
        elapsed = ed - st
        session.store_time += elapsed
        total_chunks = len(obj_keys)
        stored_chunks = len(reserved_dict)
        session.store_chunks += stored_chunks
        skipped_chunks = total_chunks - stored_chunks
        if stored_chunks:
            logger.info(
                "STORE[%s]: %d tokens (%d chunks) in %.3fs, skipped=%d, gpu=%s",
                key.request_id,
                stored_chunks * self.chunk_size,
                stored_chunks,
                elapsed,
                skipped_chunks,
                gpu_context.device,
            )
        elif skipped_chunks:
            logger.debug(
                "STORE[%s]: all %d chunks skipped (already cached), gpu=%s",
                key.request_id,
                skipped_chunks,
                gpu_context.device,
            )
        return event.ipc_handle(), True

    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        """
        Retrieves the CPU KV cache and put into GPU blocks.

        Args:
            key (IPCCacheEngineKey): The IPC key for the KV cache blocks.
                Must have worker_id != None (worker retrieve operation).
            instance_id (int): The GPU instance ID (such as PID).
            gpu_block_ids (list[int]): The GPU block IDs to retrieve into.
            event_ipc_handle (bytes): The IPC handle of the event to wait on.
            skip_first_n_tokens (int): Number of tokens to skip writing at
                the start of the retrieve range. This avoids overwriting
                APC-shared GPU blocks that may be read concurrently by other
                requests.

        Returns:
            tuple[bytes, bool]: The first element is the IPC handle of the event
                that signals the completion of the retrieve operation. The second
                element indicates whether the key was successfully retrieved.
        """
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        chunk_hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(key.start, key.end)
        ]

        st = time.perf_counter()

        assert key.worker_id is not None, "Must retrieve with worker_id != None"
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        assert instance_id in self.gpu_contexts, (
            f"KV cache not registered for GPU ID {instance_id}"
        )
        gpu_context = self.gpu_contexts[instance_id]

        self._event_bus.publish_on_stream(
            gpu_context.cupy_stream,
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata={"device": str(gpu_context.device)},
            ),
        )

        blocks_per_chunk = self.chunk_size // gpu_context.block_size

        def _retrieve_loop(keys: list[ObjectKey], memory_objs: list[MemoryObj]) -> None:
            _BATCH_SIZE = gpu_context.max_batch_size
            num_groups = gpu_context.kv_layer_groups_manager.num_groups
            for batch_idx, memory_obj_batch in enumerate(
                batched_iteration(memory_objs, batch_size=_BATCH_SIZE)
            ):
                batch_len = len(memory_obj_batch)
                chunk_start = batch_idx * self.chunk_size * _BATCH_SIZE
                chunk_end = chunk_start + self.chunk_size * batch_len

                effective_start = max(chunk_start, skip_first_n_tokens)
                if effective_start >= chunk_end:
                    # Entire batch is within APC range, skip it
                    if self._abo_enabled:
                        self.storage_manager.submit_release_tasks(  # type: ignore[attr-defined]
                            None, memory_obj_batch
                        )
                    continue

                skip_tokens_in_chunk = max(
                    0,
                    min(
                        effective_start - chunk_start,
                        self.chunk_size * batch_len - 1,
                    ),
                )
                if skip_tokens_in_chunk % gpu_context.block_size != 0:
                    logger.error(
                        "skip_first_n_tokens (%d) is not aligned to block_size (%d), "
                        "rounding down from %d tokens to %d blocks",
                        skip_first_n_tokens,
                        gpu_context.block_size,
                        skip_tokens_in_chunk,
                        skip_tokens_in_chunk // gpu_context.block_size,
                    )
                skip_blocks_in_chunk = skip_tokens_in_chunk // gpu_context.block_size

                start_chunk_id = batch_idx * _BATCH_SIZE
                end_chunk_id = start_chunk_id + batch_len
                chunk_block_ids_gpu = all_block_ids_gpu[
                    start_chunk_id * blocks_per_chunk : end_chunk_id * blocks_per_chunk
                ]

                try:
                    for chunk_idx, memory_obj in enumerate(memory_obj_batch):
                        lmcache_memcpy_async_h2d(
                            memory_obj,
                            gpu_context.get_tmp_gpu_buffer_flat(chunk_idx=chunk_idx),
                        )
                except Exception as e:
                    raise e
                finally:
                    if self._abo_enabled:
                        # Record a single event on H2D stream for the whole
                        # batch, then schedule staging release for each obj.
                        h2d_event = torch.cuda.Event()
                        h2d_event.record()
                        self.storage_manager.submit_release_tasks(  # type: ignore[attr-defined]
                            h2d_event, memory_obj_batch
                        )

                for group_idx in range(num_groups):
                    tmp_buffers = gpu_context.get_tmp_chunk_gpu_buffer_batched(
                        batch_len, group_idx
                    )
                    group_kv_pointers = gpu_context.get_group_kv_pointers(group_idx)

                    lmc_ops.multi_layer_block_kv_transfer(
                        group_kv_pointers,
                        [tb.data_ptr() for tb in tmp_buffers],
                        chunk_block_ids_gpu,
                        gpu_context.device,
                        lmc_ops.TransferDirection.H2D,
                        gpu_context.get_shape_desc(group_idx),
                        self.chunk_size,
                        gpu_context.gpu_kv_format_,
                        skip_blocks_in_chunk,
                    )

        with (
            torch.cuda.device(gpu_context.device),
            torch.cuda.stream(gpu_context.stream),
        ):
            # Stage all block_ids to GPU once before the loop
            all_block_ids_gpu = gpu_context.stage_block_ids(gpu_block_ids)

            event = torch.cuda.Event(interprocess=True)

            prefetched_keys: list[ObjectKey] = []
            retrieve_succeeded = False
            try:
                with self.storage_manager.read_prefetched_results(
                    obj_keys
                ) as memory_objs:
                    if not memory_objs or len(memory_objs) != len(obj_keys):
                        logger.error("Some keys not found during retrieve!")
                        return event.ipc_handle(), False

                    prefetched_keys = obj_keys[: len(memory_objs)]
                    _retrieve_loop(obj_keys, memory_objs)
                # Only set True when with-block exits normally
                retrieve_succeeded = True
            except Exception:
                logger.exception("Cannot retrieve keys due to exception")
                return event.ipc_handle(), False
            finally:
                event.record()
                if retrieve_succeeded:
                    gpu_context.cupy_stream.launch_host_func(
                        self.storage_manager.finish_read_prefetched,
                        prefetched_keys,
                    )
                self._event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": len(prefetched_keys),
                            "device": str(gpu_context.device),
                        },
                    ),
                )

        tokens_retrieved = len(obj_keys) * self.chunk_size
        session.update_retrieved_range(key.start, key.end)
        ed = time.perf_counter()
        elapsed = ed - st
        session.retrieve_time += elapsed
        logger.info(
            "RETRIEVE[%s]: %d tokens (%d chunks) in %.3fs, skip_first_n=%d, gpu=%s",
            key.request_id,
            tokens_retrieved,
            len(obj_keys),
            elapsed,
            skip_first_n_tokens,
            gpu_context.device,
        )

        return event.ipc_handle(), True

    def _find_layout_desc(
        self,
        model_name: str,
        world_size: int,
    ) -> MemoryLayoutDesc | None:
        """Find layout desc from a matching GPU context.

        Returns:
            The layout descriptor, or None if no context
            matches (model_name, world_size).
        """
        for gpu_id, (m, w) in self.gpu_context_meta.items():
            if m == model_name and w == world_size:
                return get_layout_desc(
                    self.gpu_contexts[gpu_id],
                    self.chunk_size,
                )
        return None

    def lookup(
        self,
        key: IPCCacheEngineKey,
        tp_size: int,
    ) -> int:
        """Submit a prefix lookup and return a prefetch job ID.

        Hashes the key, submits a prefetch task to the storage manager,
        and returns a job ID that can be polled via query_prefetch_status.

        Args:
            key: Cache key with request_id embedded.

        Returns:
            Prefetch job ID for polling via query_prefetch_status.
        """
        st = time.perf_counter()
        model_name, world_size = key.model_name, key.world_size

        # Record total tokens on the session for hit-rate tracking
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        session.lookup_ipc_key = key
        session.total_tokens = len(key.token_ids)

        self._event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id=key.request_id,
            )
        )

        layout_desc = self._find_layout_desc(model_name, world_size)
        if layout_desc is None:
            logger.error(
                "No GPU context found for model %s with world size %d during lookup!",
                model_name,
                world_size,
            )
            return self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        l1_prefix_hit_count=0,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                )
            )

        extra_count = compute_extra_count(tp_size, world_size)

        # Compute chunk hashes for all full chunks
        chunk_hashes = self.token_hasher.compute_chunk_hashes(list(key.token_ids))
        if not chunk_hashes:
            return self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        l1_prefix_hit_count=0,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                )
            )

        # Publish lookup event via EventBus for observability subscribers
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP,
                session_id=key.request_id,
                metadata={
                    "request_id": key.request_id,
                    "chunk_hashes": chunk_hashes,
                    "model_name": model_name,
                    "chunk_size": self.chunk_size,
                    "seq_len": len(key.token_ids),
                    "dtypes": [str(d) for d in layout_desc.dtypes],
                    "shapes": [list(s) for s in layout_desc.shapes],
                },
            )
        )

        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        handle = self.storage_manager.submit_prefetch_task(
            obj_keys,
            layout_desc,
            extra_count=extra_count,
            external_request_id=key.request_id,
        )
        job_id = self._register_prefetch_job(
            _PrefetchJob(
                handle=handle,
                world_size=key.world_size,
                request_id=key.request_id,
            )
        )
        elapsed = time.perf_counter() - st
        session.lookup_time += elapsed
        session.lookup_chunks += len(obj_keys)
        logger.info(
            "LOOKUP[%s]: %d chunks submitted in %.3fs, "
            "job_id=%d, model=%s, world_size=%d",
            key.request_id,
            len(obj_keys),
            elapsed,
            job_id,
            model_name,
            world_size,
        )
        return job_id

    def _register_prefetch_job(self, job: _PrefetchJob) -> int:
        with self._prefetch_job_lock:
            job_id = self._next_prefetch_job_id
            self._next_prefetch_job_id += 1
            self._prefetch_jobs[job_id] = job
            self._request_to_job_id[job.request_id] = job_id
        return job_id

    def query_prefetch_lookup_hits(
        self,
        prefetch_job_id: int,
    ) -> int | None:
        """Query the number of hits for a prefetch request before it's finished

        Returns:
            The number of hits for the prefetched keys if the lookup is all done.
            None if the lookup phase is still in progress, or the prefetch is
            already completed and consumed by query_prefetch_status, or the
            prefetch_job_id is invalid.
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(prefetch_job_id)

        if job is None:
            logger.warning(
                "Prefetch job %d not found (already completed or invalid)",
                prefetch_job_id,
            )
            return None

        found_count = self.storage_manager.query_prefetch_lookup_hits(job.handle)
        if found_count is None:
            return None

        found_count = found_count // job.world_size
        return found_count

    def query_prefetch_status(
        self,
        prefetch_job_id: int,
    ) -> int | None:
        """Poll the status of a prefetch job.

        Returns the chunk count when the prefetch is complete, or None
        if it is still in progress.  The job entry is automatically
        removed once a non-None result is returned (exactly-once
        semantics).

        Args:
            prefetch_job_id: Job ID returned by lookup().

        Returns:
            Chunk count (int) when done, None if still in progress
            or the job ID is unknown.
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(prefetch_job_id)
        if job is None:
            logger.warning(
                "Prefetch job %d not found (already completed or invalid)",
                prefetch_job_id,
            )
            return None

        st = time.perf_counter()
        found_count = self.storage_manager.query_prefetch_status(job.handle)
        if found_count is None:
            return None

        # NOTE(Kuntai): this assumes two things:
        # 1. the world size is the same between keys
        # 2. the lookup sort the keys in prefix order and breaks at the
        #    first failure
        found_count = found_count // job.world_size

        self._event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id=job.request_id,
                metadata={"found_count": found_count},
            )
        )
        logger.info(
            "QUERY_PREFETCH[job=%d]: found=%d in %.3fs",
            prefetch_job_id,
            found_count,
            time.perf_counter() - st,
        )

        return found_count

    def query_prefetch_status_with_req_id(self, request_id: str) -> int | None:
        """Query prefetch status by external request_id.

        Same semantics as :meth:`query_prefetch_status` but keyed by
        the external ``request_id`` instead of the server-internal
        ``job_id``.  Intended for the worker adapter which does not
        have access to the job_id.

        Non-destructive peek (``pop=False``); cleanup is handled by
        :meth:`end_session`.

        Args:
            request_id: The external request ID.

        Returns:
            Chunk count (int) when done, ``None`` if still in progress
            or no SYNC_LOOKUP was issued for this request.
        """
        with self._pending_lookups_lock:
            pending = self._pending_lookups.get(request_id)
        if pending is None:
            return None

        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(pending.job_id)
        if job is None:
            return None

        found_count = self.storage_manager.query_prefetch_status(job.handle, pop=False)
        if found_count is None:
            return None

        return found_count // job.world_size

    @_lmcache_nvtx_annotate
    def sync_lookup(
        self,
        key: IPCCacheEngineKey,
        tp_size: int,
    ) -> int:
        """Synchronous prefix lookup — returns hit count directly.

        Internally calls :meth:`lookup` to submit a prefetch request
        to the background prefetch loop, then polls
        :meth:`query_prefetch_lookup_hits` until the lookup phase
        (L1 prefix scan + L2 existence check) completes.

        When there are hits, saves a :class:`_PendingLookupState` so
        that RETRIEVE workers can wait for the full prefetch (L2→L1
        data transfer) to finish via :meth:`query_prefetch_status`
        before reading from L1.

        Args:
            key: Cache key with request_id embedded.
            tp_size: Tensor-parallel size for MLA multi-reader locking.

        Returns:
            Number of matched chunks (prefix hit count).
        """
        # NOTE: lookup() already publishes MP_LOOKUP_PREFETCH_START,
        # so we do not emit it here to avoid duplicate events.
        st = time.perf_counter()

        # Step 1: submit to the existing background prefetch loop
        job_id = self.lookup(key, tp_size)

        # Step 2: poll until the lookup phase finishes
        while True:
            found_count = self.query_prefetch_lookup_hits(job_id)
            if found_count is not None:
                break
            time.sleep(0.005)  # 5 ms

        # Record total tokens on the session for hit-rate tracking
        session = self.session_manager.get_or_create(key.request_id)
        session.total_tokens = len(key.token_ids)

        # Step 3: set up RETRIEVE coordination
        if found_count > 0:
            # Save pending state so QUERY_PREFETCH_STATUS_WITH_REQ_ID
            # can check completion.
            with self._pending_lookups_lock:
                self._pending_lookups[key.request_id] = _PendingLookupState(
                    job_id=job_id,
                    created_at=time.monotonic(),
                )
        else:
            # No hits — no RETRIEVE will be issued. Clean up the
            # prefetch job and pop the prefetch controller result
            # directly (end_session can't do it because we're removing
            # the tracking entries here).
            with self._prefetch_job_lock:
                job = self._prefetch_jobs.pop(job_id, None)
                self._request_to_job_id.pop(key.request_id, None)
            if job is not None and job.handle.prefetch_request_id != -1:
                self.storage_manager.query_prefetch_status(job.handle, pop=True)

        logger.info(
            "SYNC_LOOKUP[%s]: found_count=%d, world_size=%d, elapsed=%.3fs",
            key.request_id,
            found_count,
            key.world_size,
            time.perf_counter() - st,
        )

        return found_count

    def free_lookup_locks(
        self,
        key: IPCCacheEngineKey,
        tp_size: int,
    ) -> None:
        """Release read locks acquired during lookup.

        Hashes are computed only for chunks in ``[start, end)`` to avoid
        unnecessary work on tokens outside that range.
        ``start`` and ``end`` must be aligned to ``chunk_size``; it is the
        caller's responsibility to align the boundaries as desired.

        Computes the extra reader count from ``tp_size`` and
        ``world_size`` the same way :meth:`lookup` does, so
        the correct number of locks is released.

        Args:
            key: Cache key whose read locks should be released.
            tp_size: Tensor-parallel size for MLA
                multi-reader locking.
        """
        chunk_hashes = self.token_hasher.compute_chunk_hashes(
            list(key.token_ids), start=key.start, end=key.end
        )
        if not chunk_hashes:
            return
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        extra_count = compute_extra_count(tp_size, key.world_size)

        if self._abo_enabled:
            self.storage_manager.release_prefetch_staging(  # type: ignore[attr-defined]
                obj_keys, extra_count=extra_count
            )

        self.storage_manager.finish_read_prefetched(obj_keys, extra_count=extra_count)

    def delete_chunks(
        self,
        key: IPCCacheEngineKey,
        tp_size: int,
    ) -> None:
        """Force-evict the chunks identified by ``key`` from L1 cache.

        Used by the multi-server scheduler adapter to repair hit
        mismatch: when one LMCache server reports more matched chunks
        than another for the same prefix, the over-hit tail on the
        leading server is deleted so that subsequent LOOKUPs return a
        consistent (aligned) hit count across all backing servers.

        Best-effort semantics: chunks that are still under read or
        write lock are silently skipped (see :meth:`L1Manager.delete`).
        Caller is expected to issue ``FREE_LOOKUP_LOCKS`` for the same
        ``[start, end)`` range first so that the read locks acquired
        during ``LOOKUP`` are released before this call.

        Hashes are computed only for chunks in ``[start, end)``;
        ``start``/``end`` must be aligned to the chunk size (caller
        responsibility, same as :meth:`free_lookup_locks`).

        Args:
            key: Cache key whose chunks should be deleted.
            tp_size: Tensor-parallel size for MLA multi-reader
                locking.  Currently unused for delete itself but
                kept for protocol symmetry with FREE_LOOKUP_LOCKS
                so that callers can reuse the same key/tp_size pair.
        """
        del tp_size  # reserved for future use; symmetry with free_lookup_locks
        chunk_hashes = self.token_hasher.compute_chunk_hashes(
            list(key.token_ids), start=key.start, end=key.end
        )
        if not chunk_hashes:
            return
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)
        self.storage_manager.delete(obj_keys)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def ping(self) -> bool:
        """
        Respond to a ping request.

        Returns:
            bool: Always True.
        """
        return True

    def get_chunk_size(self) -> int:
        """
        Returns the chunk size used for KV cache operations.

        Returns:
            int: The chunk size.
        """
        return self.chunk_size

    def end_session(self, request_id: str) -> None:
        """Remove the session and all associated prefetch state.

        Args:
            request_id: The request ID whose session should be removed.
        """
        st = time.perf_counter()
        session = self.session_manager.get(request_id)
        if session is None:
            logger.warning(
                "END_SESSION[%s]: session not found, skipping",
                request_id,
            )
            return
        hit_rate = (
            round(session.retrieved_tokens / session.total_tokens, 4)
            if session.total_tokens > 0
            else 0.0
        )
        logger.info(
            "END_SESSION[%s]: total_tokens=%d, "
            "retrieved_tokens=%d, hit_rate=%.2f%%, "
            "lookup=%d chunks/%.3fs, "
            "retrieve=%d chunks/%.3fs, "
            "store=%d chunks/%.3fs",
            request_id,
            session.total_tokens,
            session.retrieved_tokens,
            hit_rate * 100,
            session.lookup_chunks,
            session.lookup_time,
            session.retrieve_chunks,
            session.retrieve_time,
            session.store_chunks,
            session.store_time,
        )
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_END_SESSION,
                metadata={"request_id": request_id},
            )
        )
        session = self.session_manager.remove(request_id)
        if session is None:
            logger.warning("Session %s not found, skipping touch", request_id)
            return
        if session.lookup_ipc_key is None:
            logger.warning(
                "Session %s has no lookup ipc key, skipping touch", request_id
            )
            return

        chunk_hashes = [TokenHasher.hash_to_bytes(h) for h in session.get_hashes(0)]
        obj_keys = ipc_key_to_object_keys(session.lookup_ipc_key, chunk_hashes)
        # unified touch of all keys, which include retrieved and stored keys
        # TODO(chunxiaozheng): when l2 is enabled, the prefetched keys from l2 are temp
        #  and will be deleted after finish_read_prefetched, when we touch all keys,
        #  these keys has been deleted and will not be touched.
        self.storage_manager.touch_l1_keys(obj_keys)

        # Clean up prefetch job (pop=True to consume the result from
        # the prefetch controller, preventing memory leaks).
        with self._prefetch_job_lock:
            job_id = self._request_to_job_id.pop(request_id, None)
            job = self._prefetch_jobs.pop(job_id, None) if job_id is not None else None
        if job is not None and job.handle.prefetch_request_id != -1:
            self.storage_manager.query_prefetch_status(job.handle, pop=True)

        # Clean up pending lookup state (SYNC_LOOKUP path)
        with self._pending_lookups_lock:
            self._pending_lookups.pop(request_id, None)
        logger.info(
            "END_SESSION[%s]: cleanup completed in %.3fs",
            request_id,
            time.perf_counter() - st,
        )

    def report_status(self) -> dict:
        """Return a status dict for the entire cache engine."""
        sm = self.storage_manager.report_status()

        gpu_context_meta: dict[str, dict] = {}
        for gpu_id, meta in self.gpu_context_meta.items():
            entry: dict = {
                "model_name": meta[0],
                "world_size": meta[1],
            }
            ctx = self.gpu_contexts.get(gpu_id)
            if ctx is not None:
                entry["kv_cache_layout"] = {
                    "num_layers": ctx.num_layers,
                    "block_size": ctx.block_size,
                    "hidden_dim_sizes": str(ctx.hidden_dim_sizes),
                    "dtype": str(ctx.dtype),
                    "is_mla": ctx.is_mla,
                    "num_blocks": ctx.num_blocks,
                    "gpu_kv_format": ctx.gpu_kv_format_name,
                    "gpu_kv_shape": ctx.gpu_kv_shape,
                    "gpu_kv_concrete_shape": ctx.concrete_gpu_kv_shape,
                    "attention_backend": ctx.attention_backend,
                    "cache_size_per_token": ctx.cache_size_per_token(),
                }
            gpu_context_meta[str(gpu_id)] = entry

        hit_stats = self.session_manager.report_hit_stats()

        # Prefetch job details for debugging stale/stuck jobs
        with self._prefetch_job_lock:
            prefetch_job_ids = list(self._prefetch_jobs.keys())
            pending_request_ids = list(self._request_to_job_id.keys())

        with self._pending_lookups_lock:
            pending_lookup_request_ids = list(self._pending_lookups.keys())

        registry = PeriodicThreadRegistry.get_instance()

        return {
            "is_healthy": sm["is_healthy"],
            "engine_type": self.__class__.__name__,
            "chunk_size": self.chunk_size,
            "hash_algorithm": self.token_hasher.hash_algorithm_name,
            "registered_gpu_ids": list(self.gpu_contexts.keys()),
            "gpu_context_meta": gpu_context_meta,
            "active_sessions": self.session_manager.active_count(),
            "active_prefetch_jobs": len(prefetch_job_ids),
            "prefetch_job_ids": prefetch_job_ids[:50],
            "pending_request_ids": pending_request_ids[:50],
            "pending_lookup_count": len(pending_lookup_request_ids),
            "pending_lookup_request_ids": (pending_lookup_request_ids[:50]),
            "next_prefetch_job_id": self._next_prefetch_job_id,
            "hit_stats": hit_stats,
            "storage_manager": sm,
            "periodic_threads": registry.get_summary(),
        }

    def report_block_allocations(self, records: list[BlockAllocationRecord]) -> None:
        """Publish vLLM block allocation records to the EventBus.

        Args:
            records: List of BlockAllocationRecord with per-request
                block and token allocation deltas.
        """
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_BLOCK_ALLOCATION,
                metadata={"records": records},
            )
        )

    def debug(self) -> str:
        return "OK"

    def clear(self) -> None:
        """
        Clears all stored KV cache data from the storage manager.
        """
        logger.info("CLEAR: clearing all stored KV cache data")
        with self.lock:
            self.storage_manager.memcheck()
            self.storage_manager.clear(force=True)
            self.storage_manager.memcheck()
        logger.info("CLEAR: completed")

    def close(self) -> None:
        """
        Closes the MPCacheEngine and releases all resources.
        """
        # Stop TTL scan thread and unregister from registry
        self._scan_thread.stop()
        PeriodicThreadRegistry.get_instance().unregister(self._scan_thread.name)

        # Close storage manager
        self.storage_manager.close()
        logger.info("MPCacheEngine closed")

        # Release GPU contexts
        self.gpu_contexts.clear()

    def _create_storage_manager(
        self,
        config: StorageManagerConfig,
    ) -> StorageManager:
        """Factory method: create StorageManager or ABOStorageManager."""
        if self._abo_enabled:
            return ABOStorageManager(config)
        return StorageManager(config)

    # -----------------------------------------------------------------
    # Background TTL scan
    # -----------------------------------------------------------------

    _SCAN_INTERVAL: float = 60.0  # seconds between scans
    _STALE_TTL: float = 300.0  # 5 minutes

    def register_stale_cleaner(
        self,
        name: str,
        cleaner: Callable[[], int],
    ) -> None:
        """Register an external cleanup callback.

        Args:
            name: Human-readable label for logging.
            cleaner: Callable returning the number of entries
                cleaned.
        """
        self._stale_cleaners.append((name, cleaner))

    def _ttl_scan_execute(self) -> ThreadRunSummary:
        """Execute all registered stale cleaners.

        Returns:
            ThreadRunSummary with cleanup statistics.
        """
        total = 0
        total += self.session_manager.cleanup_expired()
        total += self._cleanup_stale_jobs()
        failed_cleaners: list[str] = []
        for name, cleaner in self._stale_cleaners:
            try:
                n = cleaner()
                total += n
            except Exception:
                logger.exception("TTL scan: error in cleaner %s", name)
                failed_cleaners.append(name)
        if total:
            logger.info("TTL scan: cleaned %d stale entries", total)
        return ThreadRunSummary(
            success=len(failed_cleaners) == 0,
            message="cleaned %d entries" % total
            if not failed_cleaners
            else "errors in: %s" % ", ".join(failed_cleaners),
            extra_info={
                "cleaned_total": str(total),
                "failed_cleaners": str(len(failed_cleaners)),
            },
        )

    def _cleanup_stale_jobs(self) -> int:
        """Remove prefetch jobs and pending lookups older than
        *_STALE_TTL* seconds.

        Returns:
            Number of entries removed.
        """
        now = time.monotonic()
        cutoff = now - self._STALE_TTL
        cleaned = 0

        # Stale prefetch jobs
        with self._prefetch_job_lock:
            stale_ids = [
                jid
                for jid, job in self._prefetch_jobs.items()
                if job.handle.submit_time < cutoff
            ]
            for jid in stale_ids:
                job = self._prefetch_jobs.pop(jid)
                self._request_to_job_id.pop(job.request_id, None)
                cleaned += 1
                logger.warning(
                    "Cleaned stale prefetch job %d (request=%s, age=%.0fs)",
                    jid,
                    job.request_id,
                    now - job.handle.submit_time,
                )

        # Stale pending lookups
        with self._pending_lookups_lock:
            stale_reqs = [
                rid
                for rid, st in self._pending_lookups.items()
                if st.created_at < cutoff
            ]
            for rid in stale_reqs:
                self._pending_lookups.pop(rid, None)
                cleaned += 1
                logger.warning(
                    "Cleaned stale pending lookup (request=%s)",
                    rid,
                )

        return cleaned

    def _setup_metrics(self) -> None:
        """Register OTel observable gauges for MP engine metrics."""
        _gauge = partial(register_gauge, "lmcache.mp_engine")

        # -- Engine-level gauges --
        _gauge(
            "lmcache_mp.active_prefetch_jobs",
            "Number of active prefetch jobs",
            lambda: len(self._prefetch_jobs),
        )
        _gauge(
            "lmcache_mp.active_sessions",
            "Number of active sessions",
            lambda: self.session_manager.active_count(),
        )
        _gauge(
            "lmcache_mp.registered_gpu_count",
            "Number of registered GPU contexts",
            lambda: len(self.gpu_contexts),
        )
        _gauge(
            "lmcache_mp.pending_lookups",
            "Number of pending sync lookup states",
            lambda: len(self._pending_lookups),
        )

        # -- L1 memory gauges (via StorageManager) --
        _gauge(
            "lmcache_mp.l1_memory_used_bytes",
            "L1 CPU cache memory used in bytes",
            lambda: self._get_l1_status("memory_used_bytes", 0),
        )
        _gauge(
            "lmcache_mp.l1_memory_total_bytes",
            "L1 CPU cache total memory in bytes",
            lambda: self._get_l1_status("memory_total_bytes", 0),
        )
        _gauge(
            "lmcache_mp.l1_memory_usage_ratio",
            "L1 CPU cache memory usage ratio (0.0-1.0)",
            lambda: self._get_l1_status("memory_usage_ratio", 0.0),
        )
        _gauge(
            "lmcache_mp.l1_total_object_count",
            "Total number of objects in L1 cache",
            lambda: self._get_l1_status("total_object_count", 0),
        )
        _gauge(
            "lmcache_mp.l1_write_locked_count",
            "Number of write-locked objects in L1",
            lambda: self._get_l1_status("write_locked_count", 0),
        )
        _gauge(
            "lmcache_mp.l1_read_locked_count",
            "Number of read-locked objects in L1",
            lambda: self._get_l1_status("read_locked_count", 0),
        )

        # -- Prefetch controller gauges --
        _gauge(
            "lmcache_mp.prefetch_submission_queue_size",
            "Prefetch controller submission queue size",
            lambda: self._get_prefetch_status("submission_queue_size", 0),
        )
        _gauge(
            "lmcache_mp.prefetch_in_flight_count",
            "Prefetch controller in-flight request count",
            lambda: self._get_prefetch_status("in_flight_request_count", 0),
        )

        # -- Hit rate gauge --
        _gauge(
            "lmcache_mp.cumulative_hit_rate",
            "Cumulative token-level hit rate (0.0-1.0)",
            lambda: self.session_manager.report_hit_stats().get("hit_rate", 0.0),
        )

    def _get_l1_status(self, key: str, default: int | float):
        """Helper to safely read a field from L1 manager status."""
        try:
            sm = self.storage_manager.report_status()
            return sm.get("l1_manager", {}).get(key, default)
        except Exception:
            return default

    def _get_prefetch_status(self, key: str, default: int | float):
        """Helper to safely read a field from prefetch controller."""
        try:
            sm = self.storage_manager.report_status()
            return sm.get("prefetch_controller", {}).get(key, default)
        except Exception:
            return default


def add_handler_helper(
    server: MessageQueueServer, request_type: RequestType, handler_function
):
    payload_classes = get_payload_classes(request_type)
    handler_type = get_handler_type(request_type)
    server.add_handler(
        request_type,
        payload_classes,
        handler_type,
        handler_function,
    )


def run_cache_server(
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    obs_config: ObservabilityConfig,
    return_engine: bool = False,
    start_prometheus_http_server: bool = True,
    http_frontend_config=None,
):
    """
    Run the LMCache cache server with ZMQ message queue.

    Args:
        mp_config: Configuration for the ZMQ multiprocess server
        storage_manager_config: Configuration for the storage manager
        obs_config: Configuration for the observability stack
        return_engine: If True, return (server, engine, plugin_launcher)
                       after starting;
                       if False, run blocking loop to keep server alive
        start_prometheus_http_server: If True, start a Prometheus HTTP server
        http_frontend_config: Optional HTTPFrontendConfig; when provided it
            is forwarded to MPRuntimePluginLauncher so plugins can read the
            HTTP host/port without requiring duplicate CLI flags.

    Returns:
        If return_engine is True: tuple of (MessageQueueServer,
            MPCacheEngine, MPRuntimePluginLauncher | None)
        If return_engine is False: None (blocks until interrupted)
    """
    event_bus = init_observability(
        obs_config, start_prometheus_http_server=start_prometheus_http_server
    )

    # Initialize the engine (loggers self-register with the global controller)
    engine = MPCacheEngine(
        storage_manager_config=storage_manager_config,
        chunk_size=mp_config.chunk_size,
        hash_algorithm=mp_config.hash_algorithm,
    )

    # Register external stale cleaners from observability subscribers
    # First Party
    from lmcache.v1.mp_observability.subscribers.metrics.mp_engine import (
        MPEngineMetricsSubscriber,
    )

    metrics_sub = event_bus.find_subscriber(MPEngineMetricsSubscriber)
    if metrics_sub is not None:
        engine.register_stale_cleaner(
            "MPEngineMetricsSubscriber",
            metrics_sub.cleanup_stale,
        )

    # Initialize the message queue server
    context = zmq.Context.instance()
    server = MessageQueueServer(
        bind_url=f"tcp://{mp_config.host}:{mp_config.port}",
        context=context,
    )

    # Add handlers
    add_handler_helper(server, RequestType.REGISTER_KV_CACHE, engine.register_kv_cache)
    add_handler_helper(
        server, RequestType.UNREGISTER_KV_CACHE, engine.unregister_kv_cache
    )
    add_handler_helper(server, RequestType.STORE, engine.store)
    add_handler_helper(server, RequestType.LOOKUP, engine.lookup)
    add_handler_helper(server, RequestType.SYNC_LOOKUP, engine.sync_lookup)
    add_handler_helper(
        server, RequestType.QUERY_PREFETCH_STATUS, engine.query_prefetch_status
    )
    add_handler_helper(
        server,
        RequestType.QUERY_PREFETCH_LOOKUP_HITS,
        engine.query_prefetch_lookup_hits,
    )
    add_handler_helper(server, RequestType.FREE_LOOKUP_LOCKS, engine.free_lookup_locks)
    add_handler_helper(server, RequestType.DELETE_CHUNKS, engine.delete_chunks)
    add_handler_helper(
        server,
        RequestType.QUERY_PREFETCH_STATUS_WITH_REQ_ID,
        engine.query_prefetch_status_with_req_id,
    )
    add_handler_helper(server, RequestType.RETRIEVE, engine.retrieve)
    add_handler_helper(server, RequestType.CLEAR, engine.clear)
    add_handler_helper(server, RequestType.GET_CHUNK_SIZE, engine.get_chunk_size)
    add_handler_helper(server, RequestType.PING, engine.ping)
    add_handler_helper(server, RequestType.END_SESSION, engine.end_session)
    add_handler_helper(server, RequestType.NOOP, engine.debug)
    add_handler_helper(
        server,
        RequestType.REPORT_BLOCK_ALLOCATION,
        engine.report_block_allocations,
    )

    # Assign thread pools
    server.add_affinity_thread_pool(
        [RequestType.STORE, RequestType.RETRIEVE],
        max_workers=mp_config.max_gpu_workers,
    )
    server.add_normal_thread_pool(
        [
            RequestType.LOOKUP,
            RequestType.SYNC_LOOKUP,
            RequestType.QUERY_PREFETCH_STATUS,
            RequestType.QUERY_PREFETCH_LOOKUP_HITS,
            RequestType.QUERY_PREFETCH_STATUS_WITH_REQ_ID,
            RequestType.FREE_LOOKUP_LOCKS,
            RequestType.DELETE_CHUNKS,
            RequestType.END_SESSION,
            RequestType.CLEAR,
            RequestType.PING,
            RequestType.REPORT_BLOCK_ALLOCATION,
        ],
        max_workers=mp_config.max_cpu_workers,
    )

    logger.info(
        "LMCache ZMQ cache server is running on tcp://%s:%d",
        mp_config.host,
        mp_config.port,
    )
    # Start the ZMQ server
    torch.cuda.init()
    server.start()

    logger.info("LMCache cache server is running...")

    # Launch runtime plugins if configured
    plugin_launcher = None
    if mp_config.runtime_plugin_config.locations:
        extra_kwargs = {}
        if http_frontend_config is not None:
            extra_kwargs["http_frontend_config"] = http_frontend_config
        plugin_launcher = MPRuntimePluginLauncher(
            runtime_plugin_config=mp_config.runtime_plugin_config,
            mp_config=mp_config,
            storage_manager_config=storage_manager_config,
            obs_config=obs_config,
            **extra_kwargs,
        )
        plugin_launcher.launch_plugins()

    # Return server and engine if requested (for HTTP server integration)
    if return_engine:
        return server, engine, plugin_launcher

    # Dummy loop to keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        if plugin_launcher is not None:
            plugin_launcher.stop_plugins()
        event_bus.stop()
        server.close()
        engine.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LMCache ZMQ Cache Server (without HTTP)"
    )
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    add_observability_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mp_config = parse_args_to_mp_server_config(args)
    storage_manager_config = parse_args_to_config(args)
    obs_config = parse_args_to_observability_config(args)
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=obs_config,
    )
