# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
from dataclasses import dataclass
from typing import Any
import enum
import os
import threading
import time

# Third Party
import torch
import zmq

# First Party
from lmcache.integration.request_telemetry.factory import RequestTelemetryFactory
from lmcache.utils import _lmcache_nvtx_annotate, init_logger
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CudaIPCWrapper,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.periodic_thread import PeriodicThread, ThreadLevel, ThreadRunSummary

logger = init_logger(__name__)


class ExtraConfigDefault(enum.Enum):
    """Centralized default values for extra_config keys.

    Each member's *name* is the key used in the extra_config dict,
    and its *value* is the default.
    """

    # Timeout (seconds) for blocking MQ requests: initial
    # chunk-size query, KV cache registration/unregistration,
    # and other synchronous operations.
    mq_timeout = 300.0
    # Interval (seconds) between periodic heartbeat pings
    # to the server.
    heartbeat_interval = 10.0


# Backward-compatible aliases for the legacy `lmcache_mp_connector_0180`
# entry point, which still passes these as positional/keyword args.
DEFAULT_MQ_TIMEOUT: float = ExtraConfigDefault.mq_timeout.value
DEFAULT_HEARTBEAT_INTERVAL: float = ExtraConfigDefault.heartbeat_interval.value

_EXTRA_CONFIG_KEY_PREFIX = "lmcache.mp."


def _resolve_extra_config(
    extra_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve extra_config against :class:`ExtraConfigDefault`.

    Keys in *extra_config* are expected to carry the
    ``lmcache.mp.`` prefix (e.g. ``lmcache.mp.mq_timeout``).
    The prefix is stripped before matching against
    :class:`ExtraConfigDefault` members.

    All ``lmcache.mp.*`` entries are logged.  Entries whose
    value differs from the default are additionally marked as
    *overridden*.

    Args:
        extra_config: User-supplied config dict (may be *None*).

    Returns:
        A dict keyed by :pyattr:`ExtraConfigDefault` member names.
    """
    stripped: dict[str, Any] = {}
    if extra_config is not None:
        for k, v in extra_config.items():
            if k.startswith(_EXTRA_CONFIG_KEY_PREFIX):
                short = k[len(_EXTRA_CONFIG_KEY_PREFIX) :]
                stripped[short] = v

    resolved: dict[str, Any] = {}
    for item in ExtraConfigDefault:
        default = item.value
        raw = stripped.get(item.name)
        value = type(default)(raw) if raw is not None else default
        if value != default:
            logger.info(
                "%s%s = %s (overridden, default: %s)",
                _EXTRA_CONFIG_KEY_PREFIX,
                item.name,
                value,
                default,
            )
        else:
            logger.info(
                "%s%s = %s",
                _EXTRA_CONFIG_KEY_PREFIX,
                item.name,
                value,
            )
        resolved[item.name] = value
    return resolved


def wrap_kv_caches(kv_caches: dict[str, torch.Tensor]) -> KVCache:
    logger.info("KV caches keys are %s", list(kv_caches.keys()))
    return [CudaIPCWrapper(tensor) for tensor in kv_caches.values()]


def send_lmcache_request(
    mq_client: MessageQueueClient,
    request_type: RequestType,
    payloads: list[Any],
) -> MessagingFuture[Any]:
    """
    Helper function to send the request to the LMCache multiprocess server

    Args:
        mq_client: The LMCache multiprocess mode message queue client
        request_type: The request type
        payloads: The request payloads

    Returns:
        A messaging future for the request
    """

    future = mq_client.submit_request(
        request_type, payloads, get_response_class(request_type)
    )
    return future


def get_lmcache_chunk_size(
    mq_client: MessageQueueClient,
) -> int:
    """
    Helper function to get the LMCache chunk size from the server

    Args:
        mq_client: The LMCache multiprocess mode message queue client

    Returns:
        An integer representing the LMCache chunk size
    """
    future = send_lmcache_request(mq_client, RequestType.GET_CHUNK_SIZE, [])
    chunk_size = future.result(timeout=DEFAULT_MQ_TIMEOUT)
    return chunk_size


def send_ping(
    mq_client: MessageQueueClient,
    timeout: float,
) -> bool:
    """Send a PING request and return the result.

    Returns:
        True if server is healthy, False on timeout or error.
    """
    try:
        future = send_lmcache_request(mq_client, RequestType.PING, [])
        return future.result(timeout=timeout)
    except TimeoutError:
        return False
    except Exception:
        logger.debug("Ping failed with exception", exc_info=True)
        return False


@dataclass
class ParallelStrategy:
    use_mla: bool
    """Whether to use the MLA."""

    global_world_size: int
    """Total number of vLLM worker processes."""

    global_rank: int
    """The current worker's rank in ``[0, global_world_size)``."""

    tp_size: int
    """Tensor-parallel size of the model."""

    pp_size: int
    """Pipeline-parallel size of the model."""

    n_servers: int
    """Number of LMCache servers backing this deployment
    (1 = single shared server, >1 = one server per node)."""

    kv_world_size: int
    """How many different pieces the KV cache is split into across the
    whole deployment."""

    kv_worker_id: int
    """Which piece of the KV cache the current worker has, in
    ``[0, kv_world_size)``."""

    local_kv_world_size: int
    """Same as ``kv_world_size`` but scoped to a single LMCache server;
    consulted only when ``n_servers > 1``."""

    local_tp_size: int
    """Same as ``tp_size`` but scoped to a single LMCache server;
    consulted only when ``n_servers > 1``."""


class HeartbeatThread(PeriodicThread):
    """Periodically checks server health via PING.

    Manages a threading.Event that adapters use to gate operations.
    When unhealthy, the adapter enters degraded mode; if the server
    recovers, the adapter automatically resumes normal operation.
    """

    def __init__(
        self,
        mq_client: MessageQueueClient,
        health_event: threading.Event,
        interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ):
        """
        Args:
            mq_client: The message queue client used to send PING requests.
            health_event: A threading.Event shared with the adapter.
                Set when the server is healthy, cleared when unhealthy.
                Adapters check this event to decide whether to proceed
                with operations or enter degraded mode.
            interval: Seconds between heartbeat pings and ping timeout.
        """
        super().__init__(
            name="lmcache-heartbeat",
            interval=interval,
            level=ThreadLevel.CRITICAL,
        )
        self._mq_client = mq_client
        self._health_event = health_event
        self._interval = interval

    def _execute(self) -> ThreadRunSummary:
        was_healthy = self._health_event.is_set()
        healthy = send_ping(self._mq_client, timeout=self._interval)

        if healthy:
            self._health_event.set()
            if not was_healthy:
                logger.warning(
                    "LMCache server is healthy again — resuming normal operation"
                )
        else:
            self._health_event.clear()
            if was_healthy:
                logger.warning("LMCache server is unhealthy — entering degraded mode")

        return ThreadRunSummary(
            success=True,
            message="healthy" if healthy else "unhealthy",
        )


@dataclass
class LoadStoreOp:
    token_ids: list[int]
    """Token IDs for the load/store operation"""

    block_ids: list[int]
    """Block ids for the load/store operation"""

    start: int = 0
    """Start token index"""

    end: int = 0
    """End token index"""

    skip_first_n_tokens: int = 0
    """Number of tokens to skip writing at the beginning of the retrieve
    range. Used to avoid overwriting APC-shared GPU blocks during retrieve."""

    def __len__(self) -> int:
        return len(self.block_ids)


StoreResult = bool
RetrieveResult = bool
LookupResult = int


class LMCacheMPSchedulerAdapter:
    def __init__(
        self,
        server_urls: list[str],
        context: zmq.Context,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        save_decode_cache: bool = False,
        extra_config: dict[str, Any] | None = None,
    ):
        """
        Args:
            server_urls: A list of server URLs for the LMCache message queue
            context: The ZMQ context
            model_name: The model name used for LMCache keys
            vllm_block_size: The block size used in vLLM
            parallel_strategy:
                The parallel strategy, which includes `use_mla`,
                `kv_world_size`, `kv_worker_id` and so on
            mq_timeout: Timeout in seconds for message queue requests.
            heartbeat_interval: Interval in seconds between heartbeat pings.
            save_decode_cache: Whether to save KV cache during decode phase.
                If False (default), store operations are skipped when the
                request is in decode phase.
        """
        assert len(server_urls) >= 1, "At least one server url required"
        self._server_urls: list[str] = list(server_urls)
        self.mq_clients: dict[str, MessageQueueClient] = {
            url: MessageQueueClient(url, context) for url in self._server_urls
        }
        if extra_config is not None:
            cfg = _resolve_extra_config(extra_config)
            mq_timeout = cfg[ExtraConfigDefault.mq_timeout.name]
            heartbeat_interval = cfg[ExtraConfigDefault.heartbeat_interval.name]
        self._mq_timeout = mq_timeout

        # Two-phase lookup state (multi-server):
        # - phase 1: request_ids whose LOOKUP was successfully submitted
        #   to ALL backing servers (we no longer track a per-server job_id).
        # - phase 2: request_id -> aggregated matched-token count, cached
        #   once every server has reported its prefetch status.
        # Both maps are cleared by `cleanup_lookup_result`.
        self._lookup_job_ids: set[str] = set()
        self._finished_lookup_jobs: dict[str, int] = {}

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy
        self.save_decode_cache = save_decode_cache
        self._per_server_hits: dict[str, dict[str, int]] = {}

        # Decode phase tracking: request IDs that have entered decode phase.
        # Connectors call mark_decode_phase() to register, and
        # should_skip_store() to check whether a store should be skipped.
        self._decode_requests: set[str] = set()

        # Read chunk size from lmcache
        chunk_sizes: dict[str, int] = {}
        for url, client in self.mq_clients.items():
            try:
                chunk_sizes[url] = get_lmcache_chunk_size(client)
            except TimeoutError:
                for c in self.mq_clients.values():
                    c.close()
                raise ConnectionError(
                    f"LMCache server {url} did not respond within {mq_timeout}s"
                ) from None

        # All servers must share chunk_size, otherwise the min() aggregation
        # over per-server hits would mix different granularities.
        unique_sizes = set(chunk_sizes.values())
        assert len(unique_sizes) == 1, (
            f"All LMCache servers must share the same chunk_size, got {chunk_sizes}"
        )
        self.chunk_size = unique_sizes.pop()
        assert self.chunk_size % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = self.chunk_size // vllm_block_size

        # Health state (shared with heartbeat thread)
        self._health_events: dict[str, threading.Event] = {}
        for url in self._server_urls:
            ev = threading.Event()
            ev.set()  # start optimistic; heartbeat will clear on failure
            self._health_events[url] = ev

        # Heartbeat thread is created but NOT started yet.
        # It will be lazily started on the first lookup
        # request, by which time vLLM is fully ready.
        self._heartbeat_interval = heartbeat_interval
        self._heartbeats: dict[str, HeartbeatThread] = {}
        self._heartbeat_lock = threading.Lock()

    @property
    def world_size(self) -> int:
        """Get the kv world size."""
        if not self.parallel_strategy.use_mla and self.parallel_strategy.n_servers > 1:
            return self.parallel_strategy.local_kv_world_size
        return self.parallel_strategy.kv_world_size

    @property
    def tp_size(self) -> int:
        """The tensor parallel size."""
        if self.parallel_strategy.n_servers > 1:
            return self.parallel_strategy.local_tp_size
        return self.parallel_strategy.tp_size

    @property
    def is_healthy(self) -> bool:
        """True iff every backing LMCache server is healthy."""
        return all(ev.is_set() for ev in self._health_events.values())

    def healthy_urls(self) -> list[str]:
        """Return the list of server URLs that are currently healthy."""
        return [u for u, ev in self._health_events.items() if ev.is_set()]

    def _ensure_heartbeat_started(self) -> None:
        """Lazily start the heartbeat thread on first use."""
        if self._heartbeats:
            return
        with self._heartbeat_lock:
            if self._heartbeats:
                return
            for url, client in self.mq_clients.items():
                hb = HeartbeatThread(
                    mq_client=client,
                    health_event=self._health_events[url],
                    interval=self._heartbeat_interval,
                )
                hb.start()
                self._heartbeats[url] = hb

    @_lmcache_nvtx_annotate
    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
    ):
        """
        Submit a new lookup request to LMCache if there is no ongoing request.

        Sends a LOOKUP request to the server and blocks until a prefetch
        job ID is returned.  The actual prefetch result can then be polled
        via ``check_lookup_result``.

        Args:
            request_id: The ID of the lookup request. The same ID indicates it's
                from the same request
            token_ids: Token IDs to lookup from LMCache

        Returns:
            None

        Notes:
            This function will have a side-effect: submitting a look up request to
            LMCache, which will essentially 'lock' the KV cache chunks in the LMCache
            for later retrieve operations.
            In the meantime, this function will record the lookup request, and the
            status of the look up request can be checked by `check_lookup_result`.
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            return

        if request_id in self._lookup_job_ids:
            # Skip if there is already a lookup request
            return

        aligned_end = (len(token_ids) // self.chunk_size) * self.chunk_size

        key = self._create_key(
            token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
        ).no_worker_id_version()
        futures: dict[str, MessagingFuture[Any]] = {}
        for url in self._server_urls:
            try:
                futures[url] = send_lmcache_request(
                    self.mq_clients[url],
                    RequestType.LOOKUP,
                    [key, self.tp_size],
                )
            except Exception as e:
                logger.error("LOOKUP submit to %s failed: %s", url, e, exc_info=True)
                self._health_events[url].clear()
        results: list[tuple[str, bool]] = []
        for url, fut in futures.items():
            try:
                fut.result(timeout=self._mq_timeout)
                results.append((url, True))
            except TimeoutError:
                logger.warning(
                    "LOOKUP to %s timed out after %ss; marking unhealthy.",
                    url,
                    self._mq_timeout,
                )
                self._health_events[url].clear()
                results.append((url, False))
            except Exception as e:
                logger.error("LOOKUP to %s failed: %s", url, e, exc_info=True)
                self._health_events[url].clear()
                results.append((url, False))

        for url in self._server_urls:
            if url not in futures:
                results.append((url, False))

        if all(ok for _, ok in results):
            self._lookup_job_ids.add(request_id)
        else:
            failed = [u for u, ok in results if not ok]
            logger.error(
                "[req=%s] LOOKUP failed on servers %s -- fall back to no-hit",
                request_id,
                failed,
            )
            # Best-effort: proactively release the read locks the
            # *successful* servers have already taken, so we do not
            # have to wait for server-side GC / TTL.  We reuse the
            # same key (start=0, end=aligned_end, no_worker_id_version)
            # as the LOOKUP submitted above.
            succeeded = [u for u, ok in results if ok]
            for url in succeeded:
                client = self.mq_clients.get(url)
                if client is None:
                    continue
                try:
                    send_lmcache_request(
                        client,
                        RequestType.FREE_LOOKUP_LOCKS,
                        [key, self.tp_size],
                    )
                except Exception as e:
                    logger.warning(
                        "[req=%s] FREE_LOOKUP_LOCKS (partial-fail rollback) "
                        "to %s failed: %s "
                        "(rely on server-side GC for any residual lock)",
                        request_id,
                        url,
                        e,
                    )

    @_lmcache_nvtx_annotate
    def check_lookup_result(self, request_id: str) -> int | None:
        """
        Check the result of a previously submitted lookup request.

        Sends a QUERY_PREFETCH_STATUS_WITH_REQ_ID request to the server
        and blocks until the server responds.  Returns the matched token
        count when the prefetch is complete, or None if still in progress.

        Args:
            request_id: The ID of the lookup request submitted in
                `maybe_submit_lookup_request`

        Returns:
            An integer representing the total number of tokens matched
            in LMCache (prefix matching), or
            None if the lookup request is not finished yet.
        """
        if request_id not in self._lookup_job_ids:
            # No job — either unhealthy at submit time or already cleaned up
            return 0

        if not self.is_healthy:
            # Some server went down — give up on this lookup
            self._lookup_job_ids.discard(request_id)
            return 0

        if request_id in self._finished_lookup_jobs:
            # Return cached result if every server has already reported
            return self._finished_lookup_jobs[request_id]

        per_server = self._per_server_hits.setdefault(request_id, {})
        unresolved_urls = [u for u in self._server_urls if u not in per_server]

        futures: dict[str, MessagingFuture[Any]] = {}
        for url in unresolved_urls:
            try:
                futures[url] = send_lmcache_request(
                    self.mq_clients[url],
                    RequestType.QUERY_PREFETCH_STATUS_WITH_REQ_ID,
                    [request_id],
                )
            except Exception as e:
                logger.error(
                    "QUERY_PREFETCH_STATUS_WITH_REQ_ID submit to %s failed: %s",
                    url,
                    e,
                    exc_info=True,
                )
                self._health_events[url].clear()

        for url, fut in futures.items():
            try:
                r = fut.result(timeout=self._mq_timeout)
            except TimeoutError:
                logger.warning(
                    "QUERY_PREFETCH_STATUS_WITH_REQ_ID to %s timed out. "
                    "Marking unhealthy.",
                    url,
                )
                self._health_events[url].clear()
                continue
            except Exception as e:
                logger.error(
                    "QUERY_PREFETCH_STATUS_WITH_REQ_ID to %s failed: %s",
                    url,
                    e,
                    exc_info=True,
                )
                self._health_events[url].clear()
                continue
            if r is None:
                continue
            per_server[url] = int(r)

        if len(per_server) < len(self._server_urls):
            return None

        min_chunks = min(per_server.values())
        max_chunks = max(per_server.values())
        if min_chunks != max_chunks:
            logger.warning(
                "[req=%s] LMCache hit mismatch across servers: %s → take min=%d",
                request_id,
                dict(per_server),
                min_chunks,
            )

        token_count = min_chunks * self.chunk_size
        self._finished_lookup_jobs[request_id] = token_count
        return token_count

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def cleanup_lookup_result(self, request_id: str) -> None:
        """
        Clean up lookup state for a finished request to prevent memory leak.
        Args:
            request_id: The ID of the finished request.
        """
        self._lookup_job_ids.discard(request_id)
        self._finished_lookup_jobs.pop(request_id, None)
        self._per_server_hits.pop(request_id, None)

    def shutdown(self) -> None:
        """Shutdown the scheduler adapter and its resources."""
        for client in self.mq_clients.values():
            client.close()
        with self._heartbeat_lock:
            for hb in self._heartbeats.values():
                hb.stop()

    def mark_decode_phase(self, request_id: str) -> None:
        """Mark a request as having entered the decode phase.

        Connectors should call this when they detect that a request
        has transitioned from context (prefill) phase to decode
        (auto-regressive generation) phase.

        Args:
            request_id: The ID of the request.
        """
        self._decode_requests.add(request_id)

    def is_decode_phase(self, request_id: str) -> bool:
        """Check whether a request is in the decode phase.

        Args:
            request_id: The ID of the request.

        Returns:
            True if the request has been marked as decode phase.
        """
        return request_id in self._decode_requests

    def should_skip_store(self, request_id: str) -> bool:
        """Determine whether a store operation should be skipped.

        The store is skipped when the request is in decode phase and
        ``save_decode_cache`` is disabled.

        This centralises the decode-phase filtering logic so that
        connectors do not need to implement it themselves.

        Args:
            request_id: The ID of the request.

        Returns:
            True if the store should be skipped.
        """
        if self.save_decode_cache:
            return False
        return request_id in self._decode_requests

    def cleanup_decode_state(self, request_id: str) -> None:
        """Remove decode-phase tracking for a finished request.

        Should be called when a request finishes to prevent memory leak.

        Args:
            request_id: The ID of the finished request.
        """
        self._decode_requests.discard(request_id)

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
    ) -> None:
        """Release read locks acquired during lookup without a full retrieve.

        Use this when some chunks matched by lookup overlap with blocks that
        vLLM has already computed, so they will never be retrieved.  Calling
        this prevents those chunks from holding read locks until TTL expiry.

        Or use this when a request is cancelled or aborted after lookup but
        before retrieve to avoid holding read locks until TTL expiry.

        When ``start`` or ``end`` is not aligned to the chunk size, the
        entire chunk containing start boundary is freed but not end boundary.
        It is caller's responsibility to properly align the boundaries.

        Args:
            token_ids: Token IDs for the key (same as used in lookup).
            start: Start token index.
            end: End token index.
            request_id: The request ID.
        """
        if not self.is_healthy:
            return

        base_key = self._create_key(
            token_ids,
            start=start,
            end=end,
            request_id=request_id,
        ).no_worker_id_version()
        for url in self._server_urls:
            client = self.mq_clients.get(url)
            if client is None:
                continue
            try:
                send_lmcache_request(
                    client,
                    RequestType.FREE_LOOKUP_LOCKS,
                    [base_key, self.tp_size],
                )
            except Exception as e:
                logger.warning(
                    "[req=%s] FREE_LOOKUP_LOCKS to %s failed: %s "
                    "(rely on server-side GC for any residual lock)",
                    request_id,
                    url,
                    e,
                )

    def free_per_server_overhit_locks(
        self,
        token_ids: list[int],
        request_id: str,
    ) -> None:
        """Release the per-server tail locks that exceed the global min hit."""
        if not self.is_healthy:
            return
        per_server = self._per_server_hits.get(request_id)
        if not per_server:
            return
        min_hit_chunks = min(per_server.values())
        min_hit_end = min_hit_chunks * self.chunk_size
        for url, hit_chunks in per_server.items():
            if hit_chunks <= min_hit_chunks:
                continue
            tail_end = min(hit_chunks * self.chunk_size, len(token_ids))
            tail_key = self._create_key(
                token_ids=token_ids,
                start=min_hit_end,
                end=tail_end,
                request_id=request_id,
            ).no_worker_id_version()
            client = self.mq_clients.get(url)
            if client is None:
                continue
            try:
                send_lmcache_request(
                    client,
                    RequestType.FREE_LOOKUP_LOCKS,
                    [tail_key, self.tp_size],
                )
            except Exception as e:
                logger.warning(
                    "[req=%s] FREE_LOOKUP_LOCKS (tail) to %s failed: %s "
                    "(rely on server-side GC for any residual lock)",
                    request_id,
                    url,
                    e,
                )

    def end_session(self, request_id: str) -> None:
        """
        Notify LMCache server to remove the session for a finished request.
        Args:
            request_id: The ID of the finished request.
        """
        if not self.is_healthy:
            return

        for url in self.healthy_urls():
            send_lmcache_request(
                self.mq_clients[url],
                RequestType.END_SESSION,
                [request_id],
            )

    def report_block_allocations(
        self,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Report vLLM GPU block allocation deltas to LMCache server.

        Fire-and-forget: does not wait for a response. If the server
        is unhealthy the report is silently dropped.

        Args:
            records: List of BlockAllocationRecord with per-request
                block and token allocation deltas.
        """
        if not self.is_healthy or not records:
            return

        for url in self.healthy_urls():
            send_lmcache_request(
                self.mq_clients[url],
                RequestType.REPORT_BLOCK_ALLOCATION,
                [os.getpid(), self.model_name, records],
            )

    # Helper functions
    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
    ) -> IPCCacheEngineKey:
        """Convert token IDs to an IPC cache engine key"""
        # NOTE: for the scheduler adapter, we don't have a worker id,
        # so we set it to None in the key.
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
        )


class LMCacheMPSyncLookUpSchedulerAdapter(LMCacheMPSchedulerAdapter):
    """LMCacheMPSchedulerAdapter subclass that uses the synchronous
    SYNC_LOOKUP protocol instead of the two-phase LOOKUP + polling path.

    ``sync_lookup`` performs the entire lookup in a single blocking
    round-trip and returns the matched token count directly.
    """

    @_lmcache_nvtx_annotate
    def sync_lookup(
        self,
        request_id: str,
        token_ids: list[int],
    ) -> int:
        """Synchronous prefix lookup — single blocking round-trip.

        Sends a SYNC_LOOKUP request to the server that performs the L1
        prefix scan and L2 existence check atomically.  Returns the
        number of matched **tokens** (not chunks).

        Args:
            request_id: The request ID.
            token_ids: Full token sequence for the request.

        Returns:
            Number of matched tokens (always a multiple of
            ``chunk_size``), or 0 on error / unhealthy server.
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            return 0

        aligned_end = (len(token_ids) // self.chunk_size) * self.chunk_size

        key = self._create_key(
            token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
        ).no_worker_id_version()

        # Multi-server: fan out SYNC_LOOKUP to every backing LMCache
        # server and take the minimum hit, mirroring the async
        # LOOKUP + QUERY_PREFETCH_STATUS path on the parent class.
        futures: dict[str, MessagingFuture[Any]] = {}
        for url in self._server_urls:
            try:
                futures[url] = send_lmcache_request(
                    self.mq_clients[url],
                    RequestType.SYNC_LOOKUP,
                    [key, self.tp_size],
                )
            except Exception as e:
                logger.error(
                    "SYNC_LOOKUP submit to %s failed: %s",
                    url,
                    e,
                    exc_info=True,
                )
                self._health_events[url].clear()

        per_server: dict[str, int] = {}
        for url, fut in futures.items():
            try:
                per_server[url] = int(fut.result(timeout=self._mq_timeout))
            except TimeoutError:
                logger.warning(
                    "SYNC_LOOKUP to %s timed out after %ss. "
                    "Marking server as unhealthy.",
                    url,
                    self._mq_timeout,
                )
                self._health_events[url].clear()
            except Exception as e:
                logger.error(
                    "SYNC_LOOKUP to %s failed: %s",
                    url,
                    e,
                    exc_info=True,
                )
                self._health_events[url].clear()

        # If any server failed to report, fall back to no-hit so that we
        # never retrieve from a partially-available prefix.
        if len(per_server) < len(self._server_urls):
            failed = [u for u in self._server_urls if u not in per_server]
            logger.error(
                "[req=%s] SYNC_LOOKUP failed on servers %s -- fall back to no-hit",
                request_id,
                failed,
            )
            return 0

        min_chunks = min(per_server.values())
        max_chunks = max(per_server.values())
        if min_chunks != max_chunks:
            logger.warning(
                "[req=%s] LMCache hit mismatch across servers: %s → take min=%d",
                request_id,
                dict(per_server),
                min_chunks,
            )

        # Cache per-server hits so that `free_per_server_overhit_locks`
        # can later release tail locks held by servers whose hit
        # exceeded the global min.
        self._per_server_hits[request_id] = per_server

        hit_tokens = min_chunks * self.chunk_size
        logger.debug(
            "SYNC_LOOKUP: request_id=%s hit %d chunks (%d tokens)",
            request_id,
            min_chunks,
            hit_tokens,
        )
        return hit_tokens


class LMCacheMPWorkerAdapter:
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        save_decode_cache: bool = False,
        sync_mode: bool = False,
        extra_config: dict[str, Any] | None = None,
    ):
        self.mq_client = MessageQueueClient(server_url, context)
        if extra_config is not None:
            cfg = _resolve_extra_config(extra_config)
            mq_timeout = cfg[ExtraConfigDefault.mq_timeout.name]
            heartbeat_interval = cfg[ExtraConfigDefault.heartbeat_interval.name]
        self._mq_timeout = mq_timeout
        self.sync_mode = sync_mode

        # Instance id for GPU worker
        self.instance_id = os.getpid()

        # Registered kv caches from vLLM
        self.kv_caches: dict[str, torch.Tensor] = {}

        # Request futures
        self.store_futures: dict[str, MessagingFuture[StoreResult]] = {}
        # request_id -> (future, block_ids)
        self.retrieve_futures: dict[
            str, tuple[MessagingFuture[RetrieveResult], list[int]]
        ] = {}

        # Block IDs that failed due to retrieve timeout
        self.error_block_ids: set[int] = set()

        # The store requests that have finished execution in LMCache
        self.finished_stores: set[str] = set()
        # The finished request ids that are passed via vLLM and also
        # have corresponding store requests submitted to LMCache before
        self.previously_finished: set[str] = set()
        # Request IDs already returned as finished_sending to the scheduler.
        # Prevents re-reporting the same ID after drain clears tracking sets.
        self._returned_finished: set[str] = set()

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy
        self.save_decode_cache = save_decode_cache

        # Read chunk size from lmcache
        try:
            chunk_size = get_lmcache_chunk_size(self.mq_client)
        except TimeoutError:
            self.mq_client.close()
            raise ConnectionError(
                f"LMCache server did not respond within {mq_timeout}s. "
                "Is the server running?"
            ) from None
        assert chunk_size % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = chunk_size // vllm_block_size

        # Health state (shared with heartbeat thread)
        self._health_event = threading.Event()
        self._health_event.set()

        # Heartbeat thread is created but NOT started yet.
        # It will be lazily started on the first store or retrieve
        # request, by which time vLLM is fully ready (model loaded,
        # KV caches allocated, warmup & CUDA graph capture done).
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat: HeartbeatThread | None = None
        self._heartbeat_lock = threading.Lock()

        # request telemetry, used for prefill-decode disagg
        # TODO: pass down the configuration via vLLM connector config
        # instead of env var
        self.request_telemetry = RequestTelemetryFactory.create(
            telemetry_type=os.getenv("LMCACHE_REQUEST_TELEMETRY_TYPE", "noop"),
            config={
                "endpoint": os.getenv(
                    "LMCACHE_REQUEST_TELEMETRY_ENDPOINT",
                    "http://localhost:5768/api/v1/telemetry",
                ),
            },
        )

    @property
    def is_healthy(self) -> bool:
        """Whether the LMCache server is healthy."""
        return self._health_event.is_set()

    @property
    def world_size(self) -> int:
        """How many pieces the KV cache is split into across the LMCache
        server this worker talks to."""
        if not self.parallel_strategy.use_mla and self.parallel_strategy.n_servers > 1:
            return self.parallel_strategy.local_kv_world_size
        return self.parallel_strategy.kv_world_size

    @property
    def worker_id(self) -> int:
        """This worker's piece index, in ``[0, world_size)``.
        Unique among workers backing the same LMCache server."""
        if not self.parallel_strategy.use_mla and self.parallel_strategy.n_servers > 1:
            return (
                self.parallel_strategy.global_rank
                % self.parallel_strategy.local_kv_world_size
            )
        return self.parallel_strategy.kv_worker_id

    @property
    def use_mla(self) -> bool:
        """Whether to use MLA."""
        return self.parallel_strategy.use_mla

    @property
    def is_first_rank_of_pp_group(self) -> bool:
        """Whether this worker is the first rank of its pipeline-parallel
        group. Use ``is_first_rank_of_node`` for per-node STORE gating."""
        return self.parallel_strategy.global_rank % self.parallel_strategy.tp_size == 0

    @property
    def is_first_rank_of_node(self) -> bool:
        """Whether this worker is the first rank on its node. Use
        ``is_first_rank_of_pp_group`` outside multi-server deployments."""
        n_servers = self.parallel_strategy.n_servers
        if n_servers <= 1:
            return self.is_first_rank_of_pp_group
        ranks_per_node = self.parallel_strategy.global_world_size // n_servers
        return self.parallel_strategy.global_rank % ranks_per_node == 0

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Register the kv caches with LMCache server

        Args:
            kv_caches: A dict of kv caches to register. The keys are the
                layer names and the values are the corresponding tensors.
        """
        # First Party
        from lmcache.integration.vllm.utils import vllm_layout_hints
        from lmcache.v1.gpu_connector.utils import (
            ensure_contiguous_kv_caches,
        )

        # Register kv cache and send the request
        logger.info("Registering kv caches")

        layout_hints = vllm_layout_hints()
        kv_caches = ensure_contiguous_kv_caches(
            kv_caches, kv_layout=layout_hints.get("kv_layout")
        )

        self.kv_caches = kv_caches

        future = send_lmcache_request(
            self.mq_client,
            RequestType.REGISTER_KV_CACHE,
            [
                self.instance_id,
                wrap_kv_caches(kv_caches),
                self.model_name,
                self.world_size,
                layout_hints,
            ],
        )
        try:
            future.result(timeout=self._mq_timeout)
        except TimeoutError:
            raise ConnectionError(
                "LMCache server did not respond to "
                "register_kv_caches within "
                f"{self._mq_timeout}s. Is the server running?"
            ) from None

    def _ensure_heartbeat_started(self) -> None:
        """Lazily start the heartbeat thread on first use."""
        if self._heartbeat is not None:
            return
        with self._heartbeat_lock:
            if self._heartbeat is not None:
                return
            self._heartbeat = HeartbeatThread(
                mq_client=self.mq_client,
                health_event=self._health_event,
                interval=self._heartbeat_interval,
            )
            self._heartbeat.start()

    @_lmcache_nvtx_annotate
    def submit_store_request(
        self, request_id: str, op: LoadStoreOp, event: torch.cuda.Event
    ):
        """
        Submit a KV cache store request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the store operation.
            event: The CUDA event that is recorded after the current
                model inference step
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            return

        assert op.token_ids is not None
        key = self._create_key(op.token_ids, op.start, op.end, request_id=request_id)
        future = send_lmcache_request(
            self.mq_client,
            RequestType.STORE,
            [key, self.instance_id, op.block_ids, event.ipc_handle()],
        ).to_cuda_future()
        self.store_futures[request_id] = future

    @_lmcache_nvtx_annotate
    def submit_retrieve_request(
        self, request_id: str, op: LoadStoreOp, event: torch.cuda.Event
    ):
        """
        Submit a KV cache retrieve request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the retrieve operation.
            event: The CUDA event that is recorded after the current
                model inference step
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            self.error_block_ids.update(op.block_ids)
            return

        # In sync mode, SYNC_LOOKUP is used and L2 prefetch may still be
        # in flight.  Poll QUERY_PREFETCH_STATUS_WITH_REQ_ID until the
        # prefetch is ready before sending RETRIEVE.
        if self.sync_mode:
            result = None
            while result is None:
                result = send_lmcache_request(
                    self.mq_client,
                    RequestType.QUERY_PREFETCH_STATUS_WITH_REQ_ID,
                    [request_id],
                ).result(timeout=self._mq_timeout)
                if result is not None:
                    break
                time.sleep(0.005)

        assert op.token_ids is not None
        key = self._create_key(op.token_ids, op.start, op.end, request_id=request_id)
        future = send_lmcache_request(
            self.mq_client,
            RequestType.RETRIEVE,
            [
                key,
                self.instance_id,
                op.block_ids,
                event.ipc_handle(),
                op.skip_first_n_tokens,
            ],
        ).to_cuda_future()
        self.retrieve_futures[request_id] = (future, list(op.block_ids))

    @_lmcache_nvtx_annotate
    def batched_submit_store_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: torch.cuda.Event,
    ):
        """
        Submit a batched store request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the store operations. Should have
                the same length as request_ids
            event: The CUDA event that is recorded after the current
                model inference step
        """
        for request_id, op in zip(request_ids, ops, strict=False):
            self.submit_store_request(request_id, op, event)

    @_lmcache_nvtx_annotate
    def batched_submit_retrieve_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: torch.cuda.Event,
    ):
        """
        Submit a batched retrieve request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the retrieve operations. Should have
                the same length as request_ids
            event: The CUDA event that is recorded after the current
                model inference step
        """
        for request_id, op in zip(request_ids, ops, strict=False):
            self.submit_retrieve_request(request_id, op, event)

        # In sync mode, block until all retrieves complete so that KV is
        # ready before the model forward pass.  In async mode, futures are
        # left for get_finished() to poll.
        if self.sync_mode:
            for request_id, (r_future, _) in self.retrieve_futures.items():
                r_future.result()
            self.retrieve_futures.clear()

    def _process_finished_stores(
        self,
        finished_req_ids_from_lmcache: set[str],
        finished_req_ids_from_engine: set[str],
    ) -> set[str]:
        """Merge LMCache-side and engine-side finished store info."""
        self.finished_stores.update(finished_req_ids_from_lmcache)
        ret_stores = set()
        for req_id in finished_req_ids_from_engine:
            if req_id in self._returned_finished:
                continue
            if req_id in self.finished_stores or req_id in self.store_futures:
                self.previously_finished.add(req_id)
            else:
                ret_stores.add(req_id)
        ret_stores.update(self._update_and_get_finished_store())
        self._returned_finished.update(ret_stores)
        return ret_stores

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids_from_engine: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Check and get the finished store and retrieve requests.

        Args:
            finished_req_ids_from_engine: the set of request ids that are
                reported as finished from the vLLM engine side.

        Returns:
            A tuple of two sets:
            - The first set contains the finished store request ids. The returned
                store request ids MUST be seen before in the
                `finished_req_ids_from_engine`.
            - The second set contains the finished retrieve request ids.

        Notes:
            When enabling async scheduling in vLLM, the same request ID may appear
            multiple times in `finished_req_ids_from_engine`. The adapter should
            take care of deduplicating the request IDs and only return the request
            IDs that have not been returned before.
        """
        # If unhealthy, drain all pending futures immediately
        if not self.is_healthy:
            finished_stores = set(self.store_futures.keys())
            finished_retrieves = set()
            for request_id, (
                _r_future,
                r_block_ids,
            ) in self.retrieve_futures.items():
                finished_retrieves.add(request_id)
                self.error_block_ids.update(r_block_ids)
            self.store_futures.clear()
            self.retrieve_futures.clear()

            ret_stores = self._process_finished_stores(
                finished_stores, finished_req_ids_from_engine
            )
            # A request may have a pending retrieve AND appear in
            # finished_req_ids_from_engine (it ran without loading KV after
            # the server died).  The scheduler processes finished_recving
            # first and deletes the request, so we must not also report it
            # in finished_sending.
            ret_stores -= finished_retrieves
            return ret_stores, finished_retrieves

        finished_stores = set()
        finished_retrieves = set()
        for request_id, s_future in self.store_futures.items():
            if not s_future.query():
                continue

            s_result = s_future.result()
            finished_stores.add(request_id)

            if not s_result:
                logger.error(
                    "Something went wrong when processing the "
                    "store request for request_id=%s",
                    request_id,
                )

        for request_id, (r_future, _) in self.retrieve_futures.items():
            if not r_future.query():
                continue

            r_result = r_future.result()
            finished_retrieves.add(request_id)

            if not r_result:
                logger.error(
                    "Something went wrong when processing the "
                    "retrieve request for request_id=%s, result=%s",
                    request_id,
                    r_result,
                )

        # Remove the finished requests from the tracking dicts
        for request_id in finished_stores:
            self.store_futures.pop(request_id, None)
        for request_id in finished_retrieves:
            self.retrieve_futures.pop(request_id, None)

        # Update the internal states
        ret_stores = self._process_finished_stores(
            finished_stores, finished_req_ids_from_engine
        )

        # the invocation of `get_finished` means that
        # these requests' KV caches are already fully stored.
        # or the requests normally ends without any store.
        if ret_stores:
            self.request_telemetry.on_request_store_finished(
                request_ids_set=ret_stores,
                model_name=self.model_name,
                world_size=self.world_size,
                kv_rank=self.worker_id,
            )

        return ret_stores, finished_retrieves

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Returns the block IDs that failed due to retrieve timeout,
        then clears the internal set.
        """
        errors = self.error_block_ids.copy()
        self.error_block_ids.clear()
        return errors

    def shutdown(self):
        """
        Shutdown the LMCache MP worker adapter
        """
        logger.info("Unregistering kv caches")
        try:
            send_lmcache_request(
                self.mq_client,
                RequestType.UNREGISTER_KV_CACHE,
                [self.instance_id],
            ).result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "LMCache server did not respond to unregister within %ss. "
                "Proceeding with shutdown.",
                self._mq_timeout,
            )

        self.mq_client.close()
        self.request_telemetry.close()

    # Helper functions
    def _update_and_get_finished_store(
        self,
    ) -> set[str]:
        """Converge the internal states about finished stores
        and returns the 'safe finished store request ids' back
        """
        safe_finished_s = self.finished_stores.intersection(self.previously_finished)
        self.finished_stores.difference_update(self.previously_finished)
        self.previously_finished.difference_update(safe_finished_s)

        return safe_finished_s

    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
    ) -> IPCCacheEngineKey:
        """Convert token IDs to an IPC cache engine key"""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
        )
