# SPDX-License-Identifier: Apache-2.0
"""Worker-side XPU device-pointer transfer context (XPU offload v2 §8.2).

This module implements :class:`XPUDevicePtrTransferContext`, the worker-side
counterpart of :class:`lmcache.v1.multiprocess.modules.xpu_transfer.XpuTransferModule`.

Why a dedicated transfer context instead of reusing
:class:`HandleTransferContext` or :class:`DataTransferContext`:

- The CUDA :class:`HandleTransferContext` relies on cross-process CUDA events
  (``Event.from_ipc_handle``), which is **not** available on Kunlun XPU even
  though ``torch.cuda`` is what xmlir exposes (see ``lmcache.is_kunlun_xpu``).
  The XPU path therefore cannot return a :class:`CUDAMessagingFuture`.
- The non-CUDA :class:`DataTransferContext` performs gather/scatter on the
  worker side. XPU exposes a flat physical device address space and the
  cross-process gather/scatter is performed by the **server** using the
  ``__cuda_array_interface__`` peer-pointer pattern (XPU offload v2 §3.8 /
  §6) — the worker simply needs to (a) make sure its compute stream has
  finished before the server reads the source memory, and (b) broker the
  store/retrieve request over the MQ.

The implementation follows the threading model in §3.1 and §8.1:

- forward main thread submits ``(event, key, block_ids)`` and immediately
  returns a :class:`MessagingFuture`.
- a background thread consumes the queue, calls ``event.synchronize()``
  in-process, then issues the MQ request synchronously and resolves the
  future with the server's ack.
"""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional
import queue
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType, init_logger
from lmcache.v1.gpu_connector.utils import LayoutHints, is_mla
from lmcache.v1.multiprocess.custom_types import (
    RegisterXpuContextPayload,
    XpuGroupView,
    XpuLayerHandle,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    get_engine_group_indices,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import RegisterXpuContextResponse
from lmcache.v1.multiprocess.transfer_context.base import compute_kv_layout
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    IPCEvent,
    SendRequest,
    TransferContext,
)
from lmcache.v1.multiprocess.transfer_context.xpu_broadcast import (
    BroadcastFn,
    XpuBroadcastCoordinator,
)

logger = init_logger(__name__)


# Sentinel used to wake the BG thread for shutdown without dropping
# in-flight queue items on the floor.
_SHUTDOWN_SENTINEL: Any = object()

# Maximum time in seconds to wait for the BG thread to drain on close.
# The BG thread serializes one MQ request at a time, and each MQ request
# is itself bounded by ``mq_timeout``; this cap is just an outer guard
# against unexpected hangs in the join itself.
_BG_JOIN_TIMEOUT_SECONDS: float = 60.0

# Bound on pending submit-queue items. Acts as back-pressure on the
# forward main thread so a stuck server cannot translate into unbounded
# memory growth on the worker. The cap is intentionally generous so
# normal pipelining (multi-chunk request fan-out) never blocks.
_SUBMIT_QUEUE_MAXSIZE: int = 1024

# How long ``close()`` waits for the shutdown sentinel slot in the
# bounded submit queue. After this, ``_closed`` is already set and the
# BG loop exits via its ``_closed`` + empty-queue shortcut.
_SUBMIT_QUEUE_SHUTDOWN_PUT_SECONDS: float = 5.0

# Polling cadence for the BG loop's blocking ``get``. Keeps the thread
# responsive to ``_closed`` even when ``put`` of the shutdown sentinel
# raced with a full queue.
_BG_LOOP_POLL_SECONDS: float = 1.0


@dataclass
class _BgRequest:
    """One enqueued worker-to-server transfer request.

    Attributes:
        request_type: ``STORE_XPU`` or ``RETRIEVE_XPU``.
        event: Compute-stream event recorded by the forward main thread.
            The BG thread synchronizes on it before issuing the MQ request
            so the server can safely read the worker's KV memory.
        key: IPC cache engine key for the chunk range.
        instance_id: Worker process instance id.
        block_ids: Per-group paged block ids.
        future: Result future returned to the caller.
        skip_blocks_per_group: Per-group leading blocks to skip
            (RETRIEVE only; ``None`` for STORE).
    """

    request_type: RequestType
    event: IPCEvent
    key: Any
    instance_id: int
    block_ids: list[list[int]]
    future: MessagingFuture[bool]
    skip_blocks_per_group: Optional[list[int]]


def _build_layer_handles(
    kv_caches: dict[str, torch.Tensor],
    group_views: Sequence[EngineGroupInfo],
) -> list[XpuLayerHandle]:
    """Build cross-process layer-pointer descriptors from worker KV caches.

    Args:
        kv_caches: Worker KV cache tensors keyed by layer name.
        group_views: LMCache group metadata; selects the engine group id
            each layer reports to the server.

    Returns:
        A list of :class:`XpuLayerHandle` in dict-insertion order.

    Raises:
        ValueError: If ``group_views`` references a layer index outside the
            registered ``kv_caches`` range.
    """
    layer_items = list(kv_caches.items())
    num_layers = len(layer_items)
    engine_group_indices = get_engine_group_indices(group_views, num_layers)

    handles: list[XpuLayerHandle] = []
    for idx, (name, tensor) in enumerate(layer_items):
        dtype_str = str(tensor.dtype).replace("torch.", "")
        group_id = (
            int(engine_group_indices[idx])
            if engine_group_indices is not None
            else 0
        )
        handles.append(
            XpuLayerHandle(
                layer_name=name,
                data_ptr=int(tensor.data_ptr()),
                shape=list(tensor.shape),
                dtype_str=dtype_str,
                group_id=group_id,
            )
        )
    return handles


def _build_group_views(
    group_views: Sequence[EngineGroupInfo],
    layout_hints: LayoutHints | None,
    block_size: int,
    blocks_in_chunk: int,
    use_mla_global: bool,
) -> list[XpuGroupView]:
    """Build per-group metadata for the registration payload.

    Args:
        group_views: LMCache-owned engine KV cache group metadata.
        layout_hints: Optional inference-engine layout hints. When the
            ``per_layer_storage_blocks_per_chunk`` key is present it
            overrides ``blocks_in_chunk`` per group.
        block_size: Tokens per paged block (uniform across layers in
            current models).
        blocks_in_chunk: Default LMCache blocks-per-chunk (used when
            ``per_layer_storage_blocks_per_chunk`` is absent).
        use_mla_global: ``is_mla`` for the worker KV format. The XPU
            registration treats this as group-uniform until per-group
            MLA detection is wired up (XPU offload v2 §7.2).

    Returns:
        One :class:`XpuGroupView` per LMCache group, or a single
        default group when ``group_views`` is empty.
    """
    per_layer_storage_blocks: list[int] | None = None
    if layout_hints is not None:
        raw = layout_hints.get("per_layer_storage_blocks_per_chunk")
        if raw is not None:
            per_layer_storage_blocks = [int(v) for v in raw]

    if not group_views:
        return [
            XpuGroupView(
                group_id=0,
                block_size=block_size,
                blocks_per_chunk=int(blocks_in_chunk),
                is_mla=use_mla_global,
            )
        ]

    out: list[XpuGroupView] = []
    for group in group_views:
        if (
            per_layer_storage_blocks is not None
            and group.layer_indices
            and group.layer_indices[0] < len(per_layer_storage_blocks)
        ):
            blocks_per_chunk = int(
                per_layer_storage_blocks[group.layer_indices[0]]
            )
        else:
            blocks_per_chunk = int(blocks_in_chunk)
        out.append(
            XpuGroupView(
                group_id=int(group.engine_group_id),
                block_size=block_size,
                blocks_per_chunk=blocks_per_chunk,
                is_mla=use_mla_global,
            )
        )
    return out


class XPUDevicePtrTransferContext(TransferContext):
    """Worker-side XPU device-pointer transfer context.

    Forwards store/retrieve requests to the server via MQ, never blocks
    the forward main thread on ``event.synchronize()``. See module
    docstring for the threading rationale.
    """

    def __init__(self) -> None:
        self._mq_client: MessageQueueClient | None = None
        self._send_request: SendRequest | None = None
        self._mq_timeout: float = 0.0
        self._instance_id: int = 0
        self._tp_rank: int = 0
        self._tp_size: int = 1
        self._broadcast_buffer: torch.Tensor | None = None
        self._broadcast_buffer_bytes: int = 0
        self._mla_group_ids: tuple[int, ...] = ()
        self._retrieve_queue: "queue.Queue[Any]" = queue.Queue(
            maxsize=_SUBMIT_QUEUE_MAXSIZE
        )
        self._store_queue: "queue.Queue[Any]" = queue.Queue(
            maxsize=_SUBMIT_QUEUE_MAXSIZE
        )
        self._bg_thread: threading.Thread | None = None
        self._closed = threading.Event()

    @property
    def broadcast_buffer(self) -> torch.Tensor | None:
        """Worker-side broadcast buffer (XPU offload v2 §9.4).

        Returns:
            The pre-allocated buffer used for MLA TP broadcast, or
            ``None`` when MLA broadcast is not used by this worker.
        """
        return self._broadcast_buffer

    @property
    def tp_rank(self) -> int:
        """Worker's TP rank as reported during registration."""
        return self._tp_rank

    @property
    def tp_size(self) -> int:
        """Worker's TP size as reported during registration."""
        return self._tp_size

    @property
    def mla_group_ids(self) -> tuple[int, ...]:
        """Engine group ids classified as MLA at registration time.

        Empty when no group is MLA. Used by the connector to decide
        which retrieve hooks should publish through the broadcast
        coordinator (XPU offload v2 §9.4).
        """
        return self._mla_group_ids

    def make_broadcast_coordinator(
        self, broadcast_fn: BroadcastFn
    ) -> XpuBroadcastCoordinator | None:
        """Build an MLA TP broadcast coordinator over this rank's buffer.

        The coordinator is intentionally caller-owned (each connector
        retrieve hook may want to bind a different broadcast callable)
        and never cached on the context.

        Args:
            broadcast_fn: TP-group broadcast callable, e.g.
                ``vllm.distributed.parallel_state.get_tp_group().broadcast``.

        Returns:
            A configured :class:`XpuBroadcastCoordinator`, or ``None``
            when the worker registered without an MLA group (server
            reports ``broadcast_buffer_bytes == 0``).

        Raises:
            RuntimeError: If invoked before :meth:`register`.
        """
        if self._broadcast_buffer is None or self._broadcast_buffer_bytes <= 0:
            return None
        if self._mq_client is None:
            raise RuntimeError(
                "XPU transfer context is not registered. "
                "Call register() before make_broadcast_coordinator()."
            )
        return XpuBroadcastCoordinator(
            broadcast_fn=broadcast_fn,
            broadcast_buffer=self._broadcast_buffer,
            tp_rank=self._tp_rank,
            tp_size=self._tp_size,
        )

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
        tp_rank: int = 0,
        tp_size: int = 1,
        l1_pool_request_size: int = 0,
    ) -> None:
        """Register XPU KV caches with the server.

        Sends ``REGISTER_XPU_KV_CACHE`` carrying every layer's
        ``data_ptr()`` so the server can wrap the worker memory via
        ``__cuda_array_interface__``. Allocates the worker-side broadcast
        buffer reported by the server and starts the BG worker thread.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            model_name: Model name used by cache keys.
            world_size: KV world size.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            mq_client: Message queue client used to communicate with the server.
            mq_timeout: Timeout in seconds for synchronous request waits.
            send_request: Request sender callable used to issue MQ requests.
            layout_hints: Inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.
            engine_type: Serving engine that produced the caches. Accepted
                to satisfy the base interface; the XPU device-pointer path
                carries layer pointers directly and does not branch on it.
            tp_rank: TP rank of this worker within the inference engine
                TP group. Defaults to ``0`` when callers don't track TP.
            tp_size: TP size of the inference engine TP group. Defaults
                to ``1``.
            l1_pool_request_size: Bytes of L1 SHM pool the worker would
                like the server to back. ``0`` means "use server default".

        Raises:
            TimeoutError: If the server does not ACK within ``mq_timeout``.
            RuntimeError: If ``kv_caches`` is empty or has tensors on
                multiple devices.
            ValueError: If ``tp_rank`` is outside ``[0, tp_size)`` or
                ``tp_size`` is non-positive.
        """
        del engine_type  # unused: the XPU path ships raw layer pointers
        if not kv_caches:
            raise RuntimeError("XPU transfer context requires non-empty kv_caches")
        if int(tp_size) <= 0:
            raise ValueError(
                f"XPU transfer context requires tp_size > 0, got {tp_size}"
            )
        if not 0 <= int(tp_rank) < int(tp_size):
            raise ValueError(
                f"XPU transfer context tp_rank={tp_rank} out of range "
                f"[0, {tp_size})"
            )

        self._mq_client = mq_client
        self._send_request = send_request
        self._mq_timeout = float(mq_timeout)
        self._instance_id = int(instance_id)
        self._tp_rank = int(tp_rank)
        self._tp_size = int(tp_size)

        try:
            block_size, _num_layers, _hidden_dim, _dtype_str, gpu_kv_format = (
                compute_kv_layout(kv_caches, layout_hints=layout_hints)
            )
            use_mla_global = is_mla(gpu_kv_format)

            layer_handles = _build_layer_handles(kv_caches, engine_group_infos)
            groups = _build_group_views(
                engine_group_infos,
                layout_hints,
                block_size=block_size,
                blocks_in_chunk=blocks_in_chunk,
                use_mla_global=use_mla_global,
            )

            payload = RegisterXpuContextPayload(
                instance_id=self._instance_id,
                model_name=model_name,
                world_size=int(world_size),
                tp_rank=self._tp_rank,
                tp_size=self._tp_size,
                layer_handles=layer_handles,
                groups=groups,
                gpu_kv_format=int(gpu_kv_format),
                l1_pool_request_size=int(l1_pool_request_size),
            )

            future = send_request(
                mq_client,
                RequestType.REGISTER_XPU_KV_CACHE,
                [payload],
            )
            response = future.result(timeout=self._mq_timeout)
            if not isinstance(response, RegisterXpuContextResponse):
                raise RuntimeError(
                    "REGISTER_XPU_KV_CACHE returned unexpected response type: "
                    f"{type(response).__name__}"
                )

            self._broadcast_buffer_bytes = int(response.broadcast_buffer_bytes)
            if self._broadcast_buffer_bytes > 0:
                self._broadcast_buffer = self._allocate_broadcast_buffer(
                    self._broadcast_buffer_bytes
                )

            self._mla_group_ids = tuple(
                int(g.group_id) for g in groups if g.is_mla
            )

            logger.info(
                "Worker XPU transfer context registered "
                "(instance_id=%d tp=%d/%d num_layers=%d num_groups=%d "
                "broadcast_bytes=%d mla_groups=%s shm=%s shm_size=%d)",
                self._instance_id,
                self._tp_rank,
                self._tp_size,
                len(layer_handles),
                len(groups),
                self._broadcast_buffer_bytes,
                self._mla_group_ids,
                response.l1_shm_name or "<none>",
                response.l1_shm_size,
            )

            self._closed.clear()
            self._bg_thread = threading.Thread(
                target=self._bg_loop,
                name=f"xpu-transfer-bg-{self._instance_id}",
                daemon=True,
            )
            self._bg_thread.start()
        except BaseException:
            # Roll back partial state so a follow-up close()/register() does
            # not try to drive a half-initialized context.
            self._mq_client = None
            self._send_request = None
            self._broadcast_buffer = None
            self._broadcast_buffer_bytes = 0
            self._mla_group_ids = ()
            self._bg_thread = None
            raise

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture[bool]:
        """Enqueue a STORE_XPU request and return a future for its ACK.

        Args:
            _request_id: External request id (unused; key carries it).
            key: Cache key for the store range.
            instance_id: Worker process instance id.
            _kv_caches: Worker KV caches (unused; server reads via the
                pointers it captured during registration).
            block_ids: Per-group paged block ids.
            event: Compute-stream event recorded by the caller.
            _blocks_in_chunk: Unused; carried for interface compatibility.

        Returns:
            A future that resolves to ``True`` on server-acked success
            or ``False`` on server-side failure.

        Raises:
            RuntimeError: If :meth:`register` was not called first or
                the context has been closed.
        """
        return self._enqueue(
            request_type=RequestType.STORE_XPU,
            key=key,
            instance_id=instance_id,
            block_ids=block_ids,
            event=event,
            skip_blocks_per_group=None,
        )

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
        skip_blocks_per_group: list[int] | None = None,
    ) -> MessagingFuture[bool]:
        """Enqueue a RETRIEVE_XPU request and return a future for its ACK.

        Args:
            _request_id: External request id (unused; key carries it).
            key: Cache key for the retrieve range.
            instance_id: Worker process instance id.
            _kv_caches: Worker KV caches (unused; server writes via the
                pointers it captured during registration).
            block_ids: Per-group paged block ids.
            event: Compute-stream event recorded by the caller.
            _blocks_in_chunk: Unused; carried for interface compatibility.
            skip_blocks_per_group: Per-group leading blocks to skip
                (APC overlap guard). ``None`` / empty = no skip.

        Returns:
            A future that resolves to ``True`` on server-acked success
            or ``False`` on cache miss / server-side failure.

        Raises:
            RuntimeError: If :meth:`register` was not called first or
                the context has been closed.
        """
        return self._enqueue(
            request_type=RequestType.RETRIEVE_XPU,
            key=key,
            instance_id=instance_id,
            block_ids=block_ids,
            event=event,
            skip_blocks_per_group=list(skip_blocks_per_group or []),
        )

    def close(self) -> None:
        """Drain the BG thread, unregister, and release the broadcast buffer.

        Idempotent: subsequent calls are no-ops.

        Notes:
            On a stuck server the BG thread can outlive the join timeout.
            The thread is held as ``daemon=True`` so the process exit
            still terminates it, but the references on ``self`` are
            preserved for diagnostics so any in-flight ``MessagingFuture``
            still has a valid pointer to inspect.
        """
        if self._closed.is_set():
            return
        self._closed.set()
        # Send shutdown sentinel to both queues
        for q in (self._retrieve_queue, self._store_queue):
            try:
                q.put(
                    _SHUTDOWN_SENTINEL,
                    timeout=_SUBMIT_QUEUE_SHUTDOWN_PUT_SECONDS,
                )
            except queue.Full:
                pass

        thread = self._bg_thread
        joined_cleanly = True
        if thread is not None and thread.is_alive():
            thread.join(timeout=_BG_JOIN_TIMEOUT_SECONDS)
            if thread.is_alive():
                joined_cleanly = False
                logger.warning(
                    "XPU transfer BG thread did not exit within %.1fs; "
                    "leaving it as a daemon thread (instance_id=%d, "
                    "queue_size=%d)",
                    _BG_JOIN_TIMEOUT_SECONDS,
                    self._instance_id,
                    self._retrieve_queue.qsize() + self._store_queue.qsize(),
                )

        # Best-effort UNREGISTER_XPU_KV_CACHE; the server may already be
        # gone (e.g. during shutdown), so we swallow timeouts here.
        if (
            joined_cleanly
            and self._mq_client is not None
            and self._send_request is not None
        ):
            try:
                future = self._send_request(
                    self._mq_client,
                    RequestType.UNREGISTER_XPU_KV_CACHE,
                    [self._instance_id],
                )
                future.result(timeout=self._mq_timeout)
            except (TimeoutError, RuntimeError):
                logger.warning(
                    "UNREGISTER_XPU_KV_CACHE for instance_id=%d failed/timed out",
                    self._instance_id,
                    exc_info=True,
                )

        if joined_cleanly:
            self._bg_thread = None
            self._mq_client = None
            self._send_request = None
            self._broadcast_buffer = None
            self._broadcast_buffer_bytes = 0
            self._mla_group_ids = ()

    def _enqueue(
        self,
        request_type: RequestType,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event: IPCEvent,
        skip_blocks_per_group: Optional[list[int]],
    ) -> MessagingFuture[bool]:
        """Push a request onto the BG queue and return its future.

        Raises:
            RuntimeError: If ``register()`` has not been called or the
                context is closed, or if the BG thread has died.
        """
        if self._mq_client is None or self._send_request is None:
            raise RuntimeError(
                "XPU transfer context is not registered. "
                "Call register() before submit_store()/submit_retrieve()."
            )
        if self._closed.is_set():
            raise RuntimeError(
                "XPU transfer context is closed. "
                "Cannot submit new requests."
            )
        thread = self._bg_thread
        if thread is None or not thread.is_alive():
            raise RuntimeError(
                "XPU transfer BG thread is not running; "
                "the worker cannot accept new submissions."
            )

        future: MessagingFuture[bool] = MessagingFuture()
        req = _BgRequest(
            request_type=request_type,
            event=event,
            key=key,
            instance_id=int(instance_id),
            block_ids=block_ids,
            future=future,
            skip_blocks_per_group=skip_blocks_per_group,
        )
        target_queue = (
            self._retrieve_queue
            if request_type == RequestType.RETRIEVE_XPU
            else self._store_queue
        )
        try:
            target_queue.put(
                req,
                timeout=self._mq_timeout if self._mq_timeout > 0 else None,
            )
        except queue.Full as exc:
            raise RuntimeError(
                "XPU transfer submit queue is full "
                f"(maxsize={_SUBMIT_QUEUE_MAXSIZE}); "
                "the server is likely stuck."
            ) from exc
        return future

    def _bg_loop(self) -> None:
        """Consume dual queues with RETRIEVE priority and resolve futures.

        RETRIEVE requests are never blocked by queued STORE requests.
        The loop always checks _retrieve_queue first; a STORE is only
        started when no RETRIEVE is pending.
        """
        while True:
            if (
                self._closed.is_set()
                and self._retrieve_queue.empty()
                and self._store_queue.empty()
            ):
                return
            # Priority: always prefer retrieve over store
            req = None
            try:
                req = self._retrieve_queue.get_nowait()
            except queue.Empty:
                try:
                    req = self._store_queue.get_nowait()
                except queue.Empty:
                    try:
                        req = self._retrieve_queue.get(
                            timeout=_BG_LOOP_POLL_SECONDS
                        )
                    except queue.Empty:
                        continue
            if req is _SHUTDOWN_SENTINEL:
                self._drain_queue_with_failures()
                return
            if self._closed.is_set():
                req.future.set_result(False)
                continue
            try:
                req.event.synchronize()
            except Exception:
                logger.exception(
                    "XPU BG: event.synchronize() failed for request_type=%s; "
                    "marking future failed",
                    req.request_type.name,
                )
                req.future.set_result(False)
                continue

            assert self._mq_client is not None
            assert self._send_request is not None

            try:
                if req.request_type == RequestType.STORE_XPU:
                    payloads: list[Any] = [
                        req.key,
                        req.instance_id,
                        req.block_ids,
                    ]
                elif req.request_type == RequestType.RETRIEVE_XPU:
                    payloads = [
                        req.key,
                        req.instance_id,
                        req.block_ids,
                        list(req.skip_blocks_per_group or []),
                    ]
                else:
                    raise RuntimeError(
                        f"XPU BG: unsupported request type {req.request_type}"
                    )

                mq_future = self._send_request(
                    self._mq_client,
                    req.request_type,
                    payloads,
                )
                ok = mq_future.result(timeout=self._mq_timeout)
                req.future.set_result(bool(ok))
            except TimeoutError:
                logger.warning(
                    "XPU BG: %s request_id=%s timed out after %.1fs",
                    req.request_type.name,
                    getattr(req.key, "request_id", "<unknown>"),
                    self._mq_timeout,
                )
                req.future.set_result(False)
            except Exception:
                logger.exception(
                    "XPU BG: %s request_id=%s failed",
                    req.request_type.name,
                    getattr(req.key, "request_id", "<unknown>"),
                )
                req.future.set_result(False)

    def _drain_queue_with_failures(self) -> None:
        """Resolve any outstanding queue items as failures during shutdown."""
        for q in (self._retrieve_queue, self._store_queue):
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
                if item is _SHUTDOWN_SENTINEL:
                    continue
                req: _BgRequest = item
                if not req.future.is_done_.is_set():
                    req.future.set_result(False)

    @staticmethod
    def _allocate_broadcast_buffer(nbytes: int) -> torch.Tensor:
        """Allocate the per-rank broadcast buffer (XPU offload v2 §9.4).

        Args:
            nbytes: Buffer size in bytes (server-reported).

        Returns:
            A 1-D ``uint8`` tensor of length ``nbytes`` on the current
            XPU device.
        """
        device_index = torch_dev.current_device()
        device = torch.device(f"{torch_device_type}:{device_index}")
        return torch.empty(int(nbytes), dtype=torch.uint8, device=device)
