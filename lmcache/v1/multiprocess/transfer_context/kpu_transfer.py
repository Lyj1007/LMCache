# SPDX-License-Identifier: Apache-2.0
"""Kunlun KPU pointer-transfer context.

Kunlun runs the LMCache-driven (handle) path over
:class:`~lmcache.v1.platform.kpu.ipc_wrapper.KpuPtrIPCWrapper`: the server maps
the worker's KV cache by raw device pointer and drives the copies itself. All
of that is inherited unchanged from
:class:`~lmcache.v1.multiprocess.transfer_context.worker_transfer.LMCacheDrivenTransferContext`.

What this subclass adds is **purely a performance concern**. Kunlun has no
interprocess event, so
:class:`~lmcache.v1.platform.kpu.event_ipc.KpuEventIPCBackend` orders transfers
by blocking the host inside ``export_event`` before the MQ message goes out
(see that module for why that is correct). Doing so on the caller's thread
would stall the model forward pass, so submissions are handed to a background
thread that performs the block and the send:

* ``submit_*`` returns immediately with a deferred future;
* one background thread drains a **retrieve-priority** queue -- a blocked
  retrieve stalls token generation, whereas a store only delays eviction;
* :meth:`KpuDevicePtrTransferContext.flush_inflight_stores` waits until every
  queued store has actually been sent, which is what makes it safe for the
  engine to reuse the KV blocks afterwards.

Correctness does not depend on this thread: it is the event backend, not the
queue, that provides the happens-before edge.
"""

# Future
from __future__ import annotations

# Standard
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.utils import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    IPCEvent,
    LMCacheDrivenTransferContext,
)

logger = init_logger(__name__)

#: Warn when a single ``export_event`` host block exceeds this many seconds.
#: A large value here means the worker is producing KV faster than the server
#: drains it, or that a device queue is stuck.
ENV_SLOW_WAIT_SEC = "LMCACHE_KPU_SLOW_WAIT_SEC"
_DEFAULT_SLOW_WAIT_SEC = 5.0

#: How long :meth:`KpuDevicePtrTransferContext.close` waits for the background
#: thread to drain before giving up and logging.
_CLOSE_JOIN_TIMEOUT_SEC = 30.0


def _slow_wait_threshold() -> float:
    """Read the slow-wait diagnostic threshold from the environment."""
    raw = os.environ.get(ENV_SLOW_WAIT_SEC)
    if raw is None:
        return _DEFAULT_SLOW_WAIT_SEC
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using %.1fs",
            ENV_SLOW_WAIT_SEC,
            raw,
            _DEFAULT_SLOW_WAIT_SEC,
        )
        return _DEFAULT_SLOW_WAIT_SEC


class _DeferredMessagingFuture(MessagingFuture):
    """Future whose underlying MQ request is issued from another thread.

    Bridges the gap between ``submit_*`` returning immediately and the
    background thread actually sending the request: waiters first block until
    the real future is bound, then delegate to it. Timeouts are split across
    the two phases against a single deadline so the caller-visible timeout
    keeps its meaning.
    """

    def __init__(self) -> None:
        super().__init__()
        self._bound = threading.Event()
        self._inner: MessagingFuture | None = None
        self._error: BaseException | None = None

    def bind(self, inner: MessagingFuture) -> None:
        """Attach the real future once the request has been sent."""
        self._inner = inner
        self._bound.set()

    def fail(self, error: BaseException) -> None:
        """Mark the submission as failed before it could be sent."""
        self._error = error
        self._bound.set()

    def _resolved_inner(self, timeout: float | None) -> MessagingFuture | None:
        """Block for the bind step; re-raise a submission-time failure."""
        if not self._bound.wait(timeout):
            return None
        if self._error is not None:
            raise self._error
        assert self._inner is not None
        return self._inner

    def query(self) -> bool:
        """Return whether the request was sent *and* has completed."""
        if not self._bound.is_set():
            return False
        if self._error is not None:
            raise self._error
        assert self._inner is not None
        return self._inner.query()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for bind, then for the underlying future."""
        deadline = None if timeout is None else time.monotonic() + timeout
        inner = self._resolved_inner(timeout)
        if inner is None:
            return False
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        return inner.wait(remaining)

    def result(self, timeout: float | None = None) -> Any:
        """Return the server response, waiting for bind first."""
        deadline = None if timeout is None else time.monotonic() + timeout
        inner = self._resolved_inner(timeout)
        if inner is None:
            # First Party
            from lmcache.v1.mp_observability.errors import LMCacheTimeoutError

            raise LMCacheTimeoutError(
                "KPU submission was not dispatched within timeout"
            )
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        return inner.result(remaining)

    def set_result(self, result: Any) -> None:
        raise NotImplementedError(
            "_DeferredMessagingFuture is resolved by its bound future"
        )


@dataclass
class _Submission:
    """One queued KV transfer awaiting its host-side event block."""

    future: _DeferredMessagingFuture
    event: IPCEvent | None
    request_type: RequestType
    build_args: Callable[[bytes], list[Any]]
    is_store: bool


class KpuDevicePtrTransferContext(LMCacheDrivenTransferContext):
    """LMCache-driven transfer for Kunlun, with submissions off the hot path."""

    def __init__(self) -> None:
        super().__init__()
        self._cv = threading.Condition()
        self._q_retrieve: deque[_Submission] = deque()
        self._q_store: deque[_Submission] = deque()
        self._inflight_stores = 0
        self._closing = False
        self._thread: threading.Thread | None = None
        self._slow_wait_sec = _slow_wait_threshold()

    # ------------------------------------------------------------------
    # Registration / teardown
    # ------------------------------------------------------------------

    def register(self, *args: Any, **kwargs: Any) -> None:
        """Register with the server, then start the submission thread."""
        super().register(*args, **kwargs)
        self._start_thread()

    def register_q(self, *args: Any, **kwargs: Any) -> None:
        """Register Q caches, then start the submission thread."""
        super().register_q(*args, **kwargs)
        self._start_thread()

    def _start_thread(self) -> None:
        """Start the background submission thread once."""
        with self._cv:
            if self._thread is not None or self._closing:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="lmcache-kpu-transfer",
                daemon=True,
            )
            self._thread.start()

    def close(self) -> None:
        """Drain queued submissions, stop the thread, release server state."""
        with self._cv:
            self._closing = True
            self._cv.notify_all()
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=_CLOSE_JOIN_TIMEOUT_SEC)
            if thread.is_alive():
                logger.warning(
                    "KPU transfer thread did not drain within %.0fs; closing anyway",
                    _CLOSE_JOIN_TIMEOUT_SEC,
                )
        super().close()

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def _enqueue(
        self,
        request_type: RequestType,
        build_args: Callable[[bytes], list[Any]],
        event: IPCEvent | None,
        is_store: bool,
        caller: str,
    ) -> MessagingFuture:
        """Queue a submission for the background thread."""
        if (
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "KPU pointer transfer context is not registered. "
                f"Call register() before {caller}()."
            )
        future = _DeferredMessagingFuture()
        item = _Submission(
            future=future,
            event=event,
            request_type=request_type,
            build_args=build_args,
            is_store=is_store,
        )
        with self._cv:
            if self._closing:
                raise RuntimeError(
                    f"KPU pointer transfer context is closing; {caller}() rejected."
                )
            if is_store:
                self._q_store.append(item)
            else:
                self._q_retrieve.append(item)
            self._cv.notify()
        return future

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Queue a pointer-based store; returns before the event is drained."""
        return self._enqueue(
            request_type=RequestType.STORE,
            build_args=lambda handle: [key, instance_id, block_ids, handle],
            event=event,
            is_store=True,
            caller="submit_store",
        )

    def submit_q_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _q_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Queue a pointer-based Q store."""
        return self._enqueue(
            request_type=RequestType.STORE_Q,
            build_args=lambda handle: [key, instance_id, block_ids, handle],
            event=event,
            is_store=True,
            caller="submit_q_store",
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
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Queue a pointer-based retrieve (served ahead of pending stores)."""
        return self._enqueue(
            request_type=RequestType.RETRIEVE,
            build_args=lambda handle: [
                key,
                instance_id,
                block_ids,
                handle,
                skip_first_n_tokens,
            ],
            event=event,
            is_store=False,
            caller="submit_retrieve",
        )

    def flush_inflight_stores(self) -> None:
        """Block until every queued store has been dispatched.

        The engine calls this before reusing KV blocks. Returning early would
        let the block be overwritten while the server is still reading it.
        """
        with self._cv:
            while self._q_store or self._inflight_stores:
                self._cv.wait()

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _next_submission(self) -> _Submission | None:
        """Pop the next submission, retrieves first; ``None`` to stop."""
        with self._cv:
            while True:
                if self._q_retrieve:
                    return self._q_retrieve.popleft()
                if self._q_store:
                    item = self._q_store.popleft()
                    self._inflight_stores += 1
                    return item
                if self._closing:
                    return None
                self._cv.wait()

    def _run(self) -> None:
        """Drain the queues: block on each event, then send its request."""
        while True:
            item = self._next_submission()
            if item is None:
                break
            try:
                self._dispatch(item)
            except BaseException as exc:  # noqa: BLE001 - surfaced via future
                logger.exception("KPU %s submission failed", item.request_type)
                item.future.fail(exc)
            finally:
                if item.is_store:
                    with self._cv:
                        self._inflight_stores -= 1
                        self._cv.notify_all()

    def _dispatch(self, item: _Submission) -> None:
        """Drain the producer event, send the request, bind the future."""
        assert self._event_backend is not None
        assert self._send_request is not None
        assert self._mq_client is not None
        assert self._device is not None

        # Blocks until the producer event completes -- this is the Kunlun
        # happens-before edge, see KpuEventIPCBackend.
        started = time.monotonic()
        handle = self._event_backend.export_event(item.event, self._device)
        waited = time.monotonic() - started
        if waited >= self._slow_wait_sec:
            logger.warning(
                "KPU %s waited %.2fs for its device event "
                "(threshold %.1fs, set %s to adjust); "
                "queues: retrieve=%d store=%d",
                item.request_type,
                waited,
                self._slow_wait_sec,
                ENV_SLOW_WAIT_SEC,
                len(self._q_retrieve),
                len(self._q_store),
            )

        raw_future = self._send_request(
            self._mq_client,
            item.request_type,
            item.build_args(handle),
        )
        item.future.bind(raw_future.to_device_future(device=self._device))
