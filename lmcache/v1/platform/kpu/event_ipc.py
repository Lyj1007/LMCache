# SPDX-License-Identifier: Apache-2.0
"""Kunlun KPU device-event IPC backend.

Kunlun has **no interprocess event**: ``torch_xmlir`` exposes
``torch.cuda.Event`` but not a usable ``ipc_handle()`` /
``from_ipc_handle()`` pair, so the CUDA-style
:class:`~lmcache.v1.platform.base.event_ipc.DefaultEventIPCBackend` cannot
order a cross-process transfer here.

Rather than give up the LMCache-driven (pointer) path, this backend moves the
ordering guarantee from the *device* to the *message queue*:

* :meth:`KpuEventIPCBackend.export_event` **synchronizes the event locally**
  and then emits an empty handle.  Whoever exports has, by the time the MQ
  message leaves, finished the device work the peer would have waited on.
* :meth:`KpuEventIPCBackend.import_event` returns an already-elapsed sentinel,
  so peer-side waits are correct no-ops.

The MQ message itself becomes the happens-before edge, and it works in both
directions with the same primitive:

* **store**: the worker synchronizes the forward event before sending, so the
  server only ever reads a fully written KV cache;
* **retrieve**: the server synchronizes its copy event before replying, so the
  worker only ever reads a fully written KV cache.

The cost is a host-side block at export time.  Correctness does not depend on
who blocks, so
:class:`~lmcache.v1.multiprocess.transfer_context.kpu_transfer.KpuDevicePtrTransferContext`
moves the worker-side block off the forward thread; this backend stays the
single place where the ordering semantics live.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

#: Wire value for "no interprocess event"; Kunlun orders via the MQ message.
_EMPTY_HANDLE = b""


class _CompletedEvent:
    """Stand-in for a remote event that is guaranteed to have elapsed.

    :meth:`KpuEventIPCBackend.export_event` blocks until the exported event
    completes, so by the time a peer imports the handle there is nothing left
    to wait for. Implements the duck-typed event surface consumed by
    :class:`~lmcache.v1.multiprocess.futures.DeviceMessagingFuture` and the
    server transfer loop.
    """

    __slots__ = ()

    def record(self, stream: object | None = None) -> None:
        """No-op: the event has already elapsed."""

    def wait(self, stream: object | None = None) -> None:
        """No-op: nothing to order against."""

    def query(self) -> bool:
        """Always ``True``: the event has already elapsed."""
        return True

    def synchronize(self) -> None:
        """No-op: the host block already happened at export time."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<KpuCompletedEvent>"


class KpuEventIPCBackend:
    """Event IPC backend for Kunlun KPU, ordered through the message queue.

    Satisfies the
    :class:`~lmcache.v1.platform.base.event_ipc.EventIPCBackend` protocol.
    Local (intra-process) event operations delegate to ``torch.cuda`` -- which
    is what xmlir drives on Kunlun -- while the cross-process operations are
    replaced by the synchronize-on-export scheme described in the module
    docstring.
    """

    device_type: ClassVar[str] = "kpu"

    def check_event_support(self, device: object) -> None:
        """Always succeeds.

        The base backend fails closed when ``Event.from_ipc_handle`` is
        missing, which is exactly the Kunlun situation. Here the ordering
        guarantee is provided by :meth:`export_event` instead of by device
        event IPC, so there is nothing to reject.

        Args:
            device: Unused; accepted for protocol compatibility.
        """

    def create_event(self, device: object) -> object:
        """Create a plain (non-interprocess) device event.

        Args:
            device: Unused; the event is created on the current device.

        Returns:
            A ``torch.cuda.Event`` driven by xmlir on Kunlun hardware.
        """
        # Third Party
        import torch

        return torch.cuda.Event()

    def export_event(self, event: object, device: object) -> bytes:
        """Block until ``event`` completes, then emit an empty handle.

        This host-side block *is* the cross-process ordering guarantee: the
        caller must not publish the accompanying MQ message until the device
        work has landed, because the peer has no way to wait on the event.

        Args:
            event: Backend-native event to drain. ``None`` is tolerated so
                callers need not special-case eventless submissions.
            device: Unused; accepted for protocol compatibility.

        Returns:
            An empty handle -- there is nothing for the peer to import.
        """
        if event is not None:
            event.synchronize()  # type: ignore[attr-defined]
        return _EMPTY_HANDLE

    def import_event(self, handle: bytes, device: object) -> object:
        """Return an already-elapsed sentinel event.

        Args:
            handle: Ignored; :meth:`export_event` never produces payload.
            device: Unused; accepted for protocol compatibility.

        Returns:
            A :class:`_CompletedEvent` whose waits are no-ops.
        """
        return _CompletedEvent()

    def record_event(self, event: object, stream: object) -> None:
        """Record ``event`` on ``stream`` (local semantics)."""
        event.record(stream)  # type: ignore[attr-defined]

    def wait_event(self, event: object, stream: object) -> None:
        """Make ``stream`` wait for ``event`` (no-op for imported events)."""
        event.wait(stream)  # type: ignore[attr-defined]

    def query_event(self, event: object) -> bool:
        """Return whether ``event`` has completed."""
        return bool(event.query())  # type: ignore[attr-defined]

    def synchronize_event(self, event: object, device: object) -> None:
        """Block the host until ``event`` completes."""
        event.synchronize()  # type: ignore[attr-defined]
