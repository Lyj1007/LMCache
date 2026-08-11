# SPDX-License-Identifier: Apache-2.0
"""Kunlun KLX_XPU platform primitives.

Kunlun KLX_XPU runs PyTorch through ``torch_xmlir``, which exposes the device as
``torch.cuda`` (``torch.cuda.is_available()`` reports True and tensors report
``device.type == "cuda"``).  This backend therefore reuses the CUDA torch
module name while overriding everything CUDA-specific underneath it.

Two consequences of that shim shape the backend:

* **Device resolution.** Because tensors lie about their device type, registry
  lookups keyed on ``tensor.device.type`` would bind the CUDA spec and hand
  back CUDA IPC handles and CUDA events that Kunlun cannot honour.
  :func:`lmcache.v1.platform._device_detect.normalize_device_type` maps the
  borrowed ``"cuda"`` back onto ``"klx_xpu"`` for every registry lookup.
* **Transfer path.** Kunlun memory is a flat, cross-process addressable
  space, so the LMCache-driven path works -- but via raw device pointers
  (:class:`~lmcache.v1.platform.klx_xpu.ipc_wrapper.KlxXpuPtrIPCWrapper`) rather than
  IPC handles, and ordered through the message queue
  (:class:`~lmcache.v1.platform.klx_xpu.event_ipc.KlxXpuEventIPCBackend`) because
  there is no interprocess event.

Selection: auto-detected ahead of CUDA via
:func:`lmcache.v1.platform._device_detect.is_kunlun_klx_xpu` (see
``_detect_device``), or forced explicitly with ``DEVICE_TYPE=klx_xpu``.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
from lmcache.v1.platform.klx_xpu.pin_memory import KlxXpuPinMemoryBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    from lmcache.v1.platform.base.device_ops import DeviceOps
    from lmcache.v1.platform.base.event_ipc import EventIPCBackend
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class KlxXpuDeviceSpec(DeviceSpec):
    """Kunlun KLX_XPU device specification for the detection registry."""

    _event_backend_cache: "EventIPCBackend | None" = None

    @property
    def device_type(self) -> str:
        return "klx_xpu"

    @property
    def torch_module_name(self) -> str:
        # xmlir exposes the Kunlun device through torch.cuda.
        return "cuda"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        # First Party
        from lmcache.v1.platform.klx_xpu.device_ops import KlxXpuDeviceOps

        return KlxXpuDeviceOps

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        return KlxXpuPinMemoryBackend

    @property
    def ipc_wrapper_cls(self) -> type[DeviceIPCWrapper] | None:
        """Ship KV caches as raw device pointers.

        Kunlun's flat address space is directly addressable across processes,
        so the server maps worker memory without any IPC handle exchange.
        """
        # First Party
        from lmcache.v1.platform.klx_xpu.ipc_wrapper import KlxXpuPtrIPCWrapper

        return KlxXpuPtrIPCWrapper

    @property
    def event_ipc_backend(self) -> "EventIPCBackend":
        """Return the Kunlun event backend (MQ-ordered, no interprocess event)."""
        backend = self._event_backend_cache
        if backend is None:
            # First Party
            from lmcache.v1.platform.klx_xpu.event_ipc import KlxXpuEventIPCBackend

            backend = KlxXpuEventIPCBackend()
            self._event_backend_cache = backend
        return backend

    def is_available(self) -> bool:
        """Return True on a Kunlun KLX_XPU host (``torch_xmlir`` importable).

        Uses the low-level :func:`is_kunlun_klx_xpu` helper so this method never
        imports ``lmcache.__init__`` (avoids the platform import cycle).
        """
        # First Party
        from lmcache.v1.platform._device_detect import is_kunlun_klx_xpu

        return is_kunlun_klx_xpu()

    def is_handle_transfer_available(self) -> bool:
        """Kunlun supports handle transfer via raw device pointers.

        Note this is *not* the CUDA IPC handle path: see
        :attr:`ipc_wrapper_cls` and :attr:`event_ipc_backend`.
        """
        return True

    def create_cache_context(self, *args: Any, **kwargs: Any) -> "BaseCacheContext":
        """Reuse the GPU cache context (KLX_XPU tensors are torch.cuda via xmlir)."""
        # First Party
        from lmcache.v1.platform.cuda.cache_context import GPUCacheContext

        return GPUCacheContext(*args, **kwargs)
