# SPDX-License-Identifier: Apache-2.0
"""Kunlun KPU platform primitives.

Kunlun KPU runs PyTorch through ``torch_xmlir``, which exposes the device as
``torch.cuda`` (``torch.cuda.is_available()`` reports True and tensors report
``device.type == "cuda"``).  This backend therefore reuses the CUDA torch
module name while providing Kunlun-specific host memory ops and forcing the
engine-driven (data) multiprocess transfer path -- the CUDA IPC/handle path
does not work on Kunlun.

Selection: auto-detected ahead of CUDA via
:func:`lmcache.v1.platform._device_detect.is_kunlun_xpu` (see
``_detect_device``), or forced explicitly with ``DEVICE_TYPE=kpu``.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
from lmcache.v1.platform.kpu.pin_memory import KpuPinMemoryBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    from lmcache.v1.platform.base.device_ops import DeviceOps
    from lmcache.v1.platform.base.event_ipc import EventIPCBackend

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class KpuDeviceSpec(DeviceSpec):
    """Kunlun KPU device specification for the detection registry."""

    _event_backend_cache: "EventIPCBackend | None" = None

    @property
    def device_type(self) -> str:
        return "kpu"

    @property
    def torch_module_name(self) -> str:
        # xmlir exposes the Kunlun device through torch.cuda.
        return "cuda"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        # First Party
        from lmcache.v1.platform.kpu.device_ops import KpuDeviceOps

        return KpuDeviceOps

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        return KpuPinMemoryBackend

    @property
    def event_ipc_backend(self) -> "EventIPCBackend":
        """Return the KPU event IPC backend (torch.cuda events via xmlir)."""
        backend = self._event_backend_cache
        if backend is None:
            # Third Party
            import torch

            # First Party
            from lmcache.v1.platform.base.event_ipc import DefaultEventIPCBackend

            backend = DefaultEventIPCBackend(
                event_module=torch.cuda,
                device_type=self.device_type,
            )
            self._event_backend_cache = backend
        return backend

    def is_available(self) -> bool:
        """Return True on a Kunlun KPU host (``torch_xmlir`` importable).

        Uses the low-level :func:`is_kunlun_xpu` helper so this method never
        imports ``lmcache.__init__`` (avoids the platform import cycle).
        """
        # First Party
        from lmcache.v1.platform._device_detect import is_kunlun_xpu

        return is_kunlun_xpu()

    def is_handle_transfer_available(self) -> bool:
        """Kunlun cannot use the CUDA IPC/handle transfer path.

        Returning False routes AUTO multiprocess transfer to the
        engine-driven (data) path, which is the correctness baseline for
        Kunlun KPU offload.
        """
        return False

    def create_cache_context(self, *args: Any, **kwargs: Any) -> "BaseCacheContext":
        """Reuse the GPU cache context (KPU tensors are torch.cuda via xmlir)."""
        # First Party
        from lmcache.v1.platform.cuda.cache_context import GPUCacheContext

        return GPUCacheContext(*args, **kwargs)
