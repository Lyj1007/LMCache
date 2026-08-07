# SPDX-License-Identifier: Apache-2.0
"""Kunlun KPU KV-cache IPC wrapper.

Unlike CUDA, Kunlun exposes device memory through a **flat address space that
is directly addressable across processes**: a device pointer obtained in the
worker can be dereferenced by the LMCache server process without any
``cudaIpcGetMemHandle`` / ``cudaIpcOpenMemHandle`` round-trip.  The wrapper
therefore ships the raw pointer plus the tensor geometry, and the server
rebuilds a **zero-copy view over the worker's own KV cache**.

That aliasing is the whole point: ``retrieve`` has the server write straight
into worker memory.  A wrapper that silently produced a *copy* would make
stores look fine while retrieves quietly dropped every token, so
:meth:`KpuPtrIPCWrapper.to_tensor` fails closed if the reconstructed tensor
does not alias the original pointer.

Lifetime: the wrapper is a non-owning alias.  The worker must keep the
underlying KV-cache tensor alive for as long as the server may touch it,
which holds for vLLM/SGLang KV caches (allocated once, freed at shutdown).
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

#: Prefix for the synthetic device identity used when ``torch_xmlir`` does not
#: expose a ``uuid`` on its device properties (see :meth:`_get_device_uuid`).
_ORDINAL_UUID_PREFIX = "kpu-ordinal:"


class KpuPtrIPCWrapper(DeviceIPCWrapper):
    """Zero-copy device-pointer wrapper for Kunlun KPU KV caches.

    Bound to ``device_type="kpu"`` via
    :attr:`~lmcache.v1.platform.kpu.KpuDeviceSpec.ipc_wrapper_cls`, so
    :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory` dispatches to it
    once the reported ``"cuda"`` device type has been normalized to ``"kpu"``
    by :func:`~lmcache.v1.platform._device_detect.normalize_device_type`.
    """

    #: LMCache logical device type this wrapper handles. Note this is ``"kpu"``
    #: even though the tensors themselves report ``device.type == "cuda"``
    #: (xmlir drives Kunlun through ``torch.cuda``).
    device_type: ClassVar[str] = "kpu"

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "KpuPtrIPCWrapper":
        """Factory used by
        :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Args:
            tensor: A Kunlun KPU KV-cache tensor.

        Returns:
            A new :class:`KpuPtrIPCWrapper` aliasing ``tensor``.
        """
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.gpu_connector.kv_format.contiguity import (
            attempt_permute_to_contiguous_view,
        )

        # Same normalization as the CUDA wrapper: turn a permuted view (e.g.
        # vLLM's NHD-over-HND) back into its physical layout so the geometry
        # we ship matches memory. Unlike the CUDA wrapper we cannot fall back
        # to ``set_(storage, ...)`` on the far side -- rebuilding from a raw
        # pointer only yields a contiguous view -- so a still-discontiguous
        # tensor is rejected rather than silently mis-read.
        tensor = attempt_permute_to_contiguous_view(tensor)
        if not tensor.is_contiguous():
            raise ValueError(
                "KPU pointer transfer requires a contiguous KV-cache tensor; "
                f"got shape={tuple(tensor.shape)} stride={tuple(tensor.stride())}. "
                "Use MP transfer mode 'engine_driven' for this layout."
            )

        data_ptr = int(tensor.data_ptr())
        if data_ptr == 0:
            raise ValueError(
                "KPU pointer transfer got a null data_ptr; the KV cache is "
                "not backed by device memory."
            )

        self.data_ptr = data_ptr
        self.nbytes = tensor.numel() * tensor.element_size()

        # DeviceIPCWrapper interface fields. ``handle`` is unused -- Kunlun
        # needs no IPC handle at all -- but kept (None) so the base-class
        # equality check has a value to compare.
        self.handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        if device_index is None:
            device_index = torch_dev.current_device()
        self.device_uuid = self._get_device_uuid(int(device_index))

    # ------------------------------------------------------------------
    # Device identity
    #
    # ``torch_xmlir`` does not reliably expose ``uuid`` on device properties,
    # so the inherited UUID-based discovery would raise on Kunlun. Fall back
    # to the device ordinal, which is a sound identity here: LMCache pins one
    # worker per device and the server shares the same visible-device set.
    # ------------------------------------------------------------------

    @classmethod
    def _get_device_uuid(cls, device_index: int) -> str:
        """Return a stable device identity, falling back to the ordinal."""
        try:
            props = torch_dev.get_device_properties(device_index)
            uuid = getattr(props, "uuid", None)
            if uuid is not None:
                return str(uuid)
        except Exception:  # pragma: no cover - depends on xmlir build
            pass
        return f"{_ORDINAL_UUID_PREFIX}{device_index}"

    @classmethod
    def _get_device_index_from_uuid(cls, device_uuid: str) -> int:
        """Resolve the ordinal, honouring the synthetic ordinal identity."""
        if device_uuid.startswith(_ORDINAL_UUID_PREFIX):
            return int(device_uuid[len(_ORDINAL_UUID_PREFIX) :])
        return super()._get_device_index_from_uuid(device_uuid)

    def to_tensor(self) -> torch.Tensor:
        """Rebuild a zero-copy view over the worker's KV cache.

        Returns:
            A tensor aliasing the worker-side allocation.

        Raises:
            RuntimeError: If the reconstruction did not alias the original
                pointer (i.e. it silently produced a copy).
        """
        # First Party
        from lmcache.v1.platform.torch_ops import _tensor_from_ptr

        device_index = self._get_device_index_from_uuid(self.device_uuid)

        # xmlir presents Kunlun as ``torch.cuda``, so the shared CUDA pointer
        # path (``__cuda_array_interface__``) is the one that applies here.
        tensor = _tensor_from_ptr(
            self.data_ptr,
            self.shape,
            self.dtype,
            device=f"cuda:{device_index}",
        )

        # Fail closed. ``_tensor_from_ptr`` falls back to a device-to-device
        # ``cudaMemcpy`` when the zero-copy path raises; that fallback returns
        # a *copy*, which would silently break retrieve (the server would
        # write into scratch memory instead of the worker's KV cache).
        if int(tensor.data_ptr()) != self.data_ptr:
            raise RuntimeError(
                "KPU pointer transfer could not alias worker memory "
                f"(requested 0x{self.data_ptr:x}, got 0x{tensor.data_ptr():x}). "
                "Refusing to continue with a copy, which would drop KV writes. "
                "Use MP transfer mode 'engine_driven' instead."
            )
        return tensor
