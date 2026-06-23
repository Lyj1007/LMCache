# SPDX-License-Identifier: Apache-2.0
"""XPU device-pointer KV cache transfer module (XPU offload v2 §9).

See ``docs/design/v1/multiprocess/xpu_offload_v2_design.md`` for the
end-to-end design rationale. The server side here:

1. Wraps each worker's KV-cache layer ``data_ptr()`` as a remote tensor
   via the ``__cuda_array_interface__`` pattern. XPU exposes a flat
   physical device address space, so peer-process ``data_ptr()`` is
   directly usable in transfer kernels (see §3.8 / §6).
2. Hosts the L1 SHM pool already registered for device DMA in 4 GiB
   segments by ``xpu_cuda_compat`` (§10), used as the destination for
   D2H stores and the source for H2D retrieves.
3. Drives store/retrieve handlers synchronously per request: the worker
   forward thread is asynchronous with respect to the server (§3.1 /
   §8.1) — the worker BG thread synchronizes on its own
   ``compute_stream`` event before sending the MQ request, so by the
   time the handler runs every cross-process read is safe to issue.
"""

# Standard
from dataclasses import dataclass, field
from typing import Optional
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheEngineKey,
    RegisterXpuContextPayload,
    XpuGroupView,
    XpuLayerHandle,
)
from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext, ShmPoolInfo
from lmcache.v1.multiprocess.engine_module import HandlerSpec, ThreadPoolType
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.protocols.engine import RegisterXpuContextResponse
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


# Map torch dtype name -> __cuda_array_interface__ typestr. bfloat16/fp8
# have no native NumPy equivalent, so they ride along as int16/uint8 and
# get re-viewed once the wrapper is converted to a torch tensor.
_DTYPE_TO_TYPESTR: dict[str, str] = {
    "float16": "<f2",
    "bfloat16": "<i2",
    "float32": "<f4",
    "float64": "<f8",
    "int8": "<i1",
    "uint8": "<u1",
    "int16": "<i2",
    "int32": "<i4",
    "int64": "<i8",
    "float8_e4m3fn": "<u1",
    "float8_e5m2": "<u1",
}

# 4 GiB segment size used for L1 SHM host-registration (XPU offload v2 §10).
# xpu_host_register requires segments of at most 4 GiB.
_SHM_REGISTRATION_SEGMENT_BYTES: int = 4 * 1024 * 1024 * 1024


class _CudaArrayWrapper:
    """Wraps a raw device pointer in the ``__cuda_array_interface__`` shape.

    This mirrors the helper in ``lmcache.python_ops_fallback`` and is the
    canonical way to bring a peer-process ``data_ptr()`` into the current
    process as a ``torch.Tensor`` view (XPU exposes a flat physical address
    space, see XPU offload v2 §6).

    Args:
        ptr_int: Device pointer (peer process address space).
        shape_tuple: Tensor shape as a tuple.
        type_str: ``__cuda_array_interface__`` typestr (e.g. ``"<f2"``).
    """

    def __init__(
        self, ptr_int: int, shape_tuple: tuple, type_str: str
    ) -> None:
        self.__cuda_array_interface__ = {
            "data": (ptr_int, False),
            "shape": shape_tuple,
            "typestr": type_str,
            "version": 3,
        }


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Resolve a torch dtype name to the corresponding ``torch.dtype``.

    Args:
        dtype_str: torch dtype attribute name (e.g. ``"bfloat16"``).

    Returns:
        The matching ``torch.dtype``.

    Raises:
        ValueError: If ``dtype_str`` is not a valid torch dtype attribute.
    """
    dtype = getattr(torch, dtype_str, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"Invalid dtype_str '{dtype_str}': must be a valid torch dtype "
            "attribute name (e.g. 'bfloat16' for torch.bfloat16)."
        )
    return dtype


def _wrap_xpu_layer_tensor(
    handle: XpuLayerHandle, device: torch.device
) -> torch.Tensor:
    """Wrap a peer-process XPU device pointer as a torch tensor view.

    Args:
        handle: Cross-process pointer descriptor for one KV-cache layer.
        device: Local torch device on which the tensor view should live.

    Returns:
        A ``torch.Tensor`` view over the peer's layer storage.

    Raises:
        ValueError: If ``handle.dtype_str`` is not a known dtype name.
    """
    typestr = _DTYPE_TO_TYPESTR.get(handle.dtype_str)
    if typestr is None:
        raise ValueError(
            f"Unsupported dtype_str '{handle.dtype_str}' for XPU layer "
            f"'{handle.layer_name}'"
        )
    dtype = _resolve_dtype(handle.dtype_str)
    shape = tuple(handle.shape)
    numel = 1
    for s in shape:
        numel *= int(s)

    wrapper = _CudaArrayWrapper(int(handle.data_ptr), (numel,), typestr)
    flat = torch.as_tensor(wrapper, device=device)
    if dtype is torch.bfloat16:
        # bf16 was smuggled as int16 (<i2). Re-view at the right dtype.
        flat = flat.view(torch.bfloat16)
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        flat = flat.view(dtype)
    return flat.view(*shape)


def _layout_desc_from_groups(
    groups: list[XpuGroupView], layer_handles: list[XpuLayerHandle]
) -> MemoryLayoutDesc:
    """Build a per-group ``MemoryLayoutDesc`` describing the chunk shape.

    Each group's MemoryObj slot must hold the full multi-layer gathered
    data: ``sum(page_bytes_per_layer) * blocks_per_chunk`` bytes. We use
    int8 dtype with a 1-D shape to represent raw byte storage.

    Args:
        groups: Per-group metadata reported by the worker.
        layer_handles: Layer handles, used to compute per-block byte width.

    Returns:
        A :class:`MemoryLayoutDesc` covering all groups in ``groups`` order.
    """
    # Compute per-group total page bytes (sum across layers).
    group_page_bytes: dict[int, int] = {}
    for h in layer_handles:
        shape = tuple(h.shape)
        num_blocks = shape[0]
        numel = 1
        for s in shape:
            numel *= int(s)
        elem_size = torch.tensor([], dtype=_resolve_dtype(h.dtype_str)).element_size()
        page_bytes = (numel // num_blocks) * elem_size
        group_page_bytes[h.group_id] = group_page_bytes.get(h.group_id, 0) + page_bytes

    shapes: list[torch.Size] = []
    dtypes: list[torch.dtype] = []
    for g in groups:
        total_bytes = group_page_bytes.get(g.group_id, 0) * g.blocks_per_chunk
        shapes.append(torch.Size([total_bytes]))
        dtypes.append(torch.int8)
    return MemoryLayoutDesc(shapes=shapes, dtypes=dtypes)


def _load_xvllm_ops():
    """Load xvllm gather/scatter ops with graceful fallback.

    Returns:
        Tuple of (gather_fn, scatter_fn) or (None, None) if unavailable.
    """
    try:
        import vllm_xpu._C  # noqa: F401
    except ImportError:
        logger.info("vllm_xpu._C not importable; gather/scatter disabled.")
        return None, None
    g = getattr(torch.ops._C, "gather_multi_layer_block_kv_transfer", None)
    s = getattr(torch.ops._C, "scatter_multi_layer_block_kv_transfer", None)
    if g is None or s is None:
        logger.warning("gather/scatter ops not found in torch.ops._C")
        return None, None
    return g, s


_gather_op, _scatter_op = _load_xvllm_ops()


@dataclass
class XpuInstanceEntry:
    """Registered XPU-worker context held by the server.

    Attributes:
        remote_layer_tensors: Per-layer torch tensor views over the worker's
            KV cache storage, one entry per :class:`XpuLayerHandle` in
            registration order.
        groups: Per-group metadata as reported by the worker.
        layer_handles: Original handles preserved for lookups.
        model_name: Model name used as part of the cache key.
        world_size: World size used as part of the cache key.
        tp_rank: Worker's TP rank within the inference engine TP group.
        tp_size: TP size of the inference engine TP group.
        gpu_kv_format: ``GPUKVFormat`` enum value (int) describing the
            physical KV layout on the worker side.
        broadcast_buffer_bytes: Worker-side broadcast buffer size negotiated
            during registration.
        device: Local torch device on which the remote layer views live.
        local_device_buffer: Optional staging device buffer used for D2D
            gather/scatter when needed.
        store_lock: Serializes store-stream submissions for this instance.
        retrieve_lock: Serializes retrieve-stream submissions for this
            instance.
        group_layer_tensors_int8: Per-group int8-viewed remote tensors.
        layers_scalars_tensors: Per-group int32 device tensor of page sizes.
        paged_buffer_ptrs_devs: Per-group int64 device tensor of data ptrs.
        layer_page_sizes_per_group: Per-group list of page sizes in bytes.
        max_page_size_per_group: Per-group max page size.
        store_staging_buffer: Device int8 buffer for gather D2H staging.
        retrieve_staging_buffer: Device int8 buffer for scatter H2D staging.
        compact_buf: Pre-allocated device buffer for slow-path store compaction.
    """

    remote_layer_tensors: list[torch.Tensor]
    groups: list[XpuGroupView]
    layer_handles: list[XpuLayerHandle]
    model_name: str
    world_size: int
    tp_rank: int
    tp_size: int
    gpu_kv_format: int
    broadcast_buffer_bytes: int
    device: torch.device
    local_device_buffer: Optional[torch.Tensor] = None
    store_lock: threading.Lock = field(default_factory=threading.Lock)
    retrieve_lock: threading.Lock = field(default_factory=threading.Lock)
    # Kernel cached state (populated at registration for gather/scatter)
    group_layer_tensors_int8: list[list[torch.Tensor]] = field(default_factory=list)
    layers_scalars_tensors: list[torch.Tensor] = field(default_factory=list)
    paged_buffer_ptrs_devs: list[torch.Tensor] = field(default_factory=list)
    layer_page_sizes_per_group: list[list[int]] = field(default_factory=list)
    max_page_size_per_group: list[int] = field(default_factory=list)
    # Per-group pre-computed: total_bytes = sum(page_sizes) * blocks_per_chunk
    total_bytes_per_group: list[int] = field(default_factory=list)
    # Per-group flag: True if all layers have the same page size
    is_uniform_per_group: list[bool] = field(default_factory=list)
    store_staging_buffer: Optional[torch.Tensor] = None
    retrieve_staging_buffer: Optional[torch.Tensor] = None
    # Separate block_ids buffers for store and retrieve to avoid contention.
    store_block_ids_buffer: Optional[torch.Tensor] = None
    retrieve_block_ids_buffer: Optional[torch.Tensor] = None
    # Pre-allocated compact buffer for slow-path store D2H.
    compact_buf: Optional[torch.Tensor] = None


class XpuTransferModule:
    """Server-side XPU device-pointer KV cache transfer module.

    Implements the four XPU offload v2 handlers (REGISTER_XPU_KV_CACHE,
    UNREGISTER_XPU_KV_CACHE, STORE_XPU, RETRIEVE_XPU) per
    ``docs/design/v1/multiprocess/xpu_offload_v2_design.md`` §9.

    The server uses the ``__cuda_array_interface__`` pattern to wrap each
    worker layer's ``data_ptr()`` as a remote tensor — XPU exposes a flat
    physical address space, so peer-process pointers are directly usable
    in transfer kernels (§3.8 / §6).

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheEngineContext) -> None:
        self._ctx = ctx
        self._instances: dict[int, XpuInstanceEntry] = {}
        self._registered_shm_segments: list[tuple[int, int]] = []
        self._shm_pool_info: ShmPoolInfo = self._ctx.shm_pool_info
        self._shm_registered = False
        self._lock = threading.Lock()

    @property
    def context(self) -> MPCacheEngineContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    @property
    def instances(self) -> dict[int, XpuInstanceEntry]:
        """Per-instance XPU context registry."""
        return self._instances

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for the XPU transfer request types.

        Returns:
            A list of HandlerSpec entries for REGISTER_XPU_KV_CACHE,
            UNREGISTER_XPU_KV_CACHE, STORE_XPU, RETRIEVE_XPU.
        """
        return [
            HandlerSpec(
                RequestType.REGISTER_XPU_KV_CACHE,
                self.register_xpu_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.UNREGISTER_XPU_KV_CACHE,
                self.unregister_xpu_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.STORE_XPU,
                self.store_xpu,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.RETRIEVE_XPU,
                self.retrieve_xpu,
                ThreadPoolType.AFFINITY,
            ),
        ]

    def report_status(self) -> dict:
        """Return XPU transfer module status information.

        Returns:
            Dict with registered instance ids and per-instance metadata.
        """
        registered_xpu_ids: list[int] = []
        xpu_context_meta: dict[str, dict] = {}
        for instance_id, entry in self._instances.items():
            registered_xpu_ids.append(instance_id)
            xpu_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "tp_rank": entry.tp_rank,
                "tp_size": entry.tp_size,
                "num_layers": len(entry.layer_handles),
                "groups": [
                    {
                        "group_id": g.group_id,
                        "block_size": g.block_size,
                        "blocks_per_chunk": g.blocks_per_chunk,
                        "is_mla": g.is_mla,
                    }
                    for g in entry.groups
                ],
                "broadcast_buffer_bytes": entry.broadcast_buffer_bytes,
            }
        return {
            "registered_xpu_ids": registered_xpu_ids,
            "xpu_context_meta": xpu_context_meta,
        }

    def close(self) -> None:
        """Release XPU resources and unregister any pinned SHM segments.

        Best-effort: if ``_unregister_shm_segments_locked`` raises after
        partial cleanup, the exception is logged and re-raised so the
        caller sees the failure, but the instance map is already
        cleared so a retry can proceed without stale state.
        """
        with self._lock:
            self._instances.clear()
            try:
                self._unregister_shm_segments_locked()
            except RuntimeError:
                logger.exception("XPU SHM cleanup partially failed in close()")
                raise

    def _ensure_shm_registered(self) -> None:
        """Host-register the L1 SHM pool in 4 GiB segments (idempotent).

        See XPU offload v2 §10 — the L1 SHM pool backing D2H/H2D must be
        host-registered for device DMA before any transfer.
        """
        if self._shm_registered:
            return
        shm_name = self._shm_pool_info["shm_name"]
        pool_size = self._shm_pool_info["pool_size"]
        if not shm_name or pool_size <= 0:
            self._shm_registered = True
            return

        # First Party
        from lmcache.xpu_cuda_compat.mem_alloc import host_register

        # The L1 pool is opened by ``StorageManager`` via POSIX SHM. We
        # discover its base pointer through the L1 manager's memory
        # allocator, then host-register every 4 GiB segment.
        try:
            l1_manager = self._ctx.storage_manager.l1_manager
            allocator = l1_manager.memory_allocator
            base_ptr = int(allocator.get_base_ptr())
        except AttributeError:
            logger.warning(
                "L1 manager does not expose memory_allocator.get_base_ptr; "
                "skipping XPU SHM host registration"
            )
            self._shm_registered = True
            return

        remaining = pool_size
        cursor = base_ptr
        while remaining > 0:
            segment = min(remaining, _SHM_REGISTRATION_SEGMENT_BYTES)
            host_register(cursor, segment)
            self._registered_shm_segments.append((cursor, segment))
            cursor += segment
            remaining -= segment

        self._shm_registered = True
        logger.info(
            "XPU SHM host-registered: %d bytes across %d segments",
            pool_size,
            len(self._registered_shm_segments),
        )

    def _unregister_shm_segments_locked(self) -> None:
        """Unregister all previously host-registered SHM segments.

        Caller must hold ``self._lock``. All segments are attempted even
        when individual ``host_unregister`` calls fail; failures are
        accumulated and surfaced after the loop completes so partial
        successes still drop the bookkeeping. The internal segment list
        is cleared unconditionally to keep the module's view consistent
        with the kernel state for any segment that did succeed.
        """
        if not self._registered_shm_segments:
            return
        # First Party
        from lmcache.xpu_cuda_compat.mem_alloc import host_unregister

        segments = list(self._registered_shm_segments)
        # Drop bookkeeping eagerly so a re-entry into close() does not
        # try to double-unregister; survivors are re-tracked below.
        self._registered_shm_segments.clear()
        self._shm_registered = False

        failures: list[tuple[int, BaseException]] = []
        leaked: list[tuple[int, int]] = []
        for ptr, size in segments:
            try:
                host_unregister(ptr)
            except Exception as exc:
                logger.exception(
                    "XPU SHM host_unregister failed for 0x%x", ptr
                )
                failures.append((ptr, exc))
                leaked.append((ptr, size))

        # Re-track segments that failed to unregister so a follow-up
        # close() can retry instead of silently leaking them.
        self._registered_shm_segments.extend(leaked)
        if leaked:
            self._shm_registered = True

        if failures:
            first_ptr, first_exc = failures[0]
            raise RuntimeError(
                f"XPU SHM host_unregister failed for {len(failures)} of "
                f"{len(segments)} segments; first failure 0x{first_ptr:x}: "
                f"{first_exc}"
            )

    def _allocate_local_device_buffer(
        self, groups: list[XpuGroupView], layer_handles: list[XpuLayerHandle]
    ) -> Optional[torch.Tensor]:
        """Allocate a per-group staging device buffer for D2D gather/scatter.

        Returns ``None`` when there is no group requiring staging.

        Args:
            groups: Per-group metadata.
            layer_handles: Layer handles (used for dtype lookup).

        Returns:
            A 1D ``torch.Tensor`` sized for the largest per-chunk group,
            or ``None`` when ``groups`` is empty.
        """
        if not groups:
            return None
        # First layer dtype per group is enough; all layers share dtype.
        group_dtype: dict[int, torch.dtype] = {}
        for h in layer_handles:
            if h.group_id not in group_dtype:
                group_dtype[h.group_id] = _resolve_dtype(h.dtype_str)

        # Use the largest group * num_layers slot count as a safe upper bound.
        max_slots = 0
        slot_dtype = torch.float16
        for g in groups:
            slots = g.blocks_per_chunk * g.block_size
            if slots > max_slots:
                max_slots = slots
                slot_dtype = group_dtype.get(g.group_id, torch.float16)
        if max_slots <= 0:
            return None
        device = torch.device(
            f"{torch_device_type}:{torch_dev.current_device()}"
        )
        return torch.empty(max_slots, dtype=slot_dtype, device=device)

    def register_xpu_kv_cache(
        self, payload: RegisterXpuContextPayload
    ) -> RegisterXpuContextResponse:
        """Register a worker's XPU KV cache and return SHM pool handles.

        Args:
            payload: Registration payload (XPU offload v2 §7.1).

        Returns:
            ``RegisterXpuContextResponse`` carrying the L1 SHM pool name,
            its size, and the broadcast buffer size for the worker.
        """
        with self._lock:
            self._ensure_shm_registered()
            shm_name = self._shm_pool_info["shm_name"]
            pool_size = self._shm_pool_info["pool_size"]

            if payload.instance_id in self._instances:
                logger.warning(
                    "XPU instance %s already registered; releasing the "
                    "previous entry before re-registering",
                    payload.instance_id,
                )
                # Drop the old entry's resources (remote tensor views,
                # local device buffer, locks) before overwriting; without
                # this the new entry shadows the buffer and the old one
                # leaks until the server itself shuts down.
                stale = self._instances.pop(payload.instance_id)
                self._ctx.layout_desc_registry.unregister(
                    stale.model_name, stale.world_size
                )
                stale.remote_layer_tensors.clear()
                stale.layer_handles.clear()
                stale.groups.clear()
                stale.local_device_buffer = None

            device_index = torch_dev.current_device()
            device = torch.device(f"{torch_device_type}:{device_index}")

            remote_layer_tensors: list[torch.Tensor] = []
            for handle in payload.layer_handles:
                remote_layer_tensors.append(_wrap_xpu_layer_tensor(handle, device))

            broadcast_buffer_bytes = self._compute_broadcast_buffer_bytes(
                payload.groups, payload.layer_handles
            )

            local_device_buffer = self._allocate_local_device_buffer(
                payload.groups, payload.layer_handles
            )

            # Build per-group kernel metadata for gather/scatter
            (
                group_layer_tensors_int8,
                layers_scalars_tensors,
                paged_buffer_ptrs_devs,
                layer_page_sizes_per_group,
                max_page_size_per_group,
                total_bytes_per_group,
                is_uniform_per_group,
                store_staging_buffer,
                retrieve_staging_buffer,
                store_block_ids_buffer,
                retrieve_block_ids_buffer,
                compact_buf,
            ) = self._build_kernel_metadata(
                remote_layer_tensors, payload.groups, payload.layer_handles, device
            )

            entry = XpuInstanceEntry(
                remote_layer_tensors=remote_layer_tensors,
                groups=list(payload.groups),
                layer_handles=list(payload.layer_handles),
                model_name=payload.model_name,
                world_size=payload.world_size,
                tp_rank=payload.tp_rank,
                tp_size=payload.tp_size,
                gpu_kv_format=payload.gpu_kv_format,
                broadcast_buffer_bytes=broadcast_buffer_bytes,
                device=device,
                local_device_buffer=local_device_buffer,
                group_layer_tensors_int8=group_layer_tensors_int8,
                layers_scalars_tensors=layers_scalars_tensors,
                paged_buffer_ptrs_devs=paged_buffer_ptrs_devs,
                layer_page_sizes_per_group=layer_page_sizes_per_group,
                max_page_size_per_group=max_page_size_per_group,
                total_bytes_per_group=total_bytes_per_group,
                is_uniform_per_group=is_uniform_per_group,
                store_staging_buffer=store_staging_buffer,
                retrieve_staging_buffer=retrieve_staging_buffer,
                store_block_ids_buffer=store_block_ids_buffer,
                retrieve_block_ids_buffer=retrieve_block_ids_buffer,
                compact_buf=compact_buf,
            )
            self._instances[payload.instance_id] = entry

            layout_desc = _layout_desc_from_groups(
                payload.groups, payload.layer_handles
            )
            self._ctx.layout_desc_registry.register(
                payload.model_name, payload.world_size, layout_desc
            )

            logger.info(
                "Registered XPU KV cache instance=%d model=%s world_size=%d "
                "tp=%d/%d num_layers=%d num_groups=%d broadcast_bytes=%d",
                payload.instance_id,
                payload.model_name,
                payload.world_size,
                payload.tp_rank,
                payload.tp_size,
                len(payload.layer_handles),
                len(payload.groups),
                broadcast_buffer_bytes,
            )

            return RegisterXpuContextResponse(
                l1_shm_name=shm_name,
                l1_shm_size=pool_size,
                broadcast_buffer_bytes=broadcast_buffer_bytes,
            )

    @staticmethod
    def _compute_broadcast_buffer_bytes(
        groups: list[XpuGroupView], layer_handles: list[XpuLayerHandle]
    ) -> int:
        """Compute the worker-side broadcast buffer size for MLA TP groups.

        Returns 0 when no group is MLA (XPU offload v2 §9.4). Sized to
        hold ``num_layers * blocks_per_chunk * per_block_bytes`` for the
        largest MLA group, so the worker can XCCL-broadcast a full
        single-chunk per-layer payload through the buffer without
        re-allocation. ``per_block_bytes`` is derived from the layer's
        actual paged-block row width (``shape[1:]`` element product times
        the dtype element size); using a placeholder hidden-dim would
        under-size the buffer for any model whose paged tensors carry a
        non-unit per-block row stride (e.g. DSV4-Flash MLA layers).

        Args:
            groups: Per-group metadata.
            layer_handles: Layer handles (used for per-block byte count
                and per-group layer count).

        Returns:
            Broadcast buffer size in bytes.
        """
        mla_groups = [g for g in groups if g.is_mla]
        if not mla_groups:
            return 0

        # Per-group: dtype size, per-block row bytes (max across that
        # group's layers — different layers in the same group may share a
        # tensor with the largest row width), and layer count.
        group_dtype_size: dict[int, int] = {}
        group_per_block_bytes: dict[int, int] = {}
        layers_per_group: dict[int, int] = {}
        for h in layer_handles:
            if h.group_id not in group_dtype_size:
                dtype = _resolve_dtype(h.dtype_str)
                group_dtype_size[h.group_id] = torch.empty(
                    (), dtype=dtype
                ).element_size()
            row_elems = 1
            for dim in tuple(h.shape)[1:]:
                row_elems *= int(dim)
            row_bytes = row_elems * group_dtype_size[h.group_id]
            prev = group_per_block_bytes.get(h.group_id, 0)
            if row_bytes > prev:
                group_per_block_bytes[h.group_id] = row_bytes
            layers_per_group[h.group_id] = (
                layers_per_group.get(h.group_id, 0) + 1
            )

        # Largest single-chunk per-group payload (bytes) across MLA
        # groups: one MLA broadcast carries one group's worth of paged
        # blocks, replicated across all the group's layers.
        max_bytes = 0
        for g in mla_groups:
            per_block = group_per_block_bytes.get(g.group_id, 0)
            num_layers = layers_per_group.get(g.group_id, 0)
            payload_bytes = g.blocks_per_chunk * num_layers * per_block
            if payload_bytes > max_bytes:
                max_bytes = payload_bytes
        return max_bytes

    @staticmethod
    def _build_kernel_metadata(
        remote_layer_tensors: list[torch.Tensor],
        groups: list[XpuGroupView],
        layer_handles: list[XpuLayerHandle],
        device: torch.device,
    ) -> tuple[
        list[list[torch.Tensor]],
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[int]],
        list[int],
        list[int],
        list[bool],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Build per-group kernel metadata for gather/scatter ops.

        Returns:
            Tuple of (group_layer_tensors_int8, layers_scalars_tensors,
            paged_buffer_ptrs_devs, layer_page_sizes_per_group,
            max_page_size_per_group, total_bytes_per_group,
            is_uniform_per_group, store_staging_buffer,
            retrieve_staging_buffer, store_block_ids_buffer,
            retrieve_block_ids_buffer, compact_buf).
        """
        if _gather_op is None and _scatter_op is None:
            return [], [], [], [], [], [], [], None, None, None, None, None

        # Group layers by group_id
        per_group_layers: dict[int, list[torch.Tensor]] = {}
        group_order: list[int] = []
        for g in groups:
            per_group_layers[g.group_id] = []
            group_order.append(g.group_id)
        for idx, handle in enumerate(layer_handles):
            per_group_layers[handle.group_id].append(remote_layer_tensors[idx])

        group_layer_tensors_int8: list[list[torch.Tensor]] = []
        layers_scalars_tensors: list[torch.Tensor] = []
        paged_buffer_ptrs_devs: list[torch.Tensor] = []
        layer_page_sizes_per_group: list[list[int]] = []
        max_page_size_per_group: list[int] = []

        max_staging_bytes = 0
        for gid in group_order:
            tensors = per_group_layers[gid]
            # int8 view each tensor as [num_blocks, page_bytes]
            int8_tensors: list[torch.Tensor] = []
            page_sizes: list[int] = []
            for t in tensors:
                flat = t.view(torch.int8)
                num_blocks = int(t.shape[0])
                page_bytes = flat.numel() // num_blocks
                int8_t = flat.view(num_blocks, page_bytes)
                int8_tensors.append(int8_t)
                page_sizes.append(page_bytes)
            group_layer_tensors_int8.append(int8_tensors)
            layer_page_sizes_per_group.append(page_sizes)
            max_page = max(page_sizes) if page_sizes else 0
            max_page_size_per_group.append(max_page)

            nl = len(int8_tensors)
            # Scalars tensor (page sizes per layer)
            scalars = torch.tensor(page_sizes, dtype=torch.int32, device=device)
            layers_scalars_tensors.append(scalars)
            # Pre-staged device pointer table
            ptrs = torch.tensor(
                [int(t.data_ptr()) for t in int8_tensors],
                dtype=torch.int64, device=device,
            )
            paged_buffer_ptrs_devs.append(ptrs)

            # Track max staging needed
            bpc = next(
                (g.blocks_per_chunk for g in groups if g.group_id == gid), 0
            )
            needed = nl * bpc * max_page
            if needed > max_staging_bytes:
                max_staging_bytes = needed

        # Pre-compute per-group total_bytes and uniformity flag.
        total_bytes_per_group: list[int] = []
        is_uniform_per_group: list[bool] = []
        for gi, gid in enumerate(group_order):
            page_sizes = layer_page_sizes_per_group[gi]
            max_page = max_page_size_per_group[gi]
            bpc = next(
                (g.blocks_per_chunk for g in groups if g.group_id == gid), 0
            )
            total_bytes_per_group.append(sum(ps * bpc for ps in page_sizes))
            is_uniform_per_group.append(all(ps == max_page for ps in page_sizes))

        store_staging_buffer: Optional[torch.Tensor] = None
        retrieve_staging_buffer: Optional[torch.Tensor] = None
        compact_buf: Optional[torch.Tensor] = None
        if max_staging_bytes > 0:
            store_staging_buffer = torch.empty(
                max_staging_bytes, dtype=torch.int8, device=device
            )
            retrieve_staging_buffer = torch.empty(
                max_staging_bytes, dtype=torch.int8, device=device
            )
            max_total_bytes = max(total_bytes_per_group) if total_bytes_per_group else 0
            if max_total_bytes > 0:
                compact_buf = torch.empty(
                    max_total_bytes, dtype=torch.int8, device=device
                )

        # Pre-allocate separate block_ids buffers for store and retrieve.
        max_bpc = max(
            (g.blocks_per_chunk for g in groups if g.group_id in per_group_layers),
            default=0,
        )
        store_block_ids_buffer: Optional[torch.Tensor] = None
        retrieve_block_ids_buffer: Optional[torch.Tensor] = None
        if max_bpc > 0:
            store_block_ids_buffer = torch.empty(
                max_bpc, dtype=torch.int64, device=device
            )
            retrieve_block_ids_buffer = torch.empty(
                max_bpc, dtype=torch.int64, device=device
            )

        return (
            group_layer_tensors_int8,
            layers_scalars_tensors,
            paged_buffer_ptrs_devs,
            layer_page_sizes_per_group,
            max_page_size_per_group,
            total_bytes_per_group,
            is_uniform_per_group,
            store_staging_buffer,
            retrieve_staging_buffer,
            store_block_ids_buffer,
            retrieve_block_ids_buffer,
            compact_buf,
        )

    def unregister_xpu_kv_cache(self, instance_id: int) -> None:
        """Unregister an XPU instance and release its remote-tensor views.

        Args:
            instance_id: Worker process instance id.
        """
        with self._lock:
            entry = self._instances.pop(instance_id, None)
            if entry is None:
                logger.warning(
                    "No XPU instance registered for id=%d", instance_id
                )
                return
            self._ctx.layout_desc_registry.unregister(
                entry.model_name, entry.world_size
            )
            logger.info("Unregistered XPU KV cache instance=%d", instance_id)

    @_lmcache_nvtx_annotate
    def store_xpu(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        block_ids: list[list[int]],
    ) -> bool:
        """Store XPU KV cache chunks D2H into the L1 SHM pool.

        Args:
            key: Cache key for the chunk range.
            instance_id: Worker process instance id.
            block_ids: Per-group block ids.

        Returns:
            ``True`` if the store committed without a fatal error,
            ``False`` otherwise. A protocol-level block-id underflow
            yields ``False`` (the whole store is skipped).
        """
        st = time.perf_counter()
        entry = self._instances.get(instance_id)
        if entry is None:
            raise ValueError(f"No XPU instance registered for id={instance_id}")

        obj_keys = self._ctx.resolve_obj_keys(key)
        if not obj_keys:
            return True

        groups = entry.groups
        num_groups = len(groups)
        blocks_per_chunk_per_group = [g.blocks_per_chunk for g in groups]

        if len(block_ids) < num_groups or any(
            len(block_ids[gi]) < len(obj_keys) * blocks_per_chunk_per_group[gi]
            for gi in range(num_groups)
        ):
            logger.warning(
                "STORE_XPU block ID underflow request_id=%s; skipping store",
                key.request_id,
            )
            return False

        reserved_dict: dict[ObjectKey, MemoryObj] = {}
        store_succeeded = False
        try:
            with entry.store_lock:
                layout_desc = _layout_desc_from_groups(
                    entry.groups, entry.layer_handles
                )
                reserved_dict = self._ctx.storage_manager.reserve_write(
                    obj_keys, layout_desc, "new"
                )

                for chunk_idx, obj_key in enumerate(obj_keys):
                    memory_obj = reserved_dict.get(obj_key)
                    if memory_obj is None:
                        continue
                    self._copy_chunk_to_memory_obj(
                        entry, chunk_idx, block_ids, memory_obj
                    )
                # Sync after D2H copies to ensure data integrity.
                event = torch_dev.Event()
                event.record(torch_dev.current_stream(entry.device))
                event.synchronize()
                store_succeeded = True
        except Exception:
            logger.exception("STORE_XPU failed for request_id=%s", key.request_id)
            return False
        finally:
            stored_count = len(reserved_dict) if store_succeeded else 0
            if stored_count:
                self._ctx.storage_manager.finish_write(list(reserved_dict.keys()))

        ed = time.perf_counter()
        if reserved_dict:
            logger.info(
                "STORE_XPU stored %d tokens in %.3f s",
                len(reserved_dict) * self._ctx.chunk_size,
                ed - st,
            )
        return True

    def _copy_chunk_to_memory_obj(
        self,
        entry: XpuInstanceEntry,
        chunk_idx: int,
        block_ids: list[list[int]],
        memory_obj: MemoryObj,
    ) -> None:
        """D2H copy one chunk's KV data from peer device pointers into SHM.

        Args:
            entry: Registered XPU instance metadata.
            chunk_idx: Index into ``obj_keys`` (chunk position).
            block_ids: Per-group paged block ids (full request).
            memory_obj: Destination L1 SHM-backed memory slot.
        """
        if _gather_op is None or entry.store_staging_buffer is None:
            return
        num_groups = len(entry.groups)
        for gi in range(num_groups):
            bpc = entry.groups[gi].blocks_per_chunk
            chunk_start = chunk_idx * bpc
            chunk_end = chunk_start + bpc
            group_block_ids = block_ids[gi][chunk_start:chunk_end]
            nl = len(entry.group_layer_tensors_int8[gi])
            max_page = entry.max_page_size_per_group[gi]
            page_sizes = entry.layer_page_sizes_per_group[gi]
            n_blocks = len(group_block_ids)

            # Reuse pre-allocated store block_ids buffer
            if entry.store_block_ids_buffer is not None and n_blocks <= entry.store_block_ids_buffer.numel():
                block_ids_dev = entry.store_block_ids_buffer[:n_blocks]
                block_ids_dev.copy_(torch.tensor(group_block_ids, dtype=torch.int64))
            else:
                block_ids_dev = torch.tensor(
                    group_block_ids, dtype=torch.int64, device=entry.device
                )
            staging_view = entry.store_staging_buffer[:nl * n_blocks * max_page].view(
                nl, n_blocks, max_page
            )
            _gather_op(
                entry.group_layer_tensors_int8[gi],
                staging_view,
                block_ids_dev,
                page_sizes,
                max_page,
                entry.layers_scalars_tensors[gi],
                paged_buffer_ptrs_dev=entry.paged_buffer_ptrs_devs[gi],
            )

            # Copy staging → host MemoryObj per-group tensor
            dst_tensor = memory_obj.get_tensor(gi)
            if dst_tensor is not None:
                dst_flat = dst_tensor.view(-1)
                total_bytes = sum(ps * n_blocks for ps in page_sizes)
                # Fast path: uniform page sizes → staging is already
                # contiguous and layout-aligned with host flat tensor.
                if entry.is_uniform_per_group[gi]:
                    src = staging_view[:nl, :n_blocks, :max_page].contiguous().view(-1)[:total_bytes]
                    dst_flat[:total_bytes].copy_(src, non_blocking=True)
                else:
                    # Slow path: gather per-layer slices into pre-allocated compact buf.
                    compact_buf = entry.compact_buf[:total_bytes] if (
                        entry.compact_buf is not None and total_bytes <= entry.compact_buf.numel()
                    ) else torch.empty(total_bytes, dtype=torch.int8, device=entry.device)
                    dev_offset = 0
                    for li in range(nl):
                        ps = page_sizes[li]
                        nbytes = n_blocks * ps
                        compact_buf[dev_offset:dev_offset + nbytes].copy_(
                            staging_view[li, :n_blocks, :ps].contiguous().view(-1)
                        )
                        dev_offset += nbytes
                    dst_flat[:total_bytes].copy_(compact_buf)

    @_lmcache_nvtx_annotate
    def retrieve_xpu(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        block_ids: list[list[int]],
        skip_blocks_per_group: list[int] = [],
    ) -> bool:
        """Retrieve XPU KV cache chunks H2D from the L1 SHM pool.

        Args:
            key: Cache key.
            instance_id: Worker process instance id.
            block_ids: Per-group paged block ids.
            skip_blocks_per_group: Leading blocks to skip per group
                (APC overlap guard).

        Returns:
            ``True`` on success, ``False`` on miss or fatal error.
        """
        skip_blocks_per_group = skip_blocks_per_group or []
        st = time.perf_counter()
        entry = self._instances.get(instance_id)
        if entry is None:
            raise ValueError(f"No XPU instance registered for id={instance_id}")

        obj_keys = self._ctx.resolve_obj_keys(key)
        if not obj_keys:
            return True

        retrieve_succeeded = False
        prefetched_keys: list[ObjectKey] = []
        try:
            with entry.retrieve_lock, self._ctx.storage_manager.read_prefetched_results(
                obj_keys
            ) as memory_objs:
                if not memory_objs or len(memory_objs) != len(obj_keys):
                    logger.warning(
                        "RETRIEVE_XPU miss request_id=%s expected=%d got=%d",
                        key.request_id,
                        len(obj_keys),
                        0 if memory_objs is None else len(memory_objs),
                    )
                    return False
                prefetched_keys = obj_keys[: len(memory_objs)]
                for chunk_idx, memory_obj in enumerate(memory_objs):
                    self._copy_memory_obj_to_chunk(
                        entry,
                        chunk_idx,
                        block_ids,
                        memory_obj,
                        skip_blocks_per_group,
                    )
                retrieve_succeeded = True
        except Exception:
            logger.exception("RETRIEVE_XPU failed for request_id=%s", key.request_id)
            return False
        finally:
            if retrieve_succeeded and prefetched_keys:
                self._ctx.storage_manager.finish_read_prefetched(prefetched_keys)

        ed = time.perf_counter()
        logger.info(
            "RETRIEVE_XPU retrieved %d tokens in %.3f s",
            len(obj_keys) * self._ctx.chunk_size,
            ed - st,
        )
        return True

    def _copy_memory_obj_to_chunk(
        self,
        entry: XpuInstanceEntry,
        chunk_idx: int,
        block_ids: list[list[int]],
        memory_obj: MemoryObj,
        skip_blocks_per_group: list[int],
    ) -> None:
        """H2D copy one chunk's KV data from SHM back to peer device pointers.

        Args:
            entry: Registered XPU instance metadata.
            chunk_idx: Index into ``obj_keys`` (chunk position).
            block_ids: Per-group paged block ids (full request).
            memory_obj: Source L1 SHM-backed memory slot.
            skip_blocks_per_group: APC overlap guard, applied to chunk 0.
        """
        if _scatter_op is None or entry.retrieve_staging_buffer is None:
            return
        num_groups = len(entry.groups)
        for gi in range(num_groups):
            bpc = entry.groups[gi].blocks_per_chunk
            chunk_start = chunk_idx * bpc
            chunk_end = chunk_start + bpc
            group_block_ids = block_ids[gi][chunk_start:chunk_end]
            nl = len(entry.group_layer_tensors_int8[gi])
            max_page = entry.max_page_size_per_group[gi]
            page_sizes = entry.layer_page_sizes_per_group[gi]
            n_blocks = len(group_block_ids)

            # Copy host MemoryObj per-group tensor → staging buffer (H2D)
            src_tensor = memory_obj.get_tensor(gi)
            staging_view = entry.retrieve_staging_buffer[:nl * n_blocks * max_page].view(
                nl, n_blocks, max_page
            )
            if src_tensor is None:
                continue
            src_flat = src_tensor.view(-1)
            total_bytes = sum(ps * n_blocks for ps in page_sizes)
            # Fast path: all layers have the same page size, so host flat
            # layout matches staging layout exactly → single bulk H2D copy.
            if entry.is_uniform_per_group[gi]:
                staging_view[:nl, :n_blocks, :max_page].view(-1)[:total_bytes].copy_(
                    src_flat[:total_bytes], non_blocking=True
                )
            else:
                # Slow path: per-layer copy for heterogeneous page sizes.
                dev_offset = 0
                for li in range(nl):
                    ps = page_sizes[li]
                    nbytes = n_blocks * ps
                    staging_view[li, :n_blocks, :ps].contiguous().view(-1).copy_(
                        src_flat[dev_offset:dev_offset + nbytes], non_blocking=True
                    )
                    dev_offset += nbytes

            # Reuse pre-allocated retrieve block_ids buffer
            if entry.retrieve_block_ids_buffer is not None and n_blocks <= entry.retrieve_block_ids_buffer.numel():
                block_ids_dev = entry.retrieve_block_ids_buffer[:n_blocks]
                block_ids_dev.copy_(torch.tensor(group_block_ids, dtype=torch.int64))
            else:
                block_ids_dev = torch.tensor(
                    group_block_ids, dtype=torch.int64, device=entry.device
                )
            skip_n = 0
            if chunk_idx == 0 and gi < len(skip_blocks_per_group):
                skip_n = skip_blocks_per_group[gi]
            _scatter_op(
                entry.group_layer_tensors_int8[gi],
                staging_view,
                block_ids_dev,
                page_sizes,
                max_page,
                entry.layers_scalars_tensors[gi],
                skip_prefix_n_blocks=skip_n,
                paged_buffer_ptrs_dev=entry.paged_buffer_ptrs_devs[gi],
            )
        # Use stream event instead of full device synchronize to only wait
        # for the scatter (H2D) operations, not all device activity.
        event = torch_dev.Event()
        event.record(torch_dev.current_stream(entry.device))
        event.synchronize()

_ = (lmc_ops, lmcache_memcpy_async_d2h, lmcache_memcpy_async_h2d)
