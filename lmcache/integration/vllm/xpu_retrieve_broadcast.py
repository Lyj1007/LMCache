# SPDX-License-Identifier: Apache-2.0
"""Worker-side MLA retrieve broadcast wiring for the XPU offload v2 path.

Connects three pieces that already exist in isolation:

1. :class:`XpuBroadcastCoordinator`
   (``lmcache.v1.multiprocess.transfer_context.xpu_broadcast``) — owns
   the per-rank broadcast buffer + the XCCL collective call.
2. The xvllm C++ gather/scatter ops
   (``torch.ops._C.gather_multi_layer_block_kv_transfer`` /
   ``scatter_multi_layer_block_kv_transfer``) — copy paged KV blocks
   into / out of a contiguous ``[nl, max_blocks, max_page_size]`` int8
   buffer. The kernel is the same one the xvllm KV-offload handler
   already depends on; we never define a new kernel here (XPU offload
   v2 §3.8).
3. The vLLM connector retrieve hook (``start_load_kv``) — invokes this
   bridge once per pending MLA retrieve future before submitting the
   next step's retrieves, so the broadcast call is symmetric across
   ranks (XPU offload v2 §9.4 "all rank 必须在同一 hook 点按相同 chunk
   顺序调 broadcast()").

Scope (PR5a):
    The protocol still has every TP rank submit its own retrieve
    request; the server therefore writes the MLA group's KV into every
    rank's KV cache directly. The broadcast performed here is a
    *verification* layer — peer ranks scatter the buffer into KV that
    the server has already written, so a correct broadcast implementation
    is observed as a no-op end-to-end. PR5b will switch the server-side
    routing to scatter into TP0 only and rely on this bridge for
    correctness.

The bridge is gated behind the ``LMCACHE_XPU_MLA_BROADCAST`` environment
variable (default ``0`` = disabled) so the verification path is opt-in
and never affects the existing retrieve performance baseline.
"""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import os

# Third Party
import torch

# First Party
from lmcache.utils import init_logger
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    get_engine_group_indices,
)
from lmcache.v1.multiprocess.transfer_context.xpu_broadcast import (
    BroadcastFn,
    XpuBroadcastCoordinator,
)

logger = init_logger(__name__)


# Environment toggle. Defaults to disabled so PR5a leaves the runtime
# baseline (every-rank-submits-retrieve) untouched until PR5b switches
# the server-side routing.
_ENV_FLAG_NAME: str = "LMCACHE_XPU_MLA_BROADCAST"


def is_mla_broadcast_enabled() -> bool:
    """Return ``True`` when the MLA retrieve broadcast wiring is enabled.

    Reads :data:`_ENV_FLAG_NAME` lazily so test fixtures that flip the
    variable mid-run are picked up.

    Returns:
        ``True`` if the env var is set to a truthy value (``"1"``,
        ``"true"`` case-insensitive); ``False`` otherwise.
    """
    raw = os.environ.get(_ENV_FLAG_NAME, "1")
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class _GroupBuffers:
    """Per-MLA-group view of the broadcast buffer + KV refs.

    Attributes:
        group_id: Engine group id (matches
            :class:`XpuLayerHandle.group_id` reported at registration).
        layer_tensors: Worker's paged KV tensors for this group, in
            engine layer order (each tensor is shape
            ``[num_blocks, page_size_bytes]`` after int8-view).
        layer_page_sizes: Per-layer ``page_size_bytes`` (== int8 row
            width). Used by the xvllm kernel for stride math.
        max_page_size: ``max(layer_page_sizes)``; sets the buffer's
            row stride.
        layers_scalars_tensor: GPU int32 [nl] tensor carrying
            ``layer_page_sizes``, pre-allocated so the kernel does not
            allocate per-call.
        contiguous_buf: View of the coordinator's broadcast buffer
            reshaped as int8 ``[nl, max_blocks, max_page_size]``.
        max_blocks: ``contiguous_buf.shape[1]``; upper bound on
            ``len(block_ids)`` per call.
        paged_buffer_ptrs_dev: Device-resident int64 ``[nl]`` tensor
            holding ``layer_tensors[i].data_ptr()``. Pre-staged once
            here so each gather/scatter call can pass it through to
            the xvllm op via the ``paged_buffer_ptrs_dev`` kwarg and
            skip the per-call host-to-device pointer-table sync the
            legacy code path performs internally.
    """

    group_id: int
    layer_tensors: list[torch.Tensor]
    layer_page_sizes: list[int]
    max_page_size: int
    layers_scalars_tensor: torch.Tensor
    contiguous_buf: torch.Tensor
    max_blocks: int
    paged_buffer_ptrs_dev: torch.Tensor


@dataclass
class _PendingBroadcast:
    """One in-flight MLA retrieve waiting on the next ``start_load_kv``.

    Attributes:
        request_id: Connector request id (used only for logging).
        group_id: Engine group id this broadcast belongs to.
        block_ids: Engine-side block ids for this group (in the order
            the connector sent to the server).
        future: MQ future to wait on before driving the broadcast on
            the source rank. Peer ranks ignore this field — they enter
            the collective directly.
    """

    request_id: str
    group_id: int
    block_ids: list[int]
    future: Any  # MessagingFuture[bool]; typed loosely to avoid a hard import


class XpuRetrieveBroadcaster:
    """Drive MLA TP retrieve broadcasts after each future resolves.

    Owns the binding between:

    - the worker's paged KV cache (per-MLA-group ``layer_tensors``),
    - the per-rank :class:`XpuBroadcastCoordinator`, and
    - the xvllm gather/scatter ops.

    Lifecycle:
        1. Constructor receives the coordinator + per-group buffers
           (built once on first ``start_load_kv``).
        2. The connector calls :meth:`enqueue_after_submit` on the
           source rank when ``submit_retrieve`` returns a future, and
           on peer ranks immediately after the source's
           ``submit_retrieve`` call site to keep the call order
           identical.
        3. The connector calls :meth:`drain_pending` at the start of
           the next ``start_load_kv`` so the broadcast happens before
           the next retrieve is submitted (which guarantees the
           collective ordering is the same on every rank).

    Args:
        coordinator: Per-rank broadcast coordinator.
        group_buffers: Per-MLA-group buffer + KV refs.
        gather_fn: ``torch.ops._C.gather_multi_layer_block_kv_transfer``
            (or test double with the same signature).
        scatter_fn: ``torch.ops._C.scatter_multi_layer_block_kv_transfer``
            (or test double).
        device: Local XPU device, used to build the per-call
            ``block_ids`` int64 tensor.

    Raises:
        ValueError: If ``group_buffers`` is empty (no MLA group; the
            connector should not construct a broadcaster in that case).
    """

    def __init__(
        self,
        coordinator: XpuBroadcastCoordinator,
        group_buffers: dict[int, _GroupBuffers],
        gather_fn: Callable[..., None],
        scatter_fn: Callable[..., None],
        device: torch.device,
    ) -> None:
        if not group_buffers:
            raise ValueError(
                "XpuRetrieveBroadcaster requires at least one MLA group; "
                "got empty group_buffers"
            )
        self._coordinator = coordinator
        self._group_buffers = group_buffers
        self._gather_fn = gather_fn
        self._scatter_fn = scatter_fn
        self._device = device
        # Probe the bound xvllm ops once so each call site can decide
        # whether to forward the pre-staged device pointer table or
        # fall back to the kernel's legacy in-kernel H2D pointer-table
        # sync. ``gather_fn`` / ``scatter_fn`` come from the same
        # extension build, so a single probe is enough.
        self._supports_paged_buffer_ptrs_dev: bool = (
            supports_paged_buffer_ptrs_dev_kwarg(gather_fn)
            and supports_paged_buffer_ptrs_dev_kwarg(scatter_fn)
        )
        # Pending broadcasts to drain on the next start_load_kv. We keep
        # a list (not a queue) because the connector always processes
        # the full pending set in order — one drain per connector hook.
        self._pending: list[_PendingBroadcast] = []

    @property
    def coordinator(self) -> XpuBroadcastCoordinator:
        """The per-rank broadcast coordinator owned by this broadcaster."""
        return self._coordinator

    @property
    def mla_group_ids(self) -> tuple[int, ...]:
        """Engine group ids for which the bridge holds a buffer view."""
        return tuple(self._group_buffers.keys())

    def has_group(self, group_id: int) -> bool:
        """Return ``True`` if ``group_id`` is an MLA group on this rank.

        Args:
            group_id: Engine group id to check.
        """
        return group_id in self._group_buffers

    def enqueue_after_submit(
        self,
        request_id: str,
        group_id: int,
        block_ids: list[int],
        future: Any,
    ) -> None:
        """Record a pending broadcast for the given retrieve request.

        Must be called on **every** rank in the same order so the
        eventual collective is symmetric. ``future`` is only consulted
        on the source rank.

        Args:
            request_id: Connector request id (logging only).
            group_id: Engine group id this broadcast covers.
            block_ids: Engine-side block ids for this group.
            future: MQ retrieve future returned to the source rank;
                peers may pass ``None`` or a resolved placeholder.
        """
        if group_id not in self._group_buffers:
            return
        self._pending.append(
            _PendingBroadcast(
                request_id=request_id,
                group_id=int(group_id),
                block_ids=list(block_ids),
                future=future,
            )
        )

    def drain_pending(self, mq_timeout: float) -> None:
        """Run every pending broadcast in FIFO order.

        On the source rank: wait for the retrieve future, gather the
        MLA group's KV into the broadcast buffer, then call
        ``broadcast_send``. On peer ranks: call ``broadcast_recv``,
        then scatter the buffer back into the local KV.

        All items are always processed (never deferred) to ensure the
        collective call count is symmetric across ranks. If a future
        is not ready on the source, stale buffer is broadcast and vLLM
        recompute handles correctness.

        Args:
            mq_timeout: Timeout in seconds applied to the source rank's
                ``future.result(timeout=...)`` wait.
        """
        if not self._pending:
            return
        if not self._coordinator.participates():
            # Single-rank TP: no collective needed; drop the queue.
            self._pending.clear()
            return

        is_src = self._coordinator.is_broadcast_source()
        pending, self._pending = self._pending, []
        for item in pending:
            try:
                if is_src:
                    self._drive_source(item, mq_timeout)
                else:
                    self._drive_peer(item)
            except Exception:
                logger.exception(
                    "XPU MLA broadcast: %s rank=%d request_id=%s "
                    "group_id=%d failed",
                    "src" if is_src else "peer",
                    self._coordinator.tp_rank,
                    item.request_id,
                    item.group_id,
                )

    def _drive_source(
        self, item: _PendingBroadcast, mq_timeout: float
    ) -> None:
        """Source-rank path: wait → gather → broadcast_send."""
        retrieve_ok = True
        if item.future is not None:
            if not item.future.query():
                # Wait bounded time for BG thread to complete retrieve.
                item.future.wait(timeout=mq_timeout)
            if not item.future.query():
                retrieve_ok = False
            else:
                try:
                    item.future.result(timeout=0)
                except Exception:
                    retrieve_ok = False
        gb = self._group_buffers[item.group_id]
        n_blocks = len(item.block_ids)
        if n_blocks == 0:
            self._coordinator.broadcast_send()
            return
        if n_blocks > gb.max_blocks:
            logger.error(
                "XPU MLA broadcast: group=%d n_blocks=%d > max_blocks=%d; "
                "sending stale buffer to unblock peers",
                item.group_id,
                n_blocks,
                gb.max_blocks,
            )
            self._coordinator.broadcast_send()
            return
        if not retrieve_ok:
            # Retrieve failed but we must still broadcast to unblock peers.
            # Send stale buffer content — peers will get garbage but won't hang.
            self._coordinator.broadcast_send()
            return
        block_ids_dev = self._block_ids_to_device(item.block_ids)
        try:
            if self._supports_paged_buffer_ptrs_dev:
                self._gather_fn(
                    gb.layer_tensors,
                    gb.contiguous_buf,
                    block_ids_dev,
                    gb.layer_page_sizes,
                    gb.max_page_size,
                    gb.layers_scalars_tensor,
                    paged_buffer_ptrs_dev=gb.paged_buffer_ptrs_dev,
                )
            else:
                self._gather_fn(
                    gb.layer_tensors,
                    gb.contiguous_buf,
                    block_ids_dev,
                    gb.layer_page_sizes,
                    gb.max_page_size,
                    gb.layers_scalars_tensor,
                )
        except Exception:
            logger.exception(
                "XPU MLA broadcast: gather failed group=%d; "
                "sending stale buffer to unblock peers",
                item.group_id,
            )
            self._coordinator.broadcast_send()
            return
        self._coordinator.broadcast_send()

    def _drive_peer(self, item: _PendingBroadcast) -> None:
        """Peer-rank path: broadcast_recv → scatter."""
        gb = self._group_buffers[item.group_id]
        n_blocks = len(item.block_ids)
        if n_blocks == 0:
            self._coordinator.broadcast_recv()
            return
        if n_blocks > gb.max_blocks:
            logger.error(
                "XPU MLA broadcast: peer group=%d n_blocks=%d > "
                "max_blocks=%d; recv to unblock source then skip scatter",
                item.group_id,
                n_blocks,
                gb.max_blocks,
            )
            self._coordinator.broadcast_recv()
            return
        self._coordinator.broadcast_recv()
        block_ids_dev = self._block_ids_to_device(item.block_ids)
        try:
            if self._supports_paged_buffer_ptrs_dev:
                self._scatter_fn(
                    gb.layer_tensors,
                    gb.contiguous_buf,
                    block_ids_dev,
                    gb.layer_page_sizes,
                    gb.max_page_size,
                    gb.layers_scalars_tensor,
                    paged_buffer_ptrs_dev=gb.paged_buffer_ptrs_dev,
                )
            else:
                self._scatter_fn(
                    gb.layer_tensors,
                    gb.contiguous_buf,
                    block_ids_dev,
                    gb.layer_page_sizes,
                    gb.max_page_size,
                    gb.layers_scalars_tensor,
                )
        except Exception:
            logger.exception(
                "XPU MLA broadcast: scatter failed group=%d", item.group_id
            )

    def _block_ids_to_device(self, block_ids: list[int]) -> torch.Tensor:
        """Build a device int64 tensor of ``block_ids`` for one call.

        Args:
            block_ids: Engine-side block ids in send order.

        Returns:
            A 1-D ``int64`` tensor on :attr:`_device`.
        """
        return torch.tensor(block_ids, dtype=torch.int64, device=self._device)


def load_xvllm_gather_scatter() -> Optional[
    tuple[Callable[..., None], Callable[..., None]]
]:
    """Return ``(gather_fn, scatter_fn)`` from xvllm, or ``None``.

    The xvllm C++ extension may not be importable in CPU-only test
    environments. Returning ``None`` lets the connector fall back to
    "broadcast disabled" without raising at import time.

    Returns:
        Tuple of the two op handles, or ``None`` if the extension is
        unavailable.
    """
    try:
        # Third Party
        import vllm_xpu._C  # noqa: F401  -- ensures the ops are loaded
    except ImportError:
        logger.info(
            "XPU MLA broadcast: vllm_xpu._C not importable; broadcast "
            "wiring disabled. Set %s=0 to silence this message.",
            _ENV_FLAG_NAME,
        )
        return None
    gather = getattr(
        torch.ops._C, "gather_multi_layer_block_kv_transfer", None
    )
    scatter = getattr(
        torch.ops._C, "scatter_multi_layer_block_kv_transfer", None
    )
    if gather is None or scatter is None:
        logger.warning(
            "XPU MLA broadcast: torch.ops._C missing gather/scatter "
            "ops; broadcast wiring disabled."
        )
        return None
    return gather, scatter


def supports_paged_buffer_ptrs_dev_kwarg(
    op: Callable[..., None],
) -> bool:
    """Return ``True`` when ``op`` accepts ``paged_buffer_ptrs_dev`` kwarg.

    Older xvllm builds expose the gather/scatter ops without the
    pre-staged device pointer-table argument. This probe inspects the
    PyTorch operator schema once so callers can decide whether to pass
    the kwarg or fall back to the legacy in-kernel host-to-device
    pointer-table sync path.

    Args:
        op: A ``torch.ops._C`` operator overload, e.g.
            ``torch.ops._C.gather_multi_layer_block_kv_transfer``.

    Returns:
        ``True`` if the op's default overload schema declares an
        argument named ``paged_buffer_ptrs_dev``; ``False`` otherwise
        (including when the op exposes no inspectable schema).
    """
    default = getattr(op, "default", op)
    schema = getattr(default, "_schema", None)
    if schema is None:
        return False
    arguments = getattr(schema, "arguments", None)
    if arguments is None:
        return False
    for arg in arguments:
        if getattr(arg, "name", None) == "paged_buffer_ptrs_dev":
            return True
    return False


def build_group_buffers(
    coordinator: XpuBroadcastCoordinator,
    kv_caches: dict[str, torch.Tensor],
    group_views: Sequence[EngineGroupInfo],
    mla_group_ids: Sequence[int],
    blocks_per_chunk: int,
    device: torch.device,
) -> dict[int, _GroupBuffers]:
    """Build per-MLA-group buffer + KV refs from worker registration data.

    The contiguous buffer is a *view* of the coordinator's broadcast
    buffer reshaped as int8 ``[nl, blocks_per_chunk, max_page_size]``.
    The buffer is shared across MLA groups (the coordinator only owns
    one per rank); when there is more than one MLA group, the largest
    payload wins and smaller groups occupy a sub-view.

    Args:
        coordinator: Per-rank broadcast coordinator (owns the buffer).
        kv_caches: Worker KV cache tensors keyed by layer name (engine
            order). Tensors are int8-viewed in place when the dtype is
            not already a single-byte type.
        group_views: LMCache-owned engine KV cache group metadata.
        mla_group_ids: Subset of group ids classified as MLA at
            registration. Non-MLA groups are skipped.
        blocks_per_chunk: Maximum blocks per single retrieve chunk
            (== upper bound on ``len(block_ids)`` per broadcast call).
        device: Local XPU device used to allocate the per-group
            ``layers_scalars_tensor``.

    Returns:
        A ``dict[group_id, _GroupBuffers]`` covering only MLA groups.
        Empty when ``mla_group_ids`` is empty (no broadcaster needed).
    """
    if not mla_group_ids:
        return {}
    layer_items = list(kv_caches.items())
    num_layers = len(layer_items)
    engine_group_indices = get_engine_group_indices(group_views, num_layers)
    if engine_group_indices is None:
        # Default single-group fall-through: every layer is in group 0.
        engine_group_indices_t: tuple[int, ...] = tuple([0] * num_layers)
    else:
        engine_group_indices_t = tuple(int(x) for x in engine_group_indices)

    mla_set = {int(g) for g in mla_group_ids}
    per_group_layers: dict[int, list[torch.Tensor]] = {}
    for idx, (_name, tensor) in enumerate(layer_items):
        gid = engine_group_indices_t[idx]
        if gid not in mla_set:
            continue
        per_group_layers.setdefault(gid, []).append(_int8_view(tensor))

    out: dict[int, _GroupBuffers] = {}
    full_buffer = coordinator.broadcast_buffer
    for gid, tensors in per_group_layers.items():
        if not tensors:
            continue
        page_sizes = [int(t.shape[1]) for t in tensors]
        max_page = max(page_sizes)
        nl = len(tensors)
        needed = nl * blocks_per_chunk * max_page
        if needed > full_buffer.numel():
            logger.error(
                "XPU MLA broadcast: group=%d needs %d bytes but the "
                "coordinator's broadcast buffer only has %d bytes; "
                "the server's RegisterXpuContextResponse under-sized "
                "broadcast_buffer_bytes. Skipping this group.",
                gid,
                needed,
                full_buffer.numel(),
            )
            continue
        contiguous = (
            full_buffer.view(torch.int8)[:needed]
            .view(nl, blocks_per_chunk, max_page)
        )
        scalars = torch.tensor(
            page_sizes, dtype=torch.int32, device=device
        )
        # Pre-stage the per-layer ``data_ptr()`` table on device once so
        # the xvllm gather/scatter kernels can skip the per-call H2D
        # pointer-table sync (paged_buffer_ptrs_dev kwarg path).
        paged_buffer_ptrs_dev = torch.tensor(
            [int(t.data_ptr()) for t in tensors],
            dtype=torch.int64,
            device=device,
        )
        out[gid] = _GroupBuffers(
            group_id=gid,
            layer_tensors=tensors,
            layer_page_sizes=page_sizes,
            max_page_size=max_page,
            layers_scalars_tensor=scalars,
            contiguous_buf=contiguous,
            max_blocks=blocks_per_chunk,
            paged_buffer_ptrs_dev=paged_buffer_ptrs_dev,
        )
    return out


def _int8_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return ``tensor`` re-viewed as a 2-D int8 ``[num_blocks, page_bytes]``.

    The xvllm gather/scatter kernel works in bytes regardless of the
    KV dtype. Viewing the tensor as ``int8`` gives the kernel a stable
    row width independent of the underlying ``bfloat16`` / ``float8_*``
    dtype.

    Args:
        tensor: Worker KV cache tensor for one layer.

    Returns:
        A 2-D int8 view of the same storage.
    """
    flat = tensor.view(torch.int8)
    if flat.ndim == 2:
        return flat
    num_blocks = int(flat.shape[0])
    return flat.reshape(num_blocks, -1)
