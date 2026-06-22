# SPDX-License-Identifier: Apache-2.0
"""XPU MLA TP0→peers retrieve broadcast coordinator (XPU offload v2 §9.4).

The MLA KV cache is TP-replicated, so naively letting every TP rank issue
its own server-side H2D + scatter wastes ``tp_size``× of the bottleneck H2D
bandwidth. The design (§3.5 / §7.2 / §9.4) instead routes the MLA-group
retrieve through a single ``src`` rank (TP0 by default) and relies on
XCCL's intra-TP broadcast to publish the result to peers:

::

    TP0 worker (forward main thread, after retrieve future resolves):
        gather  TP0 KV cache → broadcast_buffer    (process-local op)
        broadcast(broadcast_buffer, src=0)         (XCCL collective)

    non-TP0 worker (forward main thread, same hook point):
        broadcast(broadcast_buffer, src=0)         (receive)
        scatter broadcast_buffer → own KV cache    (process-local op)

This file provides :class:`XpuBroadcastCoordinator`, a process-local
helper that owns the broadcast buffer and exposes the four primitives
the connector calls in order at the retrieve hook (``broadcast_send`` /
``broadcast_recv``). The actual ``gather`` / ``scatter`` between local
KV and the broadcast buffer is left to the caller — they reuse the
xvllm ``gather_multi_layer_block_kv_transfer`` /
``scatter_multi_layer_block_kv_transfer`` ops the rest of the project
already depends on (§3.8), so this module never imports those ops
directly.

The coordinator does **no** TP0-only short-circuiting on its own; the
connector is still responsible for skipping ``submit_retrieve`` on
non-source ranks for MLA groups (§7.2 row 1). The coordinator only
provides the building blocks for a correct broadcast call.
"""

# Standard
from collections.abc import Callable
from typing import Any

# Third Party
import torch

# First Party
from lmcache.utils import init_logger

logger = init_logger(__name__)


# (tensor, src) -> tensor : matches the signature exposed by
# ``vllm.distributed.parallel_state.get_tp_group().broadcast``.
BroadcastFn = Callable[[torch.Tensor, int], Any]


# Default broadcast source for MLA groups (XPU offload v2 §9.4).
# The TP0/TP7 chunk-rotated source variant (§7.3) is a follow-up.
MLA_BROADCAST_SRC: int = 0


class XpuBroadcastCoordinator:
    """MLA TP0→peers retrieve broadcast helper (XPU offload v2 §9.4).

    Owns the per-rank XCCL broadcast buffer and provides the
    ``broadcast_send`` / ``broadcast_recv`` primitives the connector
    calls at the retrieve hook. Does not own the TP routing decision —
    the caller decides which rank is the source and is responsible
    for ensuring the buffer is populated (gather) on the source rank
    before calling :meth:`broadcast_send`, and consumed (scatter) on
    peer ranks after :meth:`broadcast_recv`.

    Args:
        broadcast_fn: TP-group broadcast callable, e.g.
            ``vllm.distributed.parallel_state.get_tp_group().broadcast``.
        broadcast_buffer: Per-rank contiguous device buffer used as the
            collective payload. Must be allocated on the worker's local
            XPU device with the worker-reported size (returned by the
            server in :class:`RegisterXpuContextResponse`). Passing an
            empty buffer is a programming error.
        tp_rank: Worker's TP rank within the inference engine TP group.
        tp_size: TP size of the inference engine TP group. Coordinators
            with ``tp_size <= 1`` are degenerate and never call the
            collective; the connector should skip the broadcast hook.
        src_rank: TP rank that drives the H2D + scatter on the server
            and acts as the broadcast source. Defaults to
            :data:`MLA_BROADCAST_SRC` (``0``). Must be in
            ``[0, tp_size)``.

    Raises:
        ValueError: If ``broadcast_buffer`` is empty, ``tp_size`` is
            non-positive, or ``src_rank`` is out of range.
    """

    def __init__(
        self,
        broadcast_fn: BroadcastFn,
        broadcast_buffer: torch.Tensor,
        tp_rank: int,
        tp_size: int,
        src_rank: int = MLA_BROADCAST_SRC,
    ) -> None:
        if broadcast_buffer.numel() == 0:
            raise ValueError(
                "XpuBroadcastCoordinator requires a non-empty broadcast buffer; "
                "the server reported broadcast_buffer_bytes==0 (no MLA group?)."
            )
        if tp_size <= 0:
            raise ValueError(
                f"XpuBroadcastCoordinator requires tp_size > 0, got {tp_size}"
            )
        if not 0 <= src_rank < tp_size:
            raise ValueError(
                f"XpuBroadcastCoordinator src_rank={src_rank} out of range "
                f"[0, {tp_size})"
            )
        self._broadcast_fn = broadcast_fn
        self._broadcast_buffer = broadcast_buffer
        self._tp_rank = int(tp_rank)
        self._tp_size = int(tp_size)
        self._src_rank = int(src_rank)

    @property
    def broadcast_buffer(self) -> torch.Tensor:
        """Per-rank contiguous device buffer used as the collective payload."""
        return self._broadcast_buffer

    @property
    def tp_rank(self) -> int:
        """Worker's TP rank within the inference engine TP group."""
        return self._tp_rank

    @property
    def tp_size(self) -> int:
        """TP size of the inference engine TP group."""
        return self._tp_size

    @property
    def src_rank(self) -> int:
        """Broadcast source TP rank for MLA retrieve (default ``0``)."""
        return self._src_rank

    def is_broadcast_source(self) -> bool:
        """Return ``True`` if this rank drives the server retrieve + broadcast.

        The connector uses this to decide whether to call
        ``submit_retrieve`` (source) or skip the MQ round-trip and only
        receive via :meth:`broadcast_recv` (peer).

        Returns:
            ``True`` when ``tp_rank == src_rank``.
        """
        return self._tp_rank == self._src_rank

    def participates(self) -> bool:
        """Return ``True`` when XCCL broadcast is meaningful.

        Single-rank TP groups do not need a collective at all; the
        connector should skip both broadcast calls and the gather/scatter
        bridge in that case.

        Returns:
            ``True`` when ``tp_size > 1``.
        """
        return self._tp_size > 1

    def broadcast_send(self) -> None:
        """Publish the source rank's broadcast buffer to TP peers.

        Caller is responsible for filling :attr:`broadcast_buffer` with
        the gathered MLA-group KV before calling this. Must only be
        called on the source rank.

        Raises:
            RuntimeError: If invoked on a non-source rank.
        """
        if not self.is_broadcast_source():
            raise RuntimeError(
                "broadcast_send() called on non-source TP rank "
                f"{self._tp_rank} (src_rank={self._src_rank})"
            )
        if not self.participates():
            return
        self._broadcast_fn(self._broadcast_buffer, self._src_rank)

    def broadcast_recv(self) -> None:
        """Receive the broadcast buffer from the source rank.

        Returns once the buffer has been populated by the collective.
        Caller is responsible for scattering the buffer back into the
        local MLA KV cache. May be called on any rank — sources also
        participate in the collective to keep the calling convention
        symmetric, but should call :meth:`broadcast_send` instead so the
        intent is explicit at the call site.

        Raises:
            RuntimeError: If invoked on the source rank (use
                :meth:`broadcast_send` instead).
        """
        if self.is_broadcast_source():
            raise RuntimeError(
                "broadcast_recv() called on the source TP rank "
                f"{self._tp_rank}; use broadcast_send() instead"
            )
        if not self.participates():
            return
        self._broadcast_fn(self._broadcast_buffer, self._src_rank)
