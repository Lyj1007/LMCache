# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache.integration.vllm.xpu_retrieve_broadcast``.

Cover the parts of :class:`XpuRetrieveBroadcaster` that are reachable
without a live xvllm/XPU stack: schema-aware kwarg detection, the
pre-staged ``paged_buffer_ptrs_dev`` field on :class:`_GroupBuffers`,
and the gather/scatter dispatch decision based on whether the bound
op exposes the new optional argument.
"""

# Standard
from typing import Any
from unittest.mock import MagicMock

# Third Party
import torch

# First Party
from lmcache.integration.vllm.xpu_retrieve_broadcast import (
    XpuRetrieveBroadcaster,
    _GroupBuffers,
    supports_paged_buffer_ptrs_dev_kwarg,
)


class _FakeArg:
    """Stand-in for ``torch._C.Argument`` carrying just a name."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSchema:
    """Stand-in for ``torch._C.FunctionSchema`` carrying just arguments."""

    def __init__(self, arg_names: list[str]) -> None:
        self.arguments = [_FakeArg(n) for n in arg_names]


class _FakeOpDefault:
    """Stand-in for ``torch.ops._C.<op>.default`` exposing ``_schema``."""

    def __init__(self, arg_names: list[str]) -> None:
        self._schema = _FakeSchema(arg_names)

    def __call__(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _FakeOp:
    """Stand-in for the dispatcher op handle: has a ``default`` attr."""

    def __init__(self, arg_names: list[str]) -> None:
        self.default = _FakeOpDefault(arg_names)
        self.last_args: tuple = ()
        self.last_kwargs: dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.last_args = args
        self.last_kwargs = dict(kwargs)


def _make_group_buffers(
    *,
    nl: int = 2,
    blocks_per_chunk: int = 4,
    max_page: int = 8,
    device: torch.device | None = None,
) -> _GroupBuffers:
    """Build a ``_GroupBuffers`` dummy that does not depend on XPU.

    Args:
        nl: Number of layers in the group.
        blocks_per_chunk: ``contiguous_buf`` block dimension.
        max_page: Per-layer max page bytes (uniform here).
        device: Device for the per-call tensors.

    Returns:
        A populated :class:`_GroupBuffers` whose tensors all live on
        ``device``. The CPU default is fine for the dispatcher tests
        because the spy op never reads the data.
    """
    if device is None:
        device = torch.device("cpu")
    layer_tensors = [
        torch.zeros(blocks_per_chunk * 2, max_page, dtype=torch.int8, device=device)
        for _ in range(nl)
    ]
    layer_page_sizes = [max_page] * nl
    layers_scalars_tensor = torch.tensor(
        layer_page_sizes, dtype=torch.int32, device=device
    )
    contiguous_buf = torch.zeros(
        nl, blocks_per_chunk, max_page, dtype=torch.int8, device=device
    )
    paged_buffer_ptrs_dev = torch.tensor(
        [int(t.data_ptr()) for t in layer_tensors],
        dtype=torch.int64,
        device=device,
    )
    return _GroupBuffers(
        group_id=0,
        layer_tensors=layer_tensors,
        layer_page_sizes=layer_page_sizes,
        max_page_size=max_page,
        layers_scalars_tensor=layers_scalars_tensor,
        contiguous_buf=contiguous_buf,
        max_blocks=blocks_per_chunk,
        paged_buffer_ptrs_dev=paged_buffer_ptrs_dev,
    )


def _make_coordinator(
    *, participates: bool = True, is_source: bool = True, tp_rank: int = 0
) -> Any:
    """Build a coordinator stub that satisfies the broadcaster's API.

    Args:
        participates: Value returned from ``participates()``.
        is_source: Value returned from ``is_broadcast_source()``.
        tp_rank: Reported TP rank (used in error logs only).

    Returns:
        A :class:`unittest.mock.MagicMock` configured with the methods
        the broadcaster exercises.
    """
    coord = MagicMock(name="XpuBroadcastCoordinator")
    coord.participates.return_value = participates
    coord.is_broadcast_source.return_value = is_source
    coord.tp_rank = tp_rank
    return coord


class TestSupportsPagedBufferPtrsDevKwarg:
    """Tests for ``supports_paged_buffer_ptrs_dev_kwarg``."""

    def test_true_when_kwarg_present(self) -> None:
        op = _FakeOp(
            [
                "gpu_tensors",
                "contiguous_buf",
                "block_ids",
                "layer_page_sizes",
                "max_page_size",
                "layers_scalars_tensor",
                "skip_prefix_n_blocks",
                "paged_buffer_ptrs_dev",
            ]
        )
        assert supports_paged_buffer_ptrs_dev_kwarg(op) is True

    def test_false_when_kwarg_absent(self) -> None:
        op = _FakeOp(
            [
                "gpu_tensors",
                "contiguous_buf",
                "block_ids",
                "layer_page_sizes",
                "max_page_size",
                "layers_scalars_tensor",
            ]
        )
        assert supports_paged_buffer_ptrs_dev_kwarg(op) is False

    def test_false_when_no_schema(self) -> None:
        def bare_callable(*_args: Any, **_kwargs: Any) -> None:
            return None

        assert supports_paged_buffer_ptrs_dev_kwarg(bare_callable) is False


class TestBroadcasterDispatch:
    """Tests for the source/peer dispatch path in XpuRetrieveBroadcaster."""

    def _make_broadcaster(
        self,
        *,
        gather_args: list[str],
        scatter_args: list[str],
    ) -> tuple[XpuRetrieveBroadcaster, _FakeOp, _FakeOp, _GroupBuffers]:
        """Build a broadcaster with stub ops + group buffers."""
        gather = _FakeOp(gather_args)
        scatter = _FakeOp(scatter_args)
        gb = _make_group_buffers()
        coord = _make_coordinator()
        broadcaster = XpuRetrieveBroadcaster(
            coordinator=coord,
            group_buffers={gb.group_id: gb},
            gather_fn=gather,
            scatter_fn=scatter,
            device=torch.device("cpu"),
        )
        return broadcaster, gather, scatter, gb

    def test_source_forwards_kwarg_when_supported(self) -> None:
        new_args = [
            "gpu_tensors",
            "contiguous_buf",
            "block_ids",
            "layer_page_sizes",
            "max_page_size",
            "layers_scalars_tensor",
            "skip_prefix_n_blocks",
            "paged_buffer_ptrs_dev",
        ]
        broadcaster, gather, _scatter, gb = self._make_broadcaster(
            gather_args=new_args, scatter_args=new_args
        )
        broadcaster.enqueue_after_submit(
            request_id="req-1", group_id=gb.group_id, block_ids=[1, 2], future=None
        )
        broadcaster.drain_pending(mq_timeout=1.0)

        assert "paged_buffer_ptrs_dev" in gather.last_kwargs
        assert gather.last_kwargs["paged_buffer_ptrs_dev"] is gb.paged_buffer_ptrs_dev

    def test_source_omits_kwarg_when_unsupported(self) -> None:
        old_args = [
            "gpu_tensors",
            "contiguous_buf",
            "block_ids",
            "layer_page_sizes",
            "max_page_size",
            "layers_scalars_tensor",
        ]
        broadcaster, gather, _scatter, gb = self._make_broadcaster(
            gather_args=old_args, scatter_args=old_args
        )
        broadcaster.enqueue_after_submit(
            request_id="req-1", group_id=gb.group_id, block_ids=[1, 2], future=None
        )
        broadcaster.drain_pending(mq_timeout=1.0)

        assert "paged_buffer_ptrs_dev" not in gather.last_kwargs

    def test_peer_forwards_kwarg_when_supported(self) -> None:
        new_args = [
            "gpu_tensors",
            "contiguous_buf",
            "block_ids",
            "layer_page_sizes",
            "max_page_size",
            "layers_scalars_tensor",
            "skip_prefix_n_blocks",
            "paged_buffer_ptrs_dev",
        ]
        gather = _FakeOp(new_args)
        scatter = _FakeOp(new_args)
        gb = _make_group_buffers()
        coord = _make_coordinator(participates=True, is_source=False)
        broadcaster = XpuRetrieveBroadcaster(
            coordinator=coord,
            group_buffers={gb.group_id: gb},
            gather_fn=gather,
            scatter_fn=scatter,
            device=torch.device("cpu"),
        )
        broadcaster.enqueue_after_submit(
            request_id="req-1", group_id=gb.group_id, block_ids=[1, 2], future=None
        )
        broadcaster.drain_pending(mq_timeout=1.0)

        assert "paged_buffer_ptrs_dev" in scatter.last_kwargs
        assert (
            scatter.last_kwargs["paged_buffer_ptrs_dev"]
            is gb.paged_buffer_ptrs_dev
        )
