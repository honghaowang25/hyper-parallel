# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Independent natural-document pooling, visibility and graph replay regressions."""

import unittest

import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten, tree_unflatten

from hyper_parallel.compile.tracer.graph_tracer import run_traced_graph, trace_model_graph
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedPackedSequence, compressed_causal_topk, compressed_causal_topk_and_candidates,
    compressed_candidate_topk, reference_sparse_attention,
)
from hyper_parallel.models.deepseek_v41.aux_loss import sequence_load_balancing_loss


def _prepare(boundaries: list[int]) -> SharedCompressedPackedSequence:
    """Prepare the same geometry that the standard model runtime supplies."""
    length = boundaries[-1]
    packed = SharedCompressedPackedSequence(torch.tensor(boundaries), 0, length, length)
    packed = packed.prepare(torch.device("cpu"), (0, 1, 2))
    packed.prepare_graph_geometry(torch.device("cpu"), (0, 1, 2))
    return packed


def _pool_oracle(key: torch.Tensor, gate: torch.Tensor, boundaries: list[int]) -> torch.Tensor:
    """Pool each document separately, dropping only its incomplete tail."""
    values = []
    for start, end in zip(boundaries, boundaries[1:]):
        for row in range(start, end - 1, 2):
            values.append((key[:, row:row+2] * gate[:, row:row+2].float().softmax(1)).sum(1))
    return torch.stack(values, 1) if values else key[:, :0] + gate[:, :0].sum() * 0.0


def _attention_oracle(
        query: torch.Tensor, key: torch.Tensor, gate: torch.Tensor, boundaries: list[int],
) -> torch.Tensor:
    """Compute dense attention independently over completed own-document blocks."""
    pooled = _pool_oracle(key, gate, boundaries)
    rows = []
    base = 0
    for start, end in zip(boundaries, boundaries[1:]):
        for row in range(start, end):
            count = (row - start + 1) // 2
            visible = pooled[:, base:base+count]
            logits = (query[:, row:row+1] * visible).sum(-1)
            probability = torch.cat((logits, logits.new_zeros(1, 1)), -1).softmax(-1)[:, :-1]
            rows.append((probability.unsqueeze(-1) * visible).sum(1))
        base += (end-start) // 2
    return torch.stack(rows, 1)


class _CompressionAttention(nn.Module):
    """Exercise pooling, Top-K masks and attention with real trainable tensors."""

    def __init__(self) -> None:
        """Use deterministic, non-tied random scores."""
        super().__init__()
        generator = torch.Generator().manual_seed(107)
        self.key = nn.Parameter(torch.randn(1, 8, 3, generator=generator))
        self.gate = nn.Parameter(torch.randn(1, 8, 3, generator=generator))
        self.query = nn.Parameter(torch.randn(1, 8, 3, generator=generator))

    def forward(self, packed: SharedCompressedPackedSequence) -> torch.Tensor:
        """Return a scalar depending only on mathematically visible blocks."""
        packed = packed.prepare(self.key.device, (0, 1, 2))
        plan = packed.block_plan(2, 1, 0)
        bank = plan.pool(self.key, self.gate, None)
        indices = compressed_causal_topk(
            self.query.unsqueeze(2), bank, self.query.new_ones(1, 8, 1),
            compress_ratio=2, sparse_count=512,
            minimum_key_indices=packed.minimum_key_indices(self.key.device, 2),
            maximum_key_indices=packed.maximum_key_indices(self.key.device, 2),
        )
        output = reference_sparse_attention(
            self.query.unsqueeze(1), bank.unsqueeze(1), indices, self.query.new_zeros(1), 1.0,
        )
        return output.square().sum()


class TestPackedCompressionGraph(unittest.TestCase):
    """Cover changes in alignment, document count and empty compressed banks."""

    def test_fixed_schema_and_pool_match_independent_documents(self) -> None:
        """Capture once aligned, then replay natural and entirely incomplete documents."""
        initial, spec = tree_flatten(_prepare([0, 4, 8]))
        key = torch.randn(1, 8, 3)
        gate = torch.randn_like(key)

        def pool(key: torch.Tensor, gate: torch.Tensor, *leaves: torch.Tensor) -> torch.Tensor:
            """Pool using explicit tensor inputs in the fake trace."""
            packed = tree_unflatten(list(leaves), spec).prepare(torch.device("cpu"), (0, 1, 2))
            plan = packed.block_plan(2, 1, 0)
            return plan.pool(key, gate, None)

        graph = make_fx(pool, tracing_mode="fake")(key, gate, *initial)
        for boundaries in ([0, 4, 8], [0, 3, 8], list(range(9))):
            with self.subTest(boundaries=boundaries):
                leaves, updated_spec = tree_flatten(_prepare(boundaries))
                self.assertEqual(spec, updated_spec)
                actual = graph(key, gate, *leaves)
                expected = _pool_oracle(key, gate, boundaries)
                self.assertEqual(actual.shape, (1, 4, 3))
                torch.testing.assert_close(actual[:, :expected.shape[1]], expected)

    def test_rotary_positions_use_original_document_block_starts(self) -> None:
        """Non-aligned documents retain their own complete blocks' original positions."""
        packed = _prepare([0, 3, 8])
        leaves, spec = tree_flatten(packed)
        plan = tree_unflatten(leaves, spec).block_plan(2, 1, 0)
        cos = torch.arange(24.0).reshape(1, 8, 3)
        sin = cos + 0.25
        actual_cos, actual_sin = plan.positions(cos, sin, None)
        torch.testing.assert_close(actual_cos[:, :3], cos[:, [0, 3, 5]])
        torch.testing.assert_close(actual_sin[:, :3], sin[:, [0, 3, 5]])

    def test_natural_visibility_never_exposes_future_or_other_document_keys(self) -> None:
        """At the first token after an odd document tail, no compressed key is visible."""
        packed = _prepare([0, 3, 8])
        leaves, spec = tree_flatten(packed)
        restored = tree_unflatten(leaves, spec)
        indices = compressed_causal_topk(
            torch.ones(1, 8, 1, 1), torch.arange(4.0).view(1, 4, 1), torch.ones(1, 8, 1),
            compress_ratio=2, sparse_count=512,
            minimum_key_indices=restored.minimum_key_indices(torch.device("cpu"), 2),
            maximum_key_indices=restored.maximum_key_indices(torch.device("cpu"), 2),
        )[0]
        expected = [[], [0], [0], [], [1], [1], [1, 2], [1, 2]]
        for row, keys in zip(indices, expected):
            self.assertEqual(row[row >= 0].tolist(), keys)

    def test_joint_graph_loss_and_all_gradients_match_independent_attention(self) -> None:
        """Changing alignment/count reuses one joint graph and excludes all padded keys."""
        model = _CompressionAttention()

        def train_fn(module: nn.Module, packed: SharedCompressedPackedSequence) -> tuple:
            """Follow the production tracer scalar-loss protocol."""
            return module(packed), {}

        graph = trace_model_graph(model, train_fn, {"packed": _prepare([0, 4, 8])})
        for boundaries in ([0, 4, 8], [0, 3, 8], list(range(9)), [0, 1, 4, 5, 8]):
            with self.subTest(boundaries=boundaries):
                oracle = _attention_oracle(model.query, model.key, model.gate, boundaries).square().sum()
                expected = torch.autograd.grad(oracle, tuple(model.parameters()))
                model.zero_grad(set_to_none=True)
                actual, gradients, _ = run_traced_graph(graph, model, {"packed": _prepare(boundaries)})
                torch.testing.assert_close(actual, oracle.detach(), atol=2e-6, rtol=2e-5)
                for actual_gradient, gradient in zip(gradients, expected):
                    torch.testing.assert_close(actual_gradient, gradient, atol=2e-6, rtol=2e-5)
                    self.assertTrue(bool(actual_gradient.isfinite().all()))

    def test_explicit_key_upper_bounds_support_multiple_batch_rows(self) -> None:
        """Upper bounds remain per batch/query rather than flattening the batch axis."""
        maximum = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
        indices = compressed_causal_topk(
            torch.ones(2, 4, 1, 1), torch.ones(2, 2, 1), torch.ones(2, 4, 1),
            compress_ratio=2, sparse_count=2, maximum_key_indices=maximum,
        )
        for batch in range(2):
            for row in range(4):
                selected = indices[batch, row]
                self.assertEqual(selected[selected >= 0].tolist(), list(range(int(maximum[batch, row]))))

    def test_joint_graph_guards_allow_odd_and_even_raw_lengths(self) -> None:
        """Equal initial dimensions must not tie raw tokens to flattened pooling rows."""
        model = nn.Linear(1, 1, bias=False)

        def train_fn(module: nn.Module, key: torch.Tensor, gate: torch.Tensor,
                     packed: SharedCompressedPackedSequence) -> tuple:
            """Evaluate pooled valid blocks through the real guarded joint-graph entry."""
            plan = packed.block_plan(2, 1, 0)
            values = plan.pool(key, gate, None)
            count = packed.maximum_key_indices(key.device, 2)[:, -1:]
            valid = torch.arange(values.shape[1]).unsqueeze(0) < count
            total = values.masked_fill(~valid.unsqueeze(-1), 0.0).sum().reshape(1, 1)
            return module(total).sum(), {}

        def inputs(boundaries: list[int]) -> dict:
            """Build CPU unit-test tensors with the runtime's public preparation contract."""
            length = boundaries[-1]
            return {"key": torch.randn(1, length, 3), "gate": torch.randn(1, length, 3),
                    "packed": _prepare(boundaries)}

        initial = inputs([0, 4, 8])
        graph = trace_model_graph(model, train_fn, initial, dynamic=True)
        for boundaries in ([0, 3, 7], [0, 3, 8], [0, 5, 10], [0, 3, 11]):
            with self.subTest(boundaries=boundaries):
                current = inputs(boundaries)
                expected = model(_pool_oracle(current["key"], current["gate"], boundaries).sum().reshape(1, 1)).sum()
                expected_gradient = torch.autograd.grad(expected, model.weight)[0]
                actual, gradients, _ = run_traced_graph(graph, model, current)
                torch.testing.assert_close(actual, expected.detach())
                torch.testing.assert_close(gradients[0], expected_gradient)

    def test_candidate_graph_replays_zero_and_nonzero_padding(self) -> None:
        """Candidate block grouping must not constrain raw length to multiples of eight."""
        model = nn.Linear(1, 1, bias=False)

        def train_fn(module: nn.Module, query: torch.Tensor, key: torch.Tensor,
                     weights: torch.Tensor) -> tuple:
            """Trace both Full candidate publication and Reindex consumption."""
            indices, candidates = compressed_causal_topk_and_candidates(
                query, key, weights, compress_ratio=1, sparse_count=512, topk_blocks=128, block_size=8,
            )
            refined = compressed_candidate_topk(
                query, key, weights, candidates, compress_ratio=1, sparse_count=512, block_size=8,
            )
            total = (indices.float().sum() + refined.float().sum()).reshape(1, 1)
            return module(total).sum(), {}

        def inputs(length: int) -> dict:
            """Create CPU module-test inputs, independent of the standard training data."""
            return {"query": torch.randn(1, length, 1, 3), "key": torch.randn(1, length, 3),
                    "weights": torch.randn(1, length, 1)}

        graph = trace_model_graph(model, train_fn, inputs(32), dynamic=True)
        for length in (31, 30, 29, 32, 33):
            with self.subTest(length=length):
                current = inputs(length)
                expected, _ = train_fn(model, **current)
                actual, gradients, _ = run_traced_graph(graph, model, current)
                torch.testing.assert_close(actual, expected.detach())
                torch.testing.assert_close(gradients[0], torch.autograd.grad(expected, model.weight)[0])

    def test_router_capacity_preserves_loss_and_gradients(self) -> None:
        """Unused sequence capacity neither dilutes the mean nor changes router gradients."""
        for boundaries in ([0, 4, 8], [0, 3, 8], list(range(9))):
            with self.subTest(boundaries=boundaries):
                packed = _prepare(boundaries)
                leaves, spec = tree_flatten(packed)
                restored = tree_unflatten(leaves, spec)
                ids, capacity = restored.router_metadata(torch.device("cpu"))
                original_ids, count = packed.router_metadata(torch.device("cpu"))
                torch.testing.assert_close(ids, original_ids)
                scores = torch.rand(8, 4).requires_grad_()
                experts = scores.topk(2, -1).indices
                actual = sequence_load_balancing_loss(scores, experts, ids.flatten(), capacity)
                expected = sequence_load_balancing_loss(scores, experts, original_ids.flatten(), count)
                torch.testing.assert_close(actual, expected)
                actual_grad = torch.autograd.grad(actual, scores, retain_graph=True)[0]
                expected_grad = torch.autograd.grad(expected, scores)[0]
                torch.testing.assert_close(actual_grad, expected_grad)

    def test_symbolic_geometry_replays_a_different_short_sequence_length(self) -> None:
        """Geometry TreeSpec remains stable when the raw sequence length changes."""
        leaves, spec = tree_flatten(_prepare([0, 4, 8]))

        def geometry(*inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Consume symbolic geometry without host boundary access."""
            packed = tree_unflatten(list(inputs), spec)
            return packed.local_positions(torch.device("cpu")), packed.maximum_key_indices(torch.device("cpu"), 2)

        graph = make_fx(geometry, tracing_mode="symbolic")(*leaves)
        updated, updated_spec = tree_flatten(_prepare([0, 3, 10]))
        self.assertEqual(spec, updated_spec)
        for actual, expected in zip(graph(*updated), geometry(*updated)):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
