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
"""Test automatic model-config binding and packed graph inputs on CPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from hyper_parallel.compile import GraphCompiler, PassConfig
from hyper_parallel.data.batching.runtime_input import AttentionRuntime
from hyper_parallel.distributed.encoder_parallel.batch_input import EncoderParallelInputAdapter
from hyper_parallel.models.deepseek_v41.adapter.data.runtime import DeepseekV41Runtime
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.runtime.model_integration import DisabledModelIntegrationSession


class _PackedModel(torch.nn.Module):
    """Consume every prepared tensor used by the packed reference path."""

    def __init__(self) -> None:
        """Initialize the boundary-dependent test weight."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(
        self, input_ids: torch.Tensor, packed_seq_params: object,
    ) -> torch.Tensor:
        """Produce boundary-dependent loss and weight gradients."""
        packed = packed_seq_params.prepare(input_ids.device, (0, 1, 2))
        values = (
            input_ids.float()
            + packed.local_segment_starts(input_ids.device).float()
            + packed.minimum_key_indices(input_ids.device, 2).float()
            + packed.segment_start_mask(input_ids.device).float()
        )
        return (self.weight * values).square().mean()


def _train_fn(model, model_inputs):
    """Follow the compiler training protocol without distributed state."""
    return model(**model_inputs), {}


class TestDeepseekV41Runtime(unittest.TestCase):
    """Use final model ratios without a graph-specific constructor option."""

    @staticmethod
    def _runtime() -> DeepseekV41Runtime:
        """Use the same field ownership as the text training recipes."""
        return DeepseekV41Runtime(include_position_ids=False, include_image_sequence_start=False)

    @staticmethod
    def _batch(boundaries=(0, 4, 8), local_length=8):
        """Build one CPU packed batch with fixed physical token length."""
        return {
            "input_ids": torch.arange(local_length).unsqueeze(0),
            "cu_seq_lens": torch.tensor(boundaries, dtype=torch.int32),
        }

    @staticmethod
    def _build(runtime, batch, *, cp_rank=0, cp_size=1):
        """Exercise the ordinary adapter entry after CP sharding."""
        return runtime.build(
            batch=batch,
            parallel_context=SimpleNamespace(
                tp_rank=0, tp_world_size=1, cp_rank=cp_rank, cp_world_size=cp_size,
            ),
        )

    @staticmethod
    def _attach(runtime, config):
        """Use the actual Trainer binding path with diagnostics disabled."""
        base = BaseTrainer.__new__(BaseTrainer)
        base.get_batch = SimpleNamespace(runtime_input_adapter=runtime)
        base.model_config = config
        base.model_assets = (config,)
        base.model_integration = DisabledModelIntegrationSession()
        base.attach_model_integration_data_pipeline()

    def test_trainer_binds_final_ratios_with_diagnostics_disabled(self) -> None:
        """Automatic preparation must not depend on validation being enabled."""
        runtime = self._runtime()
        self._attach(runtime, SimpleNamespace(v41_compress_ratios=[0, 4, 1, 4]))

        packed = self._build(runtime, self._batch((0, 2, 8)))["packed_seq_params"]
        self.assertFalse(packed.compression_aligned(4))

    def test_automatic_preparation_matches_standalone_eager_metadata(self) -> None:
        """Binding changes preparation timing without changing masks or minima."""
        batch = self._batch()
        lazy = self._build(self._runtime(), batch)["packed_seq_params"]
        expected = lazy.prepare(torch.device("cpu"), (0, 1, 2))
        runtime = self._runtime()
        self._attach(runtime, SimpleNamespace(v41_compress_ratios=[0, 1, 2]))
        actual = self._build(runtime, batch)["packed_seq_params"]

        self.assertIs(actual.prepare(torch.device("cpu"), (0, 1, 2)), actual)
        for method in ("local_segment_starts", "segment_start_mask"):
            torch.testing.assert_close(
                getattr(actual, method)(torch.device("cpu")),
                getattr(expected, method)(torch.device("cpu")),
            )
        torch.testing.assert_close(
            actual.minimum_key_indices(torch.device("cpu"), 2),
            expected.minimum_key_indices(torch.device("cpu"), 2),
        )

    def test_joint_graph_replays_changed_boundaries_and_gradients(self) -> None:
        """Each real batch supplies fresh metadata to the same joint graph."""
        runtime = self._runtime()
        self._attach(runtime, SimpleNamespace(v41_compress_ratios=[0, 1, 2]))
        model = _PackedModel()
        compiler = GraphCompiler(
            model, _train_fn, pass_config=PassConfig(fsdp_enabled=False), device=torch.device("cpu"),
        )
        graph_ids = []
        losses = []
        for boundaries in ((0, 4, 8), (0, 2, 8)):
            batch = self._batch(boundaries)
            model_inputs = {"input_ids": batch["input_ids"], **self._build(runtime, batch)}
            eager_loss = model(**model_inputs)
            expected_gradient = torch.autograd.grad(eager_loss, model.weight)[0]
            model.zero_grad(set_to_none=True)

            loss, _ = compiler.forward_backward(model_inputs=model_inputs)

            torch.testing.assert_close(loss, eager_loss.detach())
            torch.testing.assert_close(model.weight.grad, expected_gradient)
            graph_ids.append(id(compiler._joint_graph))  # pylint: disable=protected-access
            losses.append(float(loss.detach()))
        self.assertEqual(graph_ids[0], graph_ids[1])
        self.assertNotEqual(losses[0], losses[1])

    def test_invalid_second_batch_is_rejected_before_graph_execution(self) -> None:
        """Boundary validation runs for later batches as well as the first."""
        runtime = self._runtime()
        self._attach(runtime, SimpleNamespace(v41_compress_ratios=[0, 1, 2]))
        self._build(runtime, self._batch())

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self._build(runtime, self._batch((0, 9, 8)))

    def test_rebinding_uses_new_final_model_configuration(self) -> None:
        """A changed model configuration cannot retain the previous ratio set."""
        runtime = self._runtime()
        runtime.bind_model_config(SimpleNamespace(v41_compress_ratios=[0, 1, 2]))
        self._build(runtime, self._batch((0, 2, 8)))
        runtime.bind_model_config(SimpleNamespace(v41_compress_ratios=[0, 1, 4]))

        packed = self._build(runtime, self._batch((0, 2, 8)))["packed_seq_params"]
        self.assertFalse(packed.compression_aligned(4))

    def test_missing_model_compression_configuration_is_rejected(self) -> None:
        """Do not silently assume the debug model's compression ratios."""
        with self.assertRaisesRegex(ValueError, "model_config.v41_compress_ratios"):
            self._runtime().bind_model_config(SimpleNamespace())

    def test_cp_local_metadata_uses_global_document_boundaries(self) -> None:
        """Automatic preparation preserves the packed CP shard geometry."""
        runtime = self._runtime()
        runtime.bind_model_config(SimpleNamespace(v41_compress_ratios=[0, 1, 2]))
        batch = self._batch((0, 2, 8), local_length=4)

        packed = self._build(runtime, batch, cp_rank=1, cp_size=2)["packed_seq_params"]

        torch.testing.assert_close(packed.local_segment_starts(torch.device("cpu")), torch.full((1, 4), 2))
        torch.testing.assert_close(packed.minimum_key_indices(torch.device("cpu"), 2), torch.ones(1, 4).long())
        self.assertFalse(bool(packed.segment_start_mask(torch.device("cpu")).any()))

    def test_encoder_wrapper_forwards_model_config_binding(self) -> None:
        """A composed adapter must not hide the inner model's preparation."""
        encoder_runtime = Mock()
        encoder_runtime.preflight_batch.return_value = ()
        wrapper = EncoderParallelInputAdapter(runtime=encoder_runtime, inner=self._runtime())
        self._attach(wrapper, SimpleNamespace(v41_compress_ratios=[0, 1, 4]))

        packed = self._build(wrapper, self._batch((0, 2, 8)))["packed_seq_params"]
        self.assertFalse(packed.compression_aligned(4))
        encoder_runtime.finalize_plan.assert_called_once_with()

    def test_generic_adapter_and_absent_adapter_remain_supported(self) -> None:
        """The default binding hook is neutral for model-independent adapters."""
        runtime = AttentionRuntime(mode="compressed", create_mask=False, reset_mask=False, sliding_window=None)
        self._attach(runtime, SimpleNamespace())
        self.assertEqual(self._build(runtime, self._batch()), {"attention_mask": None})
        self._attach(None, SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
