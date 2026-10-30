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
"""Unit tests for cadence-controlled router auxiliary-loss logging."""

import subprocess
import sys
from textwrap import dedent
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch

from hyper_parallel.components.modules.moe_utils import MoEMonitorCallback
from hyper_parallel.trainer.callbacks.logging_callback import LoggingCallback
from hyper_parallel.trainer.state import TrainerState
from tests.common.mark_utils import arg_mark


class TestLoggingCallback(unittest.TestCase):
    """Verify router metrics use the normal rank, cadence and terminal output path."""

    @staticmethod
    def _trainer(monitor=None, logging_steps: int = 2, rank: int = 0) -> SimpleNamespace:
        """Build the logging callback's minimal Trainer surface."""
        return SimpleNamespace(
            mesh=None, global_rank=rank, moe_monitor=monitor,
            config=SimpleNamespace(training=SimpleNamespace(logging_steps=logging_steps)),
            step_env_metrics={"training/total_loss": 2.5}, tqdm_callback=None,
        )

    def test_graph_metric_does_not_read_capture_time_module_attributes(self) -> None:
        """Explicit graph outputs take precedence over stale forward-side state."""
        monitor = Mock()
        callback = LoggingCallback(self._trainer(monitor))
        with (
            patch.object(type(monitor), "last_mean_aux_loss", new_callable=PropertyMock, create=True) as read,
            patch("hyper_parallel.trainer.callbacks.logging_callback.logger.info") as log,
        ):
            read.side_effect = RuntimeError("capture-time FakeTensor")
            callback.on_step_end(TrainerState(global_step=2), loss=2.5,
                                 loss_dict={"router_aux_loss": torch.tensor(1.75)}, grad_norm=1.0)
        read.assert_not_called()
        self.assertIn("training/router_aux_loss=1.75", log.call_args.args[1])

    def test_tensor_metric_is_traceable_and_replays_new_values(self) -> None:
        """The graph captures the metric calculation without converting to Python."""
        layers = [SimpleNamespace(last_aux_loss=None) for _ in range(2)]
        monitor = MoEMonitorCallback(SimpleNamespace(modules=lambda: iter(layers)))

        def metric(values: torch.Tensor) -> torch.Tensor:
            """Evaluate the graph regression fixture."""
            for index, layer in enumerate(layers):
                layer.last_aux_loss = values[index]
            return monitor.last_mean_aux_loss_tensor

        graph = torch.fx.experimental.proxy_tensor.make_fx(metric)(torch.tensor([1., 2.]))
        torch.testing.assert_close(graph(torch.tensor([3., 5.])), torch.tensor(4.))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_callback_import_does_not_load_model_runtime(self) -> None:
        """
        Feature: CPU callback imports.
        Description: Import logging in a fresh interpreter with model dependencies unavailable.
        Expectation: Logging callbacks load without importing the Transformers model runtime.
        """
        script = dedent("""
            import importlib.abc
            import sys

            class RejectModelRuntime(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path, target=None):
                    if fullname == "transformers.modeling_utils":
                        raise ImportError(f"Unexpected model dependency: {fullname}")
                    return None

            sys.meta_path.insert(0, RejectModelRuntime())
            import hyper_parallel.trainer.callbacks as callbacks
            from hyper_parallel.trainer.callbacks.logging_callback import LoggingCallback

            assert callbacks.LoggingCallback is LoggingCallback, (
                f"Expected {LoggingCallback!r}, got {callbacks.LoggingCallback!r}")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False, timeout=60,
        )
        self.assertEqual(
            result.returncode, 0,
            f"Expected import exit code 0, got {result.returncode}: {result.stderr}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_raw_aux_loss_is_logged_with_bias_updates_disabled(self) -> None:
        """
        Feature: Auxiliary-loss logging.
        Description: The real monitor supplies mean raw router losses independently of bias updates.
        Expectation: Rank zero prints one aux metric without modifying shared metrics or the main loss.
        """
        layers = [SimpleNamespace(last_aux_loss=torch.tensor(value)) for value in (1.0, 1.5)]
        monitor = MoEMonitorCallback(SimpleNamespace(modules=lambda: iter(layers)))
        trainer = self._trainer(monitor)
        callback = LoggingCallback(trainer)
        state = TrainerState(global_step=2)
        with patch("hyper_parallel.trainer.callbacks.logging_callback.logger.info") as log:
            callback.on_step_end(state, loss=2.5, loss_dict={}, grad_norm=1.0)
            callback.on_step_end(state, loss=2.5, loss_dict={}, grad_norm=1.0)
        log.assert_called_once()
        message = log.call_args.args[1]
        self.assertIn("training/router_aux_loss=1.25", message)
        self.assertIn("training/total_loss=2.5", message)
        self.assertEqual(trainer.step_env_metrics, {"training/total_loss": 2.5})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_skipped_logs_do_not_read_aux_device_scalars(self) -> None:
        """
        Feature: Auxiliary-loss logging.
        Description: Disabled logs, nonzero ranks and intermediate steps must not read auxiliary scalars.
        Expectation: The aux property is never evaluated and no message is emitted.
        """
        for cadence, rank, step in ((0, 0, 2), (2, 1, 2), (2, 0, 1)):
            with self.subTest(cadence=cadence, rank=rank, step=step):
                monitor = Mock()
                trainer = self._trainer(monitor, logging_steps=cadence, rank=rank)
                callback = LoggingCallback(trainer)
                with (
                    patch.object(type(monitor), "last_mean_aux_loss", new_callable=PropertyMock, create=True) as read,
                    patch("hyper_parallel.trainer.callbacks.logging_callback.logger.info") as log,
                ):
                    callback.on_step_end(TrainerState(global_step=step), loss=2.5, loss_dict={}, grad_norm=1.0)
                read.assert_not_called()
                log.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_missing_aux_is_omitted_and_zero_aux_is_logged(self) -> None:
        """
        Feature: Auxiliary-loss logging.
        Description: Missing or disabled aux is omitted while a computed zero remains visible.
        Expectation: Ordinary loss logging continues and zero aux is not mistaken for a missing metric.
        """
        for monitor in (None, SimpleNamespace(last_mean_aux_loss=None), SimpleNamespace(last_mean_aux_loss=0.0)):
            with self.subTest(monitor=monitor):
                callback = LoggingCallback(self._trainer(monitor))
                with patch("hyper_parallel.trainer.callbacks.logging_callback.logger.info") as log:
                    callback.on_step_end(TrainerState(global_step=2), loss=2.5, loss_dict={}, grad_norm=1.0)
                message = log.call_args.args[1]
                if monitor is not None and monitor.last_mean_aux_loss is not None:
                    self.assertIn("training/router_aux_loss=0", message)
                else:
                    self.assertNotIn("training/router_aux_loss", message)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_aux_log_uses_active_tqdm_writer(self) -> None:
        """
        Feature: Auxiliary-loss logging.
        Description: Router auxiliary metrics share the active progress bar's structured output path.
        Expectation: The tqdm writer receives the aux field and standard logging does not duplicate it.
        """
        trainer = self._trainer(SimpleNamespace(last_mean_aux_loss=1.25))
        trainer.tqdm_callback = SimpleNamespace(write=Mock(return_value=True))
        callback = LoggingCallback(trainer)
        with patch("hyper_parallel.trainer.callbacks.logging_callback.logger.info") as log:
            callback.on_step_end(TrainerState(global_step=2), loss=2.5, loss_dict={}, grad_norm=1.0)
        trainer.tqdm_callback.write.assert_called_once()
        self.assertIn("training/router_aux_loss=1.25", trainer.tqdm_callback.write.call_args.args[0])
        log.assert_not_called()
