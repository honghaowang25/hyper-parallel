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
"""Real CPU collectives validate dynamic joint graphs after FSDP sharding."""

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from hyper_parallel.compile import GraphCompiler, PassConfig


def _loss(model, x, y):
    loss = (model(x) - y).square().mean()
    return loss, {"mse": loss.detach()}


@pytest.mark.parametrize("overlap", [False, True])
def test_dynamic_fsdp(overlap):
    """Compare losses, reduced gradient shards and optimizer updates with eager."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 4))
        reference = copy.deepcopy(model)
        compiler = GraphCompiler(
            model, _loss, device=torch.device("cpu"),
            pass_config=PassConfig(fsdp_enabled=True, enable_overlap=overlap),
            dynamic_arg_dims={"x": [0, 1], "y": [0, 1]},
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, foreach=False)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-2, foreach=False)
        joint = None
        for batch, sequence in [(2, 5), (3, 7), (4, 3)]:
            torch.manual_seed(100 * batch + rank)
            x, y = (torch.randn(batch + rank, sequence + rank, 4) for _ in range(2))
            actual, losses = compiler.forward_backward(x=x, y=y)
            if joint is None:
                joint = compiler._joint_graph
            assert compiler._joint_graph is joint, f"Expected graph reuse on rank {rank}"
            expected, _ = _loss(reference, x, y)
            expected.backward()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(losses["mse"], expected.detach())
            for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
                # The existing graph FSDP pass uses SUM reduce-scatter.
                dist.all_reduce(ref_parameter.grad)
                shard = ref_parameter.grad.chunk(world_size, dim=0)[rank]
                torch.testing.assert_close(parameter.grad, shard)
            optimizer.step()
            ref_optimizer.step()
            for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter, ref_parameter.chunk(world_size, dim=0)[rank])
            optimizer.zero_grad()
            ref_optimizer.zero_grad()
    finally:
        dist.destroy_process_group()
