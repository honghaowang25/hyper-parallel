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
"""Optional run.sh entry that audits the standard TextTrainer dynamic graph path."""

import json
import os
from functools import wraps
from pathlib import Path

import torch
from torch.utils import _pytree as pytree

from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.text_trainer import TextTrainer


def main() -> None:
    """Train normally and require multiple input lengths with one compilation."""
    config = parse_training_args()
    trainer = TextTrainer(config)
    compiler = trainer.base.graph_compiler
    if compiler is None:
        raise ValueError("This verification entry requires compile.use_joint_graph")
    rank = int(os.environ.get("RANK", "0"))
    audit = {"rank": rank, "compilations": 0, "steps": []}
    original_compile = compiler.compile
    original_step = compiler.forward_backward

    @wraps(original_compile)
    def counted_compile(**inputs):
        audit["compilations"] += 1
        return original_compile(**inputs)

    @wraps(original_step)
    def measured_step(**inputs):
        shape = list(inputs["model_inputs"]["input_ids"].shape)
        try:
            loss, losses = original_step(**inputs)
        except ValueError:
            joint = compiler._joint_graph
            if joint is not None and joint.input_guards is not None:
                print(f"DYNAMIC_GRAPH_GUARDS rank={rank} {joint.input_guards.expression}", flush=True)
                tensors = [value for value in pytree.tree_leaves(inputs) if isinstance(value, torch.Tensor)]
                metadata = [(tuple(value.shape), value.stride(), value.storage_offset()) for value in tensors]
                print(f"DYNAMIC_GRAPH_INPUTS rank={rank} {metadata}", flush=True)
            raise
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Nonfinite loss for input shape {shape}")
        record = {"input_shape": shape, "loss": loss.item()}
        if compiler.specialization_stats:
            record["dispatch"] = compiler.specialization_stats["last_dispatch"]
        audit["steps"].append(record)
        print(f"DYNAMIC_GRAPH_STEP rank={rank} {json.dumps(record)}", flush=True)
        return loss, losses

    compiler.compile = counted_compile
    compiler.forward_backward = measured_step
    trainer.train()
    shapes = {tuple(step["input_shape"]) for step in audit["steps"]}
    audit["unique_shapes"] = sorted(shapes)
    audit["optimizer_steps"] = trainer.base.state.global_step
    audit["specialization"] = compiler.specialization_stats
    output = Path(os.environ.get("OUTPUT_DIR", "output/automodel_text_graph"))
    output.mkdir(parents=True, exist_ok=True)
    (output / f"dynamic_audit_rank{rank}.json").write_text(json.dumps(audit, indent=2))
    if audit["compilations"] != 1 or len(shapes) < 2 or audit["optimizer_steps"] != config.training.train_iters:
        raise RuntimeError(f"Dynamic graph verification failed: {audit}")
    if compiler.compile_sizes:
        stats = compiler.specialization_stats
        if not all(stats[key] > 0 for key in ("compilations", "cache_hits", "general_calls", "folded_nodes")):
            raise RuntimeError(f"Expected generated variants, cache hits and general fallback: {stats}")
    print(f"DYNAMIC_GRAPH_VERIFIED rank={rank} compilations=1 shapes={sorted(shapes)}", flush=True)


if __name__ == "__main__":
    main()
