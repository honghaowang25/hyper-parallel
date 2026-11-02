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
"""Prepare local configuration and tokenizer for a graph-FSDP debug run."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from examples.training_demo.deepseek_v41.prepare_deepseek_v41_assets import prepare_deepseek_v41_assets
from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import build_deepseek_v41_validation_config


def prepare_debug_assets(
    model_dir: Path,
    output_dir: Path,
    *,
    num_hidden_layers: int = 40,
) -> None:
    """Derive a random-initialized debug configuration from local official assets.

    Args:
        model_dir: Official V4.1 config and tokenizer directory; weights are unused.
        output_dir: Destination for debug model, tokenizer and Engram assets.
        num_hidden_layers: Retained decoder prefix; 40 preserves the released depth.
    """
    model_dir = model_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_files = ("config.json", "tokenizer.json", "tokenizer_config.json")
    for name in source_files:
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"Missing local V4.1 asset: {model_dir / name}")
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    for name in source_files:
        shutil.copy2(model_dir / name, tokenizer_dir / name)
    engram_path = output_dir / "engram_d8.json"
    prepare_deepseek_v41_assets(model_dir, engram_path, bucket_base=64,
                               parameter_divisor=8, table_pad_multiple=2)
    config = build_deepseek_v41_validation_config(
        str(model_dir), str(engram_path), num_hidden_layers=num_hidden_layers,
        text_parameter_divisor=8, num_routed_experts=8,
        exercise_post_training_indexer=True, enable_candidate_pool=True,
        indexer_loss_coeff=0.001,
        router_aux_loss_coeff=1.0e-4, router_bias_update_rate=0.0,
    )
    # The native eager expert loop selects a data-dependent number of rows.
    # HF's existing batched backend retains learned routing and fixed tensor shapes.
    config._experts_implementation = "batched_mm"  # pylint: disable=protected-access
    config.save_pretrained(output_dir / "model")
    source = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["text_config"]
    cropped = config.to_dict()
    dimensions = ("hidden_size", "moe_intermediate_size", "num_attention_heads", "head_dim",
                  "q_lora_rank", "o_lora_rank", "o_groups", "index_n_heads", "index_head_dim",
                  "num_hidden_layers", "num_experts_per_tok", "vocab_size", "hc_mult")
    ledger = {
        "source_files_sha256": {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
                                for name in source_files},
        "dimensions": {name: {"released": source[name], "debug": cropped[name]} for name in dimensions},
        "routed_experts": {"released": source["n_routed_experts"], "debug": config.num_local_experts},
        "retained_roles": {
            "compress_ratios": config.v41_compress_ratios,
            "kv_source_layer_ids": config.v41_kv_source_layer_ids,
            "index_source_layer_ids": config.v41_index_source_layer_ids,
            "engram_layer_ids": config.v41_engram_layer_ids,
        },
        "omitted_roles": {name: [index for index in source[name] if index >= num_hidden_layers]
                          for name in ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids")},
        "feature_switches": {
            "all_features_profile": True,
            "exercise_post_training_indexer": True,
            "indexer_loss_coeff": config.v41_indexer_loss_coeff,
            "candidate_source_layer_id": config.v41_candidate_source_layer_id,
            "candidate_topk_blocks": config.v41_candidate_topk_blocks,
            "candidate_block_size": config.v41_candidate_block_size,
            "reindex_remap": getattr(config, "v41_validation_reindex_remap", None),
            "training_data": "external WikiText; sample count and boundaries are determined by the loader",
        },
        "excluded_features": ["vision", "DSpark/MTP", "expert bias updates", "pretrained weights"],
        "dtype": "float32",
    }
    (output_dir / "crop_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Prepare the explicitly selected local assets without downloading weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--num-hidden-layers", type=int, default=40)
    args = parser.parse_args()
    output_dir = args.output_dir or "output/automodel_text_graph/deepseek_v41_all_features_depth40"
    prepare_debug_assets(Path(args.model_dir), Path(output_dir),
                         num_hidden_layers=args.num_hidden_layers)


if __name__ == "__main__":
    main()
