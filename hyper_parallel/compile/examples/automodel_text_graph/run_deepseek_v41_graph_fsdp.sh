#!/bin/bash
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
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
ASSET_DIR="${ASSET_DIR:-output/automodel_text_graph/deepseek_v41_all_features_depth40}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/train_deepseek_v41_graph_all_features_depth40_fsdp2.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OUTPUT_LOG="${OUTPUT_LOG:-${ASSET_DIR}/run_graph_fsdp.log}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"

for name in model/config.json tokenizer/tokenizer.json engram_d8.json online.jsonl; do
    if [[ ! -s "${ASSET_DIR}/${name}" ]]; then
        echo "Missing ${ASSET_DIR}/${name}. Run prepare_deepseek_v41_debug first." >&2
        exit 1
    fi
done
mkdir -p "$(dirname "${OUTPUT_LOG}")"
python -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT:-29743}" \
    scripts/train_lm.py "${CONFIG}" \
    --model.pretrained_model_name_or_path="${ASSET_DIR}/model" \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${ASSET_DIR}/tokenizer" \
    --dataset.data_path="${ASSET_DIR}/online.jsonl" \
    --fsdp_config.dp_shard_size="${NPROC_PER_NODE}" \
    --training.global_batch_size="${NPROC_PER_NODE}" \
    "$@" 2>&1 | tee "${OUTPUT_LOG}"
