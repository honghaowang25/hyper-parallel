# Joint-graph text training with dynamic token batching

This example uses the regular TextTrainer + BaseTrainer + GraphCompiler path,
the online plaintext transform, DynamicBatchDataLoader, and packed dense attention.
It runs TP2 + FSDP2 on four NPUs. The model/tokenizer directory is supplied by the
caller; the text generator needs no downloaded dataset.

Run from the HyperParallel repository root in a torch + torch_npu environment:

```bash
python -m hyper_parallel.compile.examples.automodel_text_graph.prepare_dynamic_text

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 bash \
  hyper_parallel/compile/examples/automodel_text_graph/run.sh \
  hyper_parallel/compile/examples/automodel_text_graph/train_online_lm_graph_dynamic_tp2_fsdp2.yaml \
  --model.pretrained_model_name_or_path=/path/to/Qwen3-0.6B
```

The YAML sets `compile.dynamic: true`; no manual dimension annotations are needed.
The batcher selects a variable number of records within its token budget, then
the online collator packs them into `[1, packed_length]`. Thus the physical batch
dimension stays one while the sequence dimension changes. The generated data has
several lengths to exercise this behavior over six optimizer steps (two
microsteps each). Replace `dataset.data_path` to use a different plaintext source.

`run.sh` accepts a config followed by the existing CLI config overrides. Its
default is the existing indexed/static YAML. Launcher settings such as
`MASTER_PORT`, `NPROC_PER_NODE`, and `OUTPUT_DIR` may be overridden through the
environment. The default training entry remains `scripts/train_lm.py`.

## Audit actual graph reuse

For a checked smoke run, select the optional verification entry. It constructs
the same TextTrainer, runs its normal training loop, records the real compiler
inputs, and asserts that training completed, loss stayed finite, more than one
shape occurred, and each rank compiled exactly once:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR="$PWD/output/automodel_text_graph/dynamic_npu" \
TRAIN_ENTRY="$PWD/hyper_parallel/compile/examples/automodel_text_graph/verify_dynamic_training.py" \
bash hyper_parallel/compile/examples/automodel_text_graph/run.sh \
  hyper_parallel/compile/examples/automodel_text_graph/train_online_lm_graph_dynamic_tp2_fsdp2.yaml \
  --model.pretrained_model_name_or_path=/path/to/Qwen3-0.6B
```

The output directory contains the launcher log and `dynamic_audit_rank<N>.json`.
The JSON records per-microstep input shapes/losses, unique shapes, compilation
count and completed optimizer steps. Diagnostic input metadata is printed if a
shape guard rejects a batch; guards are never bypassed.

Two details are essential for this path:

- Current/step token counts are explicit graph inputs, so loss weighting follows
  each packed batch rather than reusing the capture batch's counts.
- Symbolic tracing uses PyTorch's decomposition of `aten.constant_pad_nd`.
  Native padding can specialize lengths on NPU; decomposition preserves the
  normal shifted-label causal cross-entropy computation.

The focused NPU regression compares causal cross-entropy loss and parameter
gradients with eager execution in float32 and bfloat16:

```bash
python -m pytest tests/torch/compile/test_dynamic_shapes.py::test_dynamic_causal_loss_npu -q
```

This is a short functional smoke test, not a throughput/convergence benchmark.
Dynamic PP, arbitrary shape-dependent Python branches and size-0/1 generalization
remain outside the supported path.

## Lazy concrete-size FX graphs

Add the following overrides to the audited command above (or uncomment them in
the YAML) to exercise lazy generation and dispatch:

```bash
  '--compile.compile_sizes=[116,118,124,128]' \
  --compile.compile_size_input=model_inputs.input_ids \
  --compile.compile_size_dim=1
```

The first execution uses the general graph. Later configured sizes generate a
variant once per complete input metadata signature, then reuse it. Other sizes
use the general graph. `compile.max_specializations` defaults to 8; new signatures
at capacity also use the general graph. With no `compile_sizes`, behavior is unchanged.

This prototype binds shape expressions and regenerates FX code from the graph
after parallel passes. It does not rerun model capture, shard parameters again,
execute collectives while generating a variant, or invoke Inductor/NPU kernel
compilation. Tensor values, including token counts and weights, stay live inputs.

Audit JSON now includes `specialization` counters and a per-step `dispatch` field.
When specialization is enabled, the audit also requires generated variants,
cache hits and at least one folded shape expression. Choose sizes that actually
recur with your dataset; a different tokenizer or dataset can change packed lengths.
