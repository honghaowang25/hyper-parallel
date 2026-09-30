# HyperParallel Graph Mode

Graph-mode architecture for automatic parallelization with FSDP.

> **Note**: The `compile/` subpackage relies on a patched autograd engine for
> joint-graph capture (see Limitations). Its `torch.*` imports — including
> `torch.fx.experimental` and `torch._guards` — are therefore intentional and
> carry a `forbidden-backend-import` suppression in
> `.jenkins/check/config/filter_pylint.txt`.

## Core Concept

**User**: Write model code + parallel configuration
**Framework**: Graph capture → FSDP partitioning → Communication-compute overlap → Execution

## Architecture

### Layer 1: Configuration

```python
from hyper_parallel.compile import GraphParallelPlan, PassConfig

# Configure which modules to mark for FSDP
parallel_plan = GraphParallelPlan()
parallel_plan.fsdp_mark("tok_embeddings")
parallel_plan.fsdp_mark_pattern("layers.*")

# Parallel configuration
pass_config = PassConfig(enable_overlap=True)
```

### Layer 2: Graph Capture

`trace_model_graph` captures forward + backward into a joint FX graph:

- Parameters/buffers are static inputs (placeholders), not `get_attr` nodes
- `torch.autograd.grad` runs inside the traced function
- Uses `FakeTensorMode` + `make_fx` for symbolic tracing

### Layer 3: Pass Pipeline

```text
DeadCodeElimination → CanonicalizeGraph → FSDPPass → AutoOverlapPass
```

**FSDPPass**:

- Identifies FSDP parameter placeholders via GraphParallelPlan
- Inserts `all_gather` after each FSDP parameter (Shard → Replicate)
- Inserts `reduce_scatter` on gradient outputs (Replicate → Shard)
- Physically shards live model parameters (dim 0) so optimizer is FSDP-agnostic

**AutoOverlapPass**:

- Reorders `wait_tensor` nodes for communication-compute overlap

### Layer 4: Execution

```python
from hyper_parallel.compile import GraphTrainer

trainer = GraphTrainer(
    model=model,
    train_fn=train_fn,
    pass_config=pass_config,
    parallel_plan=parallel_plan,
)

# Compile on first batch, then run forward + backward + optimizer
trainer.train(dataloader, max_steps=100, log_interval=10)
```

The compiled graph is a `GraphModule` that:

- Takes sharded parameters as inputs
- Gathers them via AllGather at each step
- Computes forward + backward
- Scatters gradients via ReduceScatter
- Outputs loss + grads

## Usage

```python
from hyper_parallel.compile import GraphParallelPlan, GraphTrainer, PassConfig

# 1. Model
model = Llama3Model(config)

# 2. Parallel plan
parallel_plan = GraphParallelPlan()
parallel_plan.fsdp_mark_pattern("layers.*")

# 3. Trainer
trainer = GraphTrainer(
    model=model,
    train_fn=lambda m, x, y: m(x).loss(y),
    pass_config=PassConfig(enable_overlap=True),
    parallel_plan=parallel_plan,
)

# 4. Training
trainer.train(dataloader, max_steps=1000)
```

## Dynamic input shapes

Static tracing remains the default. Enable symbolic user inputs with one option:

```python
compiler = GraphCompiler(model=model, train_fn=train_fn, dynamic=True)
```

For precise control, pass `dynamic_arg_dims` to `GraphCompiler` or `GraphTrainer`:

```python
compiler = GraphCompiler(
    model=model,
    train_fn=lambda m, x, y: ((m(x) - y) ** 2).mean(),
    dynamic_arg_dims={"x": [0, 1], "y": [0, 1]},
)
# x: [batch, sequence, input_features]; y: [batch, sequence, output_features]
loss, loss_dict = compiler.forward_backward(x=x, y=y)
```

The first call captures one symbolic forward/backward graph. Later batches with
compatible batch/sequence lengths reuse it; gradients still accumulate into the
live parameters. `GraphTrainer` exposes the same options and owns optimizer steps.
See [the runnable CPU example](examples/dynamic_shapes.py).

For the existing **BaseTrainer + GraphCompiler** path, configure:

```yaml
compile:
  enabled: true
  use_joint_graph: true
  dynamic_arg_dims:
    model_inputs.input_ids: [0, 1]
    labels: [0, 1]
```

Add paths for other tensors whose sizes vary, such as
`model_inputs.attention_mask: [0, 1]`. Paths refer to the keyword arguments of
`train_fn`; BaseTrainer supplies `model_inputs` and `labels`. Use `dynamic: true`
instead of the mapping to automatically symbolize all user tensor dimensions.

- A mapping takes precedence over `dynamic`; unspecified dimensions stay static.
  An empty mapping selects no dynamic dimensions. Negative dimension indices and
  nested dictionary/list paths (`batch.items.0`) are supported. Registered pytree
  objects can use attribute paths. Invalid paths or dimensions raise `ValueError`.
- Parameters and buffers always keep concrete shapes, including with automatic
  input symbolization. No Dynamo marks are added to the caller's tensors.
- The tracer uses `ShapeEnv` and per-tensor symbolic contexts directly, because
  this path captures through `make_fx`, not Dynamo. Runtime checks enforce traced
  size/stride relationships, input metadata, tensor identity aliases and Python
  constants before executing the joint graph. Shape-dependent Python branches
  are guarded; they do not become arbitrary runtime control flow.
- Use a representative first batch with varying dimensions **greater than 1**.
  PyTorch specializes dimensions of size 0/1 and may constrain dimensions used
  by operators or Python control flow. Inputs outside these constraints raise
  an explanatory error. They are not automatically recompiled: FSDP compilation
  mutates live parameter shards, so blindly retracing would be unsafe.
- First-stage scope: symbolic FX execution, loss and parameter gradients,
  gradient accumulation and optimizer updates. CPU unit tests cover the
  BaseTrainer entry and FSDP graph rewriting. The two-rank Gloo test covers
  real FSDP collectives, gradient shards and optimizer updates with overlap
  both enabled and disabled. The [NPU text example](examples/automodel_text_graph/README.md)
  exercises dynamic token packing through TP2 + FSDP2 with Qwen3-0.6B. Python scalar arguments remain
  constants. Pipeline parallel with dynamic shapes is rejected before tracing.
  Hot-size specialization, compiled-kernel backends, CUDA/NPU graph capture,
  serialized graph caches and data-dependent output shapes are not added here.

Run the focused checks with:

```bash
python -m pytest tests/ut/compile/test_dynamic_shapes.py -q
python -m pytest tests/torch/compile/test_dynamic_shapes.py -q
```

The approach follows MagiCompiler's separation of dynamic and static dimensions
and reuse of a general symbolic graph, adapted to HyperParallel's joint-graph
tracer. It adds no runtime dependency on the separate MagiCompiler checkout.

## Key Design Decisions

1. **Static Inputs**: Parameters are graph inputs, not `get_attr`. This allows passes to split the graph by reshaping placeholders.

2. **Joint Graph**: Forward + backward captured together via `torch.autograd.grad` inside the traced function.

3. **FSDP-Agnostic Optimizer**: FSDPPass shards the live model's parameters in place. `model.parameters()` returns shards, so optimizer needs no FSDP awareness.

4. **Declarative Sharding**: GraphParallelPlan uses FQN patterns (`layers.*`) instead of imperative module wrapping.

## Limitations

- FSDP only (TP/EP/PP planned)
- Parameters must have dim 0 divisible by world_size
- Requires torch with patched autograd engine for joint-graph capture
