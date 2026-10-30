# DeepSeek V4.1 静态图 FSDP 训练

本文是本次适配的最终说明，包含当前实现、唯一推荐训练配置、复现方法及验证边界。
开发目录为 `/home/whh/dsv4_adapt`，分支为 `dsv4_adapt`，运行容器为 `whh_titan`。

## 1. 最终交付与运行设置

适配代码及回归测试保留。训练相关文件只保留以下一组：

| 文件 | 用途 |
|---|---|
| `hyper_parallel/compile/examples/automodel_text_graph/prepare_deepseek_v41_debug.py` | 从本地官方 config/tokenizer 生成随机初始化 debug 模型、Engram 资产、packing 文本及裁减记录 |
| `hyper_parallel/compile/examples/automodel_text_graph/train_deepseek_v41_graph_all_features_depth40_fsdp2.yaml` | 唯一训练配置：40 层、Packing + Compressor + Full/Reindex + Indexer KL 全开、FSDP2 |
| `hyper_parallel/compile/examples/automodel_text_graph/run_deepseek_v41_graph_fsdp.sh` | 默认使用上述 YAML 和对应的 40 层全开资产，调用标准 `scripts/train_lm.py` |

历史文档、基础配置及四层配置仅保存在服务器本地，不进入当前提交。
模型资产、训练日志和诊断程序也只保存在本地 `output/`，不进入提交。

当前配置使用 FP32 随机初始化、AdamW，TP/CP/EP/PP 均为 1，FSDP 为 2，
全局 batch 为 2、每卡 micro batch 为 1。联合图编译开启，参数预取和梯度通信重叠关闭。
`attn_implementation: eager` 选择 attention 的可移植 reference 实现；训练仍由联合静态图执行。

## 2. 全开配置与裁减范围

| 功能 | 当前设置与运行含义 |
|---|---|
| Packing | 流式 `build_online_iterable_dataset` → `TokenBatchLoader` → `TextParallelBatch`；每个文档 16 tokens，`token_budget=32`，实际边界为 `[0,16,32]` |
| Packed metadata | `DeepseekV41Runtime` 自动从最终模型配置读取 `[0,1,2]`，每个真实 batch 在捕获/重放前准备；不再配置 `prepare_compress_ratios` |
| Compressor | 保留 KV source 层 2/8/14/20；使用原有 Compressor 模块与压缩比 |
| Indexer | 保留 index source 层 2/8/14/20/24/28/32/36；Full=20，Reindex=24/28/32/36 |
| 候选池 | 资产生成显式设置 `enable_candidate_pool=True` 和 `exercise_post_training_indexer=True`；候选块宽度 128、块大小 8 |
| Indexer KL | 系数 `0.001`，经现有辅助损失自动缩放机制向 Indexer 注入梯度；不改变前向 CE 标量的定义 |
| MoE | 8 个 routed experts，保持 learned top-6 路由，使用 HF 已有的 `batched_mm` 后端；router auxiliary coefficient 为 `0.0001` |
| 融合算子 | `use_fused_ops=False`，使用支持候选池的 reference 路径 |

保留原生 40 层深度及层角色位置。主要参数维度按 8 裁减：hidden size 5120→640、
MoE intermediate 2304→288、attention heads 64→8、Q rank 1280→160、O rank 1024→128、
Indexer heads 32→4。head dim=512、Indexer head dim=128、rotary dim=64、o_groups=8、
hc_mult=4、sinkhorn iterations=20、词表 129280 保持不变。
Engram head dim 256→32，bucket base 64，表行数按 FSDP2 对齐；使用官方 tokenizer 归一化。
详细尺寸、源文件哈希和层角色写入资产目录中的 `crop_ledger.json`。

短序列下默认候选池宽于有效候选范围，Full/Reindex 确实执行，但不能据此声称发生了实际候选裁剪。
这是结构与功能穿刺配置，不能代表完整模型或长上下文性能。

## 3. 逐项代码修改及其作用

### 3.1 为 packed 对象注册 pytree

原 `SharedCompressedPackedSequence` 被 tracer 当作不支持的 Python 叶子，报
`trace_model_graph requires all pytree leaves ... to be tensors or primitives`。
现在 flatten/unflatten 将 `cu_seq_lens`、segment starts、minima、positions、masks 作为 Tensor 子树传入，
local/global 几何和各 ratio 的文档对齐布尔值作为静态 context。
重建对象恢复缓存与 owned snapshot，不在捕获阶段读取 Tensor 数值。

边界及派生 Tensor 随每批输入变化，避免把第一批边界固定为常量。
同样几何和对齐状态下 `[0,64,128] → [0,32,128]` 可复用图；
对齐状态变化由已有 pytree guard 拒绝，避免执行捕获时的错误分支。
文档数量、local/global 几何仍为静态契约，未宣称任意 packing 几何都支持动态图复用。

### 3.2 在捕获前自动准备 metadata

原模型在 forward 内懒准备 packed metadata，其中包含 `int(boundaries[0/-1])`、
Tensor 布尔结果参与 Python 判断、以及读取 host boundaries 等操作。
FakeTensor 捕获只有形状/类型信息，无法执行这些依赖真实数值的操作。

`RuntimeInputAdapter` 新增默认 no-op 的 `bind_model_config()` 生命周期接口；
Trainer 挂接数据管线时绑定最终 `model.config`；DeepSeek adapter 从 `v41_compress_ratios`
去重获取比例，每批提前调用原 `prepare()` 和 `segment_start_mask()`；
`EncoderParallelInputAdapter` 将绑定转发给 inner adapter。
这替代了早期手填 `prepare_compress_ratios` 的方案，eager 与 graph Trainer 使用相同入口。

输入合法时 mask/minima 与原懒准备一致；校验、分配与同步时机提前，异常可能更早出现。
未绑定的独立 eager adapter 仍保留模型内懒准备。绕过 Trainer 的独立图入口须绑定 adapter
并调用 build 流程，或传入已经准备的 packed 对象。

### 3.3 保留新分支的自然文档压缩

当前分支已允许自然长度文档，不能恢复旧分支“所有边界必须整除 ratio”的限制。
新增 `compression_aligned()` 在真实准备阶段缓存对齐布尔值，`block_plan()` 和候选池检查复用它。
未对齐文档的 eager block plan 继续采用原计算；candidate-pool 仍保留其原有对齐限制。

`indexer_segments()` 和上游新增的 `raw_document_starts()` 仅在 fused 路径读取。
reference 路径消费 Tensor masks，无须在捕获阶段把边界转成 Python intervals。
没有改变压缩分组、attention mask 或 RoPE 的数学定义。

### 3.4 短序列分块保持符号终点

原 `for start in range(0, S, chunk_size)` 需要将符号长度 S 转为具体整数。
例如以 S=16 捕获后会得到 `size()[1] == 16` 等 guard，随后 S=32 输入被拒绝：
`Graph input shape/stride violates the traced shape guards`。

改为在 `S <= chunk_size` 时直接返回 `((0,S),)`，保留符号 slice 终点；
S=0 返回空 tuple；更长序列仍使用原分块顺序。对非负长度和合法 chunk size，分块覆盖与原 range 一致。
S=16、32 都处于单块区间，可在满足其他 guard 时复用图；这不意味着能跨单块/多块分支任意复用。
Top-k、candidate 和 KL 的打分、mask 与梯度公式保持不变。
新增回归覆盖 S=0/16/256/300/600；未宣称上层训练算子支持空序列。

### 3.5 KL 使用 Tensor 控制流

原 KL 实现通过 `if not torch.any(valid_rows): continue` 和 `row_loss[valid_rows]` 处理有效行，
分别触发依赖 Tensor 值的 Python 控制流和动态长度索引。
现在移除 Python 分支，采用 `row_loss.masked_fill(~valid_rows,0.0).sum()`。

保留原 B×S 分母、损失系数、teacher no-grad 及解析 backward。
独立 autograd oracle 覆盖混合有效行与全无效行，确认损失和梯度一致且有限；全无效行贡献为零。

### 3.6 Router metric 从联合图显式输出

原监控属性把各层 auxiliary loss 转为 Python 数值。图重放不会重新执行 Python forward 的属性赋值，
从属性读取可能得到捕获阶段的 FakeTensor 或旧值。
新增 `last_mean_aux_loss_tensor`，使用 detached Tensor 均值；Trainer 将其写入
`loss_dict["router_aux_loss"]`，日志优先读取显式图输出，原 eager 属性作为 fallback 保留。

此改动仅影响监控，不修改训练目标。当前配置每次更新只有一个 micro-step；
多 micro-step 时 loss_dict 聚合与旧“最后 micro-step 属性”口径可能不同，未单独验收其日志语义。

### 3.7 示例配置补齐契约，复用已有实现

mHC 的 fn/base/scale、输入输出，以及 Engram 的 q/k/embed/wkv 参数通过 YAML 显式声明契约，
避免最终规划中存在未覆盖参数。TP replicate 不代表免于 FSDP；已有 pass 负责 scale 的奇数维 padding。
使用 HF `batched_mm` 保留 learned routing，避免 eager expert 的数据相关 Python 行选择。
该后端选择用于 debug 图穿刺，不代表其为完整模型的最优性能实现。

未修改图编译器、tracer、FSDP pass、AdamW 或 Transformers 安装包。
AutoModel 继续尊重 YAML 显式设置的 `load_base_model: false`，没有全局强制覆盖。
通信重叠关闭是复现已有 40 层 enqueue/sigmoid 停滞问题的配置规避，未宣称修复该问题。

## 4. 复现最终配置

先准备本地官方 `config.json`、`tokenizer.json`、`tokenizer_config.json`，不需要下载权重。
`MODEL_DIR` 指向这些文件所在目录；当前服务器的源资产副本位于
`output/deepseek_v41_migration_20261030/official_assets`。

```bash
cd /home/whh/dsv4_adapt
python -m pip install --no-deps --no-build-isolation -e .
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'
MODEL_DIR=output/deepseek_v41_migration_20261030/official_assets
python -m hyper_parallel.compile.examples.automodel_text_graph.prepare_deepseek_v41_debug \
  --model-dir "${MODEL_DIR}" \
  --output-dir output/automodel_text_graph/deepseek_v41_all_features_depth40 \
  --num-hidden-layers 40 --sequence-length 16 --all-features
bash hyper_parallel/compile/examples/automodel_text_graph/run_deepseek_v41_graph_fsdp.sh
```

启动脚本默认使用 NPU 0/1、两进程和上述唯一 YAML/资产目录，并检查必要文件。
可通过 `ASCEND_RT_VISIBLE_DEVICES`、`MASTER_PORT`、`OUTPUT_LOG` 覆盖运行设置；不会自动下载资源。
改变模型深度、文本长度或 token budget 时，需同步资产与配置，不能继续引用旧资产推断开关已启用。

## 5. 验证结果与边界

迁移后的全开观测验证：40 层两 rank 各三步，loss 为 11.65103 → 10.70528 → 5.85692，
峰值约 19.49 GiB/卡。每 rank 的 35 个 Indexer/Compressor 参数张量在三步中均有非零有限梯度和更新；
Compressor、Full/Reindex、KL 调用均被捕获，每 rank 三步复用同一联合图。

固定形状、改变 packing 边界的两 rank 对照覆盖 104 个参数：
最大梯度绝对误差 `6.70552e-8`、AdamW 权重绝对误差 `2.90386e-6`，满足原组合容差。
梯度使用 `rtol=2e-3`，Indexer `atol=1e-10`、其他参数 `atol=2e-5`；
权重使用 `rtol=2e-4, atol=2e-6`，不能只将最大绝对误差与 atol 单独比较。
扩展 CPU 回归 163 项通过，包含新分支已有的自然文档压缩回归。

CPU 回归仅在隔离进程恢复 CPU Tensor 的原生 SDPA，以隔离已安装 torch_npu 的全局 NPU SDPA patch；
NPU 训练和数值对照不使用该处理。AutoGit 的旧 C9002 禁止 Torch 导入，与当前 Torch-only 规范冲突，
原始失败日志保留；有效 pylint 检查仅在检查进程关闭该规则，没有修改仓库 checker。
文档 catalog、链接、shell 语法及 whitespace 检查独立完成。

最终默认入口另行复跑 40 层全开 FSDP2 三步通过，loss 为 11.651 → 10.7053 → 5.85692，
峰值 19.4863 GiB/卡。完整日志写入服务器 `output/deepseek_v41_final_20261030/`。
原完整日志、逐 rank JSON、数值对照与历史材料位于
`output/deepseek_v41_migration_20261030/`；历史文档和配置另有最终整理前的备份。

验收仅覆盖 FP32、随机初始化、text-only、FSDP2、小序列功能。
未验收 BF16、完整权重、视觉、其他并行拓扑、通信重叠、长上下文，
也未验收未对齐自然文档的联合图压缩；自然文档 eager 功能由已有回归覆盖。
