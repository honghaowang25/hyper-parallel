# DeepSeek V4.1：标准 WikiText 静态图 FSDP 训练

本文说明合并提交相对父版本 e0cdf62c 的全部适配修改、运行方式和验证边界。
本次将原 e168612f 与后续 WikiText 修复整理为一个提交，未引入本地后续的多 chunk、混合并行提交。
实际训练使用容器 whh_titan，整理工作区为 /home/whh/dsv4_e168_wikitext。

## 1. 当前结论

40 层 FP32 debug crop，Indexer、Compressor、Packing、Indexer KL 与后段候选池全开，
标准 WikiText FSDP2 在 compile.dynamic=true、无固定 padding 下完成 30 个 optimizer step 并正常退出。
这证明本轮短序列功能可运行，不代表全 epoch、任意动态长度或完整 FSDP 数值精度已经验收。

所有训练使用 scripts/train_lm.py、TextTrainer/BaseTrainer、原生 mapping Dataset、PlaintextTransform、
TokenBatchLoader、TextParallelBatch 和模型 RuntimeInputAdapter。没有手动构造训练 batch 或调整 packing 边界。

## 2. 唯一配置、启动与模型资产

| 文件（均位于 hyper_parallel/compile/examples/automodel_text_graph） | 用途 |
| --- | --- |
| prepare_deepseek_v41_debug.py | 仅准备模型 config、tokenizer、Engram 与 crop ledger，不生成训练文本 |
| train_deepseek_v41_graph_all_features_depth40_fsdp2.yaml | WikiText、自然 packing、dynamic joint graph、FSDP2，默认 30 步 |
| run_deepseek_v41_graph_fsdp.sh | 检查本地资产与 WikiText 文件，调用标准 Trainer；不下载数据或权重 |

模型资产准备需要本地官方 config.json、tokenizer.json、tokenizer_config.json：

```bash
python -m hyper_parallel.compile.examples.automodel_text_graph.prepare_deepseek_v41_debug \
  --model-dir /path/to/official_assets \
  --output-dir output/automodel_text_graph/deepseek_v41_all_features_depth40
```

默认保留 40 层及原层角色：KV source 2/8/14/20，Indexer source 2/8/14/20/24/28/32/36，
候选池由第 20 层发布。ratio=2 的前段覆盖自然文档压缩，后段 candidate 使用 ratio=1。
主要裁剪：hidden 5120→640、MoE intermediate 2304→288、attention heads 64→8、
Q rank 1280→160、O rank 1024→128、Indexer heads 32→4，routed experts=8、top-6。
head dim=512、Indexer head dim=128、rotary dim=64、o_groups=8、hc_mult=4、Sinkhorn=20、词表 129280 保持不变。
Engram bucket base=64、表行数按 FSDP2 对齐。完整尺寸与源文件哈希由 crop_ledger.json 记录。
这是自行从官方配置推导的 validation crop，不是官方发布的小模型规格。

准备好 WikiText-2-raw-v1 train Arrow 或 Parquet 后启动：

```bash
export PATH=/home/whh/dsv4_e168_wikitext_env/bin:$PATH
export WIKITEXT_TRAIN_FILE=/path/to/wikitext-train.arrow
export ASCEND_RT_VISIBLE_DEVICES=4,5
bash hyper_parallel/compile/examples/automodel_text_graph/run_deepseek_v41_graph_fsdp.sh
```

服务器现成缓存为：
`/home/whh/used/hyper_dev/.assets/datasets/Salesforce___wikitext/wikitext-2-raw-v1-a4b9c4ef0cfbce13/0.0.0/master/wikitext-train.arrow`。
该缓存 36,718 条 text 已与官方原始 Parquet 逐条比较一致；数据身份记录在实验目录。
YAML 使用可替换的相对数据路径，不把服务器缓存绝对路径写成通用默认值。

默认每卡 micro batch=1、global batch=2、max_seq_len=32、token_budget=32、min_buffered_samples=200、
drop_ragged_tail=false、不固定 padding；边界和实际长度由原生 loader 自然产生。
Optimizer 为 AdamW，FP32 初始化，TP/CP/EP/PP=1，通信预取/重叠关闭，使用 reference DSA。
attn_implementation=eager 选择 attention 实现，compile.enabled=true 仍执行联合静态图训练。
启动器支持 ASSET_DIR、CONFIG、NPROC_PER_NODE、MASTER_PORT、OUTPUT_LOG 与末尾标准 CLI overrides。
默认两进程经过验证；更改卡数不代表其他拓扑已验收。

## 3. 提交中的代码修改

### 3.1 原生 RuntimeInputAdapter 生命周期

hyper_parallel/data/batching/runtime_input.py 新增默认 no-op 的 bind_model_config()。
Trainer 在数据管线挂接时绑定最终模型配置；EncoderParallelInputAdapter 转发给 inner adapter。
模型 adapter 从 v41_compress_ratios 自动推导准备比例，不要求用户配置 prepare_compress_ratios。
通用框架只负责可选生命周期，不包含 DeepSeek 模型分支。

### 3.2 Packed pytree 与捕获前的完整准备

共享 attention 组件注册 SharedCompressedPackedSequence 为 pytree，解决原不支持 Python leaf 的捕获错误。
原 e168 的 alignment 布尔值和 local/global 长度放在 context，标准 WikiText 第 7 步 alignment 改变时被拒绝；
非对齐文档的 block plan 还会在捕获中读取真实边界，触发 unhashable non-nested SymInt。

现在 CP1 的标准 adapter 在捕获前准备 starts、minima/maxima、positions、masks、
压缩行索引/块起点和 router 文档 ID，均作为 Tensor children。context 只保留模式标识，长度从 Tensor shape 推导。
原始文档数、边界值和 alignment 不再决定该图的树结构。
图恢复对象只消费准备好的几何；host boundaries 不可在回放中重新读取。
独立 eager 对象及 CP 路径保留原 owned snapshot、版本失效与 host 契约。

### 3.3 固定容量压缩 plan

图采用统一的 Tensor pooling plan，容量 floor(S/ratio)。有效完整块构成紧凑前缀，
剩余槽位通过 query 的 key bounds 排除，不参与注意力选点或 KL target。
同文档分组、FP32 softmax pooling、舍弃每篇文档不完整尾块、原块起点 RoPE 的数学定义保持不变。
eager 保留原紧凑 bank 与 aligned 快路径。临时 shape 可以不同，不能宣称计算开销为零。

### 3.4 修正自然文档的 causal key 上界

旧 reference 只传 lower bound，上界仍取 floor((global_position+1)/ratio)，
在非对齐文档后可能看到尚未完成的压缩块。现在传入 Tensor 上界：
key_base + floor((position-document_start+1)/ratio)。

例如边界 [0,3,8]、ratio=2，位置 3 是第二篇文档首 token。旧 lower=1、upper=2 会允许 key 1；
正确 upper=1，此时尚无完整压缩块可见。此项有意修正非对齐文档的 eager/reference 行为；aligned 情况公式一致。
原 ratio>1 hierarchical candidate 的对齐限制保留，未靠移除限制获得成功。

### 3.5 Router 文档元数据

模型通过 router_metadata() 消费准备好的文档 ID，替代 forward 中读取原始 cu_seq_lens。
eager 使用实际文档数；图预留至多 S 篇文档容量，复用原 auxiliary loss 仅平均非空文档的定义。
独立回归检查 loss 和 gradient；额外容量对长序列内存的影响待评估。

### 3.6 Indexer 分块和 candidate padding

保留原 e168 的 _query_chunk_ranges()：单块直接返回 (0,S)，避免 range 将短序列 S 固定为捕获值。
超过 chunk_size 的 Python 分块分支未改成 while_loop，没有引入后续多 chunk 提交。

另修复 candidate block-max 的 zero-padding 特化：原 functional.pad 在补 0 时生成 K%8==0 guard，
真实 WikiText 开启 dynamic 后完成 8 步，第 9 步被拒绝。现以索引扩展加 -inf mask 构造相同数量的块，
保留 ceil(K/block_size)、输出形状、最后可达块 pinning 与 Top-K 规则。
真实 guarded joint graph 回归覆盖 32→31/30/29/32/33。

### 3.7 KL 与 router 日志

KL 原 Tensor 布尔判断和动态行索引改为固定形状 masked reduction，保留 B×S 分母、loss coefficient、
teacher no-grad 和解析 backward；独立 autograd oracle 覆盖混合有效行与全无效行。

MoE monitor 提供 detached Tensor metric，联合图将 router_aux_loss 显式输出，避免回放时读取旧 Python 属性或 FakeTensor。
logging callback 优先使用显式图输出，eager 属性作为 fallback。metric 在训练 objective 固定后加入，不改变训练目标。
多 micro-step 日志聚合口径仍需单独覆盖。

### 3.8 规划契约与测试

YAML 为 mHC fn/base/scale、输入输出，以及 Engram q/k/embed/wkv 补齐声明，复用已有 FSDP pass 的奇数维 padding。
HF batched_mm backend 保留 learned routing，避免原 eager expert loop 的数据相关行数。
没有改写通用 tracer、FSDP pass 或 AdamW，也没有强制 AutoModel 忽略 load_base_model。

保留原配置绑定、pytree/KL、graphtrainer、logging 测试，新增 9 项自然 packing CPU 回归。
CP1 Runtime 走 Tensor 几何；旧未准备对象的 alignment guard 作为兼容性保护保留，不把其受限路径当作真实数据验收。

## 4. 无用或过时修改的处理

移除资产脚本中的合成 JSONL、固定 16-token 对齐文本及其导入/CLI 参数，不再把“固定两篇文档”记为 packing 契约。
最终 YAML 和启动脚本改为 WikiText mapping Dataset、自然长度和 dynamic=true。
模型资产准备仍保留，因为随机初始化 crop、tokenizer 与 Engram 是可复现运行所必需。
原 e168 的 config 绑定、短序列符号终点、KL mask、图 metric 与规划契约仍被正式路径使用，保留。
旧实验配置、脚本、日志和初始 checkpoint 仅保留在服务器，不进入提交。

## 5. 已完成的验证

| 原生任务 | 步数 | 结果 |
| --- | ---: | --- |
| WikiText eager FSDP2，自然长度 | 30 | 正常退出 |
| WikiText eager FSDP2，pad_to_length=32 | 30 | 正常退出 |
| graph FSDP2，pad_to_length=32，dynamic 默认关闭 | 30 | 正常退出 |
| graph FSDP2，无固定 padding，dynamic=true | 30 | 正常退出；越过原 alignment 与 candidate guard |
| graph 单卡，自然长度，dynamic=true | 10 | 正常退出；原首步 SymInt 错误消失 |
| 相同初始权重、单卡、full determinism graph/eager | 各 3 | loss/router loss/norm 在日志精度下逐步一致 |
| 相同初始权重、FSDP2 graph/eager | 各 3 | 功能通过，完整数值精度尚未通过验收 |

共享初始 checkpoint 从同一 crop 的原生模型生成，966 张量完整加载、initialized=0、missing/unexpected=0。
它仅用于诊断，不是 optimizer/RNG/dataloader 全状态 continuation；实验程序不实现替代训练循环。

针对性测试 40 passed。扩展模型套件 165 passed、1 failed；MegaMoe 精确相等断言在另一个干净 e168 也失败，
未放宽断言或跳过测试。Pylint、catalog 和 whitespace 检查通过。
组件独立 oracle 验证 pooling、RoPE、可见性、attention loss/梯度、router 空容量及 candidate padding。

本次提交整理后，使用最终 YAML 和 run_deepseek_v41_graph_fsdp.sh 再跑 FSDP2、dynamic=true、
自然 WikiText 30 步，正常退出；相关 61 项单测通过，资产准备产物与已验证配置完全一致且不生成 JSONL。
最终改动的 Python Pylint、shell 语法、Markdown、catalog 与 whitespace 检查通过。
容器缺少 npm，autogit check 无法安装 markdownlint-cli2；Markdown 在本机使用仓库同一规则检查，
其余检查在容器完成，未将 autogit check 的环境失败写成通过。

运行记录位于 output/wikitext_fix_e168_20261102/；提交整理后的最终入口复验位于
output/wikitext_commit_20261102/。历史合成 fixture 的数值测试不作为本次真实 WikiText 验收证据。
一组 NPU 首个 HCCL AllGather 初始化 STUCK，换至已成功卡组后复验通过；环境底层原因未修复，设备未重置。

## 6. 精度和覆盖限制

完整 FSDP2 数值精度待定位：同初始 checkpoint 首步 graph/eager CE 约 11.9444/11.9363，
norm 13.8244/19.0632，full determinism 下仍存在。单卡打印值一致，双进程组件 loss 归约回归通过；
这些证据不等于完整模型精度证明。尚未记录逐 rank 完整输入 hash，也未证明首个不同的模型中间张量。
下一步需先核对输入/参数身份，再查数据派发、分布式归约、clipping 和更新。物理分片参数与 DTensor clipping helper 的衔接需核实。

当前最大 S=32、query_chunk_size=256，仅单 query chunk；Top-K/candidate 数量大于有效 key 数，不证明长序列稀疏收益。
未验收全 epoch、长期收敛、>256 多 chunk、跨滑窗 128、0/1 长度分支、BF16、完整权重、视觉、
EP/TP/SP/CP、混合并行、SAC/recompute、fused DSA 或通信重叠。
固定容量 bank、router 文档容量增加的长序列性能/显存开销待测；并发短跑不是严谨性能基准。
