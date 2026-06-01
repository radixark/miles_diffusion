# Ask Codex Input

## Question

你是一位资深的分布式训练/推理系统架构师，正在对一份实现草稿做首轮规划评审。请全程用**中文**回答。

## 仓库背景
miles_diffusion 是一个 diffusion GRPO 强化学习训练框架（从 LLM RL 框架 miles/slime fork 而来），训练后端用 FSDP2，rollout 用 sglang-diffusion（含 router）。目标：为未来的**视频** diffusion 训练加入**序列并行（SP）**，注意力算子采用 **USP**（Ulysses all-to-all + Ring 混合），并尽量与 sglang-diffusion 的算子精度对齐；若 SP 性能不达标，回退到 Mooncake（KVCache 为中心的解耦/卸载）方案。

## 已查证的关键代码事实（供你评审，无需重复核对）
- 在 DiT/diffusion 语境下 CP（Context Parallel）≡ SP（Sequence Parallel），都是沿 latent 序列维切分。现有 `(dp, cp)` 设备网格的 cp 维即 SP 维，但被 `miles/utils/arguments.py:1149` 的 `assert context_parallel_size == 1` 锁死。
- `miles/backends/training_utils/{loss,data,cp_utils,log_utils}.py` 是从 LLM RL 继承的死代码：仅彼此互相 import，无任何 diffusion 入口调用；带 causal/prompt/response/logits 假设。diffusion 训练在 `fsdp_utils/actor.py` 自带 loss（PPO-clip，`_forward_tile`/`advantage_window`）。
- `cp_utils.py` 的序列切分是 causal LLM 的 zigzag 负载均衡 2-chunk 切法；DiT 是双向注意力，不需要 zigzag。
- `patch_usp_attention.py` 已导入 sglang-diffusion 的 USPAttention（当前重定向到 SDPA 作对齐桩）；`sglang_diffusion_engine.py` 已转发 sp_degree/ulysses/ring degree。
- 用户确定采用 FSDP×SP **复合 device mesh**（当前 `apply_fsdp2` 只 wrap 在 mesh["dp"] 子网格）；训练拓扑为 **4+1**（4 卡训练 + 1 卡独占 reward），dp×sp 整除 4。
- 权重同步：训练侧 `diffusion_update_weight_utils.py` 用 `redistribute([Replicate()]*ndim).to_local()` 收全张量；`connect_rollout_engines` 有 rank 映射硬假设（`start_rank=i*rollout_num_gpus_per_engine`）；rollout 接收端 sglang 的 WeightsUpdater 对 sp_degree>1 的支持待确认；已有 `get_weights_checksum` 可作校验。

## 你的任务
对下面这份草稿做批判性评审，挑战其假设、找出缺失需求、提出更强方向。严格按以下 6 段格式、全程用中文输出（每段用列表）：

- `CORE_RISKS:` 最高风险的假设与潜在失败模式
- `MISSING_REQUIREMENTS:` 可能遗漏的需求或边界情况
- `TECHNICAL_GAPS:` 可行性或架构上的缺口
- `ALTERNATIVE_DIRECTIONS:` 可行的替代方案及其权衡
- `QUESTIONS_FOR_USER:` 需要人来决策的问题
- `CANDIDATE_CRITERIA:` 候选验收标准建议

## 待评审草稿（全文如下）

# 清理 LLM 血统的 CP 残留并为 diffusion 实现 FSDP + USP 序列并行

## 原始想法

我要给miles-diffusion未来视屏训练适配sequence parallellism，算子方面用usp，尽量可以和sglang-diffusion的算子精度对齐，另外要测试训练表现，如果不行的话要考虑mooncake的方案

## 主方向：先清理 LLM 血统的 CP 残留，再在干净地基上构建 FSDP + USP 序列并行

### 理由

经代码核查确认了两件关键事实，它们共同决定了本方向：(1) 在 DiT/diffusion 语境下 **CP（Context Parallelism）与 SP（Sequence Parallelism）是同一件事**——都是沿 latent token 序列维切分；现有 `(dp, cp)` 网格里的 `cp` 维本身就是序列并行维，只是被 `cp_size==1` 断言锁死。(2) 仓库里现存的 CP 代码绝大部分是从上游 miles/slime **LLM RL 训练继承来的死代码**，带着 causal/prompt/response 假设，且 **diffusion 训练路径根本不调用它们**。因此正确的做法不是"扩展现有 CP"，而是先**清理掉不被使用、且会误导后续实现的 LLM CP 残留**（前提是不影响当前纯 DP 训练），再在干净地基上为 diffusion 自有的训练路径实现 SP，注意力算子采用 USP。

### 方案概述

分阶段推进，前一阶段是后一阶段的地基：

**阶段 0 — 清理 LLM 血统的 CP 残留（不影响现有训练）。**
经核查，`miles/backends/training_utils/{loss,data,cp_utils,log_utils}.py` 这套是 LLM RL 训练代码：它们只彼此互相 import，没有任何 diffusion actor / `train_diffusion.py` 入口链 / flow_grpo 引用。diffusion 训练在 `fsdp_utils/actor.py` 里**自带** loss/advantage 实现（`actor.py:685-688` 的 PPO-clip + `_forward_tile`），完全不碰这套。清理动作：删除或隔离这套无活引用的 LLM 套件（含 `cp_utils.py` 全部 5 个 causal 函数），保留 `parallel.py` 的网格机制与 `ParallelState` 的 cp 槽位。验收口径：当前纯 DP（`cp_size=1`）训练逐位不变。

**阶段 1 — 解锁 SP 维并建立 FSDP×SP 复合 mesh。**
解除 `miles/utils/arguments.py:1149` 的 `assert context_parallel_size == 1`。以现有 2D `(dp, cp)` 网格为起点（`cp` 维即 SP 维，`mesh.get_group("cp")` 即未来 `sp_group`，可正名为 `sp`），但关键改动是**让 FSDP 感知整个复合 mesh**：当前 `actor.py:97` 的 `apply_fsdp2` 只 wrap 在 `mesh["dp"]` 子网格上，需扩展为在 FSDP 参数分片维 × SP 序列维组成的复合 device mesh 上 wrap（若同时要 HSDP 的 replicate×shard，则为 (replicate, shard, sp) 的更高维 mesh）。这样参数沿 FSDP 维分片、沿 SP 维复制，序列沿 SP 维切分，三者在一个 mesh 内协同。

**阶段 2 — USP 注意力算子。**
在 diffusion DiT 的注意力调用点接入 USP（Ulysses all-to-all 按头切 + Ring 按序列切的混合），替换当前 `parallel.py:34-37` 仅在 `cp_size>1` 才挂、且依赖与 transformers>=5.4 不兼容的 `ring_flash_attn`（纯 Ring）。复用 rollout 侧已有的 `patch_usp_attention.py`（已导入 sglang-diffusion 的 `USPAttention`，当前是对齐桩），把它升级为真正的 USP 语义，使训练与 rollout 共用一份算子定义，服务"算子精度和 sglang-diffusion 对齐"。

**阶段 3 — 让 diffusion 自有训练逻辑感知 SP。**
让 actor 自己的 `_forward_tile`、advantage/loss、log_prob 在序列被切分后仍正确：序列维切分、跨 `sp_group` 的损失/优势归约、以及 SP 组内一致的噪声 RNG。**这里需要重写一套 diffusion-native 的 SP 工具，而不是复用 causal 的 `cp_utils`**（理由见下）。

**阶段 4 — 权重同步（train→sglang）双侧适配。**
SP 不分片参数（每个 SP rank 持完整的 FSDP 分片副本），需分三层处理：(a) **训练侧打包（因复合 mesh 而必须适配）**——`update_weights` 用 `param.redistribute([Replicate()]*device_mesh.ndim).to_local()` 收全张量。由于阶段 1 确定走 FSDP×SP 复合 mesh，参数 DTensor 的 `device_mesh` 不再是 1D：`[Replicate()]*ndim` 会在所有维度（含 SP 维）上做 all-gather，需复核它在复合 mesh 下产出的张量正确、且不会因 SP 维本就复制而做无谓通信；同时多个 SP rank 会各自持有并准备发送相同的完整参数，须去重以免冗余 IPC（见 (b)）。(b) **训练↔rollout rank 映射**——`connect_rollout_engines` 硬编码 `start_rank=i*rollout_num_gpus_per_engine`、`tp_rank=rank-start_rank`，并假设 train actor 与 rollout engine 的 GPU id 相同；dp×sp 拓扑下 rank 排布改变，须重新对齐该映射，并解决"多个 SP rank 持相同参数是否冗余 IPC、如何选 gather src"。这是最可能需要改的点。(c) **rollout 接收端**——`sglang_diffusion_engine.py` 的 `update_weights_from_tensor` 只转发完整张量；若 rollout 也开 `sp_degree>1`，sglang-diffusion 的 WeightsUpdater 能否把完整权重正确分发到各 SP rank，需到 sglang 侧确认（本仓库之外）。可复用已有的 `get_weights_checksum` 做同步后的逐模块校验作回归门。

### 客观证据

- **CP≡SP、且 cp 切的是序列维**：`miles/backends/training_utils/cp_utils.py` 全程沿 `dim=0` 切 token（`qkv_format="thd"`），采用 zigzag 负载均衡 2-chunk 切法（`chunk_0` 前段 + `chunk_1` 对称后段，`cp_utils.py:32-33`）——这是 ring-attention 类 SP 的标准切分。
- **LLM 套件是死代码**：全仓库 grep 显示 `training_utils/{loss,data,cp_utils,log_utils}.py` 仅彼此互相 import（`data.py:18`、`log_utils.py:17-18`、`loss.py:23`），无任何 diffusion 入口引用。
- **diffusion 自带 loss，不走 LLM 套件**：`fsdp_utils/actor.py:301`（reward 广播到去噪步）、`:685-688`（`-advantage_tile*ratio` → PPO-clip → `per_cell_loss.mean()`）、`_forward_tile`/`advantage_window`/`tstep_indices` 全是 diffusion 自有实现；`loss.py:27-70` 则是纯 LLM（`get_responses(logits[1,T,V], tokens, response_lengths)`、`rollout_temperature`）。
- **diffusion 实际依赖的 CP 触点仅 `parallel.py`**：`actor.py:44` 调 `create_fsdp_parallel_state`；实际只用 `dp_mesh`(`:97`)、`dp_size`(`:48`)、`dp_cp_rank/dp_cp_size/dp_src_rank/dp_cp_group_gloo`(`:212-236` gather metrics)，**未使用 `cp_rank/cp_size/cp_group`**——当前 `cp_size=1` 时 `dp_cp_*` 即 world。
- **断言阻断**：`miles/utils/arguments.py:1149` `assert args.context_parallel_size == 1`（在 `validate_args` 中无条件执行）。
- **USP 基础已就位**：`miles/backends/sglang_diffusion_utils/monkey_patches/patch_usp_attention.py` 已导入 sglang-diffusion 的 `USPAttention`（当前重定向到 SDPA）；`sglang_diffusion_engine.py:319` 已通过 `ServerArgs` 转发 `sp_degree`/ulysses/ring degree。
- **现有 SP 算子的脆弱性**：`parallel.py:32-33` 注释指明 `ring_flash_attn` 与 transformers>=5.4 在纯 DP 下不兼容——这正是用 USP 取代它的理由之一。
- **权重同步链路与拓扑假设**：训练侧 `diffusion_update_weight_utils.py:72-75` 用 `redistribute([Replicate()]*device_mesh.ndim).to_local()` 收全张量（对任意维 mesh 成立）；`:102-116` `connect_rollout_engines` 硬编码 `start_rank=i*rollout_num_gpus_per_engine`、`tp_rank=rank-start_rank` 并假设 train/rollout GPU id 相同；rollout 接收端 `sglang_diffusion_engine.py:235` `update_weights_from_tensor` 仅转发完整张量；`:261` 已有 `get_weights_checksum` 可作同步回归校验。

### 已知风险

- **"清理不破坏现有训练"需可证伪**：删除 LLM 套件前必须确认没有动态/字符串/反射式引用，并以纯 DP 训练逐位对照（loss/grad）作为回归门。`parallel.py` 与 `ParallelState` 的 cp 字段被 diffusion 间接使用（`dp_cp_*`），属保留项，不可一并删除。
- **`cp_utils` 不能直接复用**：其 zigzag 负载均衡是为补偿 **causal mask 下位置算量不均**而设计；DiT 是**双向注意力**，负载本就均衡，zigzag 只会引入无谓复杂度。加之 prompt/response/logits 的 `-1` 因果偏移对 diffusion 无意义。结论是**参考其切分/gather 思路、另写 diffusion-native 实现**，而非复用。
- **跨后端精度**：diffusers+FSDP（autocast/bf16）对 sglang-diffusion 内核，叠加 USP 集合通信的舍入差异，逐位对齐不免费，需专门验证（见 Alt-1）。
- **权重同步是隐藏的关键面**：SP 不分片参数，但 train↔rollout 的 rank 映射假设（`connect_rollout_engines`）会因 dp×sp 拓扑失配；多个 SP rank 持相同参数可能造成冗余 IPC；rollout 侧若开 `sp_degree>1`，sglang-diffusion 的 WeightsUpdater 是否支持权重分发需先确认（本仓库之外）。低估这一面会导致同步静默错误，须以 `get_weights_checksum` 把关。
- **USP 性能未必达标**：长视频序列下 Ulysses all-to-all + Ring 的通信开销可能压不过收益，需基准决定是否走回退（见 Alt-2、Alt-3）。

## 考虑过的备选方向

### Alt-1：对照 sglang-diffusion 的精度对齐验证框架
- 要点：验证优先——在 FSDP 训练路径与 sglang-diffusion rollout 路径同时捕获注意力/norm/RoPE 中间激活，按可配置 `atol`/`rtol`（余弦/MAE/最大绝对误差）做差分，复用已有轨迹相似度工具与逐算子内核测试，作为 USP 算子的验收门，直接落实"算子精度对齐"。
- 客观证据：
  - `miles/backends/sglang_diffusion_utils/monkey_patches/` 已有逐算子 patch（`USPAttention`/`RMSNorm`/`LayerNormScaleShift`/`QKNormRoPE`/`MulAdd`），证明算子失配真实存在且已逐个管理。
  - sglang `tools/compare_diffusion_trajectory_similarity.py`（约 500 行，逐时间步 MAE/MSE/余弦/PSNR + JSON 输出）可迁移到算子级钩子。
  - sglang `jit_kernel/tests/diffusion/test_qknorm_rope.py`（`ATOL=8e-2, RTOL=1e-2` 的 split/fused 对照范式）。
- 为何不作主方向：它验证正确性但不构建 SP 能力，是围绕主方向的质量门而非特性本身——但应尽早建好作为主方向的验收依据。

### Alt-2：Mooncake 解耦/卸载回退
- 要点：想法明确保留的回退（"如果不行的话要考虑mooncake的方案"）：以 KVCache 为中心的传输引擎解耦，把长视频激活分段到 GPU/锁页主机/远端 GPU 池，扩展现有 move-based 与部分卸载端点及解耦式 placement group，不依赖进程内 SP 集合通信。
- 客观证据：
  - `miles/ray/rollout.py:198-214` 的 `offload()`/`onload()` 调 `release_/resume_memory_occupation`（支持 tag）；`:14` 导入 `GPU_MEMORY_TYPE_{WEIGHTS,KV_CACHE,CUDA_GRAPH}`（内存池已分离）。
  - `miles/ray/placement_group.py:81-118` 解耦式 actor/rollout 放置（独立 GPU 池）。
  - `miles/backends/fsdp_utils/diffusion_update_weight_utils.py:92-157` 的 `FlattenedTensorBucket`+gloo gather（可扩展的分布式状态传输范式）。
  - `docs/developer_guide/batch_sizes_in_miles_d.md` 的 `tstep_microbatch`/`sde_window_size` 切分（支持逐帧/逐切片卸载粒度）。
- 为何不作主方向：仅有 API 级脚手架（未实现 KVCache 解耦），作为应急方案应仅在 Alt-3 基准显示 USP 不达标时启动。

### Alt-3：轻量 10 步 perf 对照闸（是否需要 Mooncake 的判据）
- 要点：不是大而全的扫描/自动调参，而是一个**轻量的 go/no-go 闸**——固定配方、各跑 **10 步 RL**，对照三档：**① 什么都不开（baseline）、② 纯 FSDP、③ FSDP+SP（USP）**，记录下表少量指标，判断 SP 是否达预期；达不到即触发 Alt-2（Mooncake）。
- **指标与阈值（初稿，待按真实硬件/模型校准）**：分三组——先过正确性护栏，再看容量（SP 存在的根本理由），最后看效率。
  | 组 | 指标 | 口径 | 初稿阈值（go） | 不达标含义 |
  |---|---|---|---|---|
  | 护栏 | 数值一致性 | 同种子同序列长度下，③ 的 10 步 loss / 优势 / grad-norm 对 ② 的相对偏差 | loss 逐步相对偏差 < 2%、grad-norm 同量级 | SP 实现有 bug，先修正确性，perf 数据无效 |
  | 护栏 | 训练可跑通 | ③ 跑满 10 步无 OOM / 无 NCCL 挂死 | 必须通过 | 拓扑或显存配置不对，需先调 |
  | 容量 | 最大序列长度 | 不 OOM 前提下 ③ vs ② 能支撑的最大 latent 序列长度（帧数×分辨率） | ③ ≥ ② 的 **2×**（sp=4 时，理想 4×、打折后 2×） | SP 没解决长序列问题 → 强烈倾向 Mooncake |
  | 容量 | 峰值显存/卡 | 固定同一较长序列下，③ vs ② 的激活相关峰值显存 | ③ 较 ② 下降 **≥ 40%** | 显存收益不足 → 评估 Mooncake 卸载 |
  | 效率 | 并行效率 | ③ 的有效吞吐(序列长×样本/wall-clock) ÷ (② 在可放下的较短序列下吞吐 × 理想线性) | **≥ 60%**（即 SP 开销 < 40%） | 通信太重 → 比较 Mooncake 是否更划算 |
  | 效率 | SP 通信占比 | USP 的 all-to-all + ring 通信耗时 ÷ 单步时间 | **< 30%** | 通信瓶颈，同上 |
  | 效率 | 单步/10 步 wall-clock、吞吐、MFU | 三档直接对照（参考量，不单独设闸） | 记录即可 | — |
- **go/no-go 判定**：护栏全过的前提下——容量两项达标且并行效率 ≥ 60% → **SP 可行，不上 Mooncake**；容量达标但效率 < 50% → SP 通信过重，**评估 Mooncake**；容量不达标（长序列提升 < 1.5× 或 显存下降 < 30%）→ SP 未解决根本问题，**强烈倾向 Mooncake**。
- **前置说明（拓扑）**：采用 **4+1** 划分——4 卡训练、1 卡独占 reward。dp×sp 须整除训练卡数 4，三档对照的并行配置为：① 什么都不开 = 4 卡 DDP（不分片参数、不切序列）；② 纯 FSDP = dp=4×sp=1（仅参数分片）；③ FSDP+SP = dp=2×sp=2 或 dp=1×sp=4（参数分片 + 序列切分）。三档统一用 4 卡，保证对照公平。
- 客观证据：
  - `miles/utils/timer.py` + `miles/utils/train_metric_utils.py`（`log_perf_data_raw()`，已接入训练循环边界），可直接产出单步/分段耗时。
  - `miles/utils/wandb_utils.py:151-171`（`perf/*` 绑定 `rollout/step`）便于三档对照可视化。注：若阶段 0 清理掉 LLM 套件里的 `log_utils.py`，perf 埋点需改挂到 diffusion actor 自有指标。
  - `scripts/run-diffusion-grpo-wan22-pickscore-8gpu.sh` + `wandb/` + `logs/`（现成基线配方与埋点产物，10 步短跑即可复用）。
- 为何不作主方向：它是主方向的放行/否决仪器，量化主方向而非交付主方向；但因为它直接决定要不要上 Mooncake，应在阶段 1-2 一落地就尽早跑起来。

### Alt-4：SP 序列切分的实现选择——复用残留 vs 新写 diffusion-native
- 要点：一个真实的设计岔路。选项 A：保留并改造 `cp_utils.py` 的 zigzag 2-chunk 切分/gather；选项 B：彻底重写一套面向 DiT 双向注意力的 SP 切分（连续均分即可，无需 zigzag，去掉 prompt/response/logits 概念）。
- 客观证据：
  - `cp_utils.py:26-33`（zigzag chunk 偏移）、`:55-122`（`get_sum_of_sample_mean` 的 prompt/response 切分）、`:179-221`（`slice_with_cp` 的 thd 切分）——均为 causal 假设。
  - diffusion 侧无 prompt/response/causal 结构（`actor.py` 的 advantage/loss 沿去噪步与样本维，非 token 因果维）。
- 为何不作主方向：这是主方向阶段 3 内部的实现决策，倾向选项 B（双向注意力下 zigzag 无收益），但保留作为显式权衡点供 gen-plan 定夺。

## 综合说明

主方向是一条有严格阶段依赖的单一路线，备选则环绕其布置：阶段 0 的清理是"地基净化"，必须先行且以纯 DP 逐位回归为验收门，否则 LLM 残留会持续误导 SP 实现（本次讨论中它就已造成"扩展现有 CP""3D 网格"等误判）；阶段 1-2（解锁 cp 维 + USP 算子）是核心交付，Alt-1 的精度框架应与之并行建好，作为"算子和 sglang-diffusion 对齐"的客观验收；阶段 3 必须与阶段 2 同批次落地，因为算子正确不等于梯度正确——序列分片下的 loss/优势归约与 RNG 一致性若不到位，训练数值就是错的，而这里要走 Alt-4 的选项 B（新写 diffusion-native 切分）而非复用 causal `cp_utils`。阶段 4（权重同步双侧适配）是容易被忽略却会"静默跑偏"的一面：SP 不分片参数，但既然确定走复合 mesh，训练侧打包的 `redistribute` 要适配多维 `device_mesh`，更大的坑在 train↔rollout 的 rank 映射、SP 维参数复制的冗余 IPC 去重、以及 rollout 侧 `sp_degree>1` 的接收分发，须以 `get_weights_checksum` 把关，并尽早确认 sglang-diffusion WeightsUpdater 的 SP 支持。验证侧由两件仪器收口：Alt-1 精度框架确认"算子和 sglang-diffusion 对齐"，Alt-3 的**轻量 10 步 perf 闸**对照 ①什么都不开 / ②纯 FSDP / ③FSDP+SP 三档，用少量关键指标（指标与阈值待讨论）判断 SP 是否达预期；达不到即触发 Alt-2（Mooncake）这一明确保留的安全出口——仅当 USP 的集合通信开销无法扩展到目标拓扑上的视频序列长度时才启动。一句话：阶段 0 净化地基，阶段 1-4 是 FSDP+USP 的交付主体（含权重同步），Alt-1 与 Alt-3 是精度与性能的放行仪器，Alt-2 是回退出口，Alt-4 是阶段 3 的实现选型。

## Configuration

- Model: gpt-5.5
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-06-01_08-47-38
- Tool: codex
