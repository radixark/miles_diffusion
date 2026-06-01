Read and execute below with ultrathink

## Goal Tracker Setup (REQUIRED FIRST STEP)

Before starting implementation, you MUST initialize the Goal Tracker:

1. Read @/workspace/809a2940-8360-4812-81c2-c7383f3f43e7/miles_diffusion/.humanize/rlcr/2026-06-01_09-39-22/goal-tracker.md
2. If the "Ultimate Goal" section says "[To be extracted...]", extract a clear goal statement from the plan
3. If the "Acceptance Criteria" section says "[To be defined...]", define 3-7 specific, testable criteria
4. Populate the "Active Tasks" table with MAINLINE tasks from the plan, mapping each to an AC and filling Tag/Owner
5. Record any already-known side issues in either "Blocking Side Issues" or "Queued Side Issues"
6. Write the updated goal-tracker.md

## Round Contract Setup (REQUIRED BEFORE CODING)

Before starting implementation, create @/workspace/809a2940-8360-4812-81c2-c7383f3f43e7/miles_diffusion/.humanize/rlcr/2026-06-01_09-39-22/round-0-contract.md with:

1. **One mainline objective** for this round
2. **Target ACs** (1-2 ACs only)
3. **Blocking side issues in scope** for this round
4. **Queued side issues out of scope** for this round
5. **Round success criteria**

Use this contract to keep the round focused. Do NOT let non-blocking bugs or cleanup work replace the mainline objective.

**IMPORTANT**: The IMMUTABLE SECTION can only be modified in Round 0. After this round, it becomes read-only.

---

## Implementation Plan

For all tasks that need to be completed, please use the Task system (TaskCreate, TaskUpdate, TaskList).

Every task MUST start with exactly one lane tag:
- `[mainline]` for plan-derived work that directly advances the round objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` tasks are the primary success condition for the round
- `[blocking]` tasks may be resolved in the round only if they truly block mainline progress
- `[queued]` tasks must NOT become the round objective and do NOT need to be cleared before moving on
- If a new issue is not blocking the current objective, tag it `[queued]` and keep moving on the mainline

## Task Tag Routing (MUST FOLLOW)

Each task must have one routing tag from the plan: `coding` or `analyze`.

- Tag `coding`: Claude executes the task directly.
- Tag `analyze`: Claude must execute via `/humanize:ask-codex`, then integrate Codex output.
- Keep Goal Tracker "Active Tasks" columns **Tag** and **Owner** aligned with execution (`coding -> claude`, `analyze -> codex`).
- If a task has no explicit tag, default to `coding` (Claude executes directly).

# 为 miles_diffusion 视频训练实现 FSDP + USP 序列并行（含 LLM CP 残留清理）

## 目标描述（Goal Description）

为 miles_diffusion 的 diffusion GRPO 训练（FSDP2 后端，Wan2.2-T2V 视频 DiT）引入**序列并行（SP）**，注意力算子采用 **USP**（Ulysses all-to-all + Ring 混合），以支撑未来更长的视频序列训练。实现路径分五阶段：先**隔离**从 LLM RL 继承、当前 diffusion 训练不调用的 CP 死代码（不破坏现有纯 DP 训练，物理删除推迟到 SP 稳定后）；再解锁 SP 维并以 **Option B 复合并行**（FSDP 仍只在 dp 维分片参数、SP 用独立 process group、参数在 SP 维复制）搭建基础设施；通过**直接复用本地 sglang-d 的 USP 算子**（与 hao-ai-lab/FastVideo 同源、与 rollout 同一份代码）保证算子精度与 sglang-diffusion 对齐；让 diffusion 自有的 loss/advantage/log_prob 在序列分片后数值正确；并完成训练→rollout 的权重同步适配（rollout 本期保持非 SP）。最后用一个**轻量 10 步 RL perf 对照闸**判断 SP 是否达预期，据此决定是否回退到 Mooncake 方案。

与业界对齐：USP 实现对齐 sglang-d / FastVideo（Ulysses+Ring，layout `[B,S_local,H,D]`，`sp=ulysses×ring`）；复合并行形态对齐 USP/yunchang、Open-Sora 的"SP 独立 group + ZeRO/FSDP 正交"范式。Wan2.2-T2V-A14B 实测 `num_attention_heads=40`、`head_dim=128`、`num_layers=40`，满足 `sp=2/4` 的 Ulysses 整除约束。

## 验收标准（Acceptance Criteria）

- **AC-1：阶段0 隔离 LLM CP 残留，不破坏现有纯 DP 训练。**
  - Positive Tests：
    - `cp_size=1` 下，隔离前后用同一固定种子跑同一配方，loss / grad-norm / reward / 权重 checksum 在既定 tolerance 内一致（若当前栈非逐位确定性，则用 tolerance 而非逐位）。
    - 新增的"防引用测试"导入 `training_utils/{loss,data,cp_utils,log_utils}` 时直接失败（import-level fail test），证明这些模块已被标记 deprecated 且无活引用。
  - Negative Tests：
    - 任何现存 diffusion 训练入口（`train_diffusion.py` 链路、`fsdp_utils/actor.py`、flow_grpo）若仍 import 上述死代码，CI 必须报错。
    - 误删 `parallel.py` / `ParallelState` 的 cp 槽位（`dp_cp_*` 被 diffusion 实际使用）应导致现有训练启动失败——此项不允许发生。
  - AC-1.1：物理删除门控——在 AC-2~AC-6 全绿前，物理删除动作不得合入。

- **AC-2：解锁 SP 维并建立 Option B 复合并行（mesh + rank 映射明确）。**
  - Positive Tests：
    - 移除 `miles/utils/arguments.py` 中 `assert context_parallel_size==1` 后，`context_parallel_size>1`（正名 `sequence_parallel_size`）能正常初始化；启动日志打印 `dp/sp rank`、mesh 形状、`sp_group` 成员。
    - 在 `train_world=4` 下，`dp2×sp2` 与 `dp1×sp4` 两档都能给出确定的 rank→(dp_rank,sp_rank) 映射表，且参数在 sp 维 replicated、在 dp 维由 FSDP sharded。
    - **不假定卡数**：`dp × ulysses_degree × ring_degree = world` 的任意合法组合（从 2 卡到 1000+ 卡）均能初始化；`ulysses`/`ring` degree 可配置，`sp = ulysses × ring`。4 卡两档仅为当前验证点。
  - Negative Tests：
    - `num_heads % ulysses_degree != 0` 的配置在启动时被显式拒绝（Wan2.2 为 40，`ulysses∈{2,4,5,8}` 等整除值合法，`ulysses=3` 应报错）。
    - `dp × ulysses × ring != world` 的非法组合启动即报错。

- **AC-3：USP 算子复用 sglang-d，且训练侧 forward/backward/checkpoint/混精与非 SP 数值对齐。**
  - Positive Tests：
    - 单层 attention + 小 DiT block，固定 `(B,S,H=40,D=128)`、mask、RoPE、bf16，`sp>1` 的 forward 输出、输入梯度、QKV/proj 权重梯度对 `sp=1` 参考实现满足 dtype 对应 tolerance。
    - gradient checkpointing 开/关两种情况下 AC-3 的 parity 均通过。
  - Negative Tests：
    - 若训练侧误用 rollout 的 `patch_usp_attention.py`（bypass 成 SDPA 的 inference 桩），SP 下 attention 不做 all-to-all/ring，parity 测试必须失败并被捕获。
    - GQA/MQA 形态（若未来模型 head 不足）走 Ulysses 应被拒绝。

- **AC-4：序列切分契约 + RoPE 全局 position offset。**
  - Positive Tests：
    - 序列在 patchify 之后、attention 之前切分；每个 sp_rank 的 RoPE 位置索引使用全局 token index（`offset = sp_rank * S_local`），各 rank 索引不相交且并起来覆盖 `[0, S_global)`。
    - `S_global % ulysses_degree == 0`（必要时 padding）得到验证。
  - Negative Tests：
    - 若各 rank 用 local `[0, S_local)` 生成 RoPE（漏掉全局 offset），与非 SP 输出比对必须失败。

- **AC-5：SP 梯度同步协议（与 FSDP 次序明确，DDP parity）。**
  - Positive Tests：
    - 写明同步对象、通信 op、求和/平均系数、与 FSDP backward hook 的先后、optimizer.step 前的不变量；按此实现后，`dp2×sp2` 训练的参数更新与"全序列单卡/DDP 参考"在 tolerance 内一致。
  - Negative Tests：
    - 缺失 SP 梯度 all-reduce 时，不同 sp_rank 的参数在若干 step 后发散（用该现象作为反向探测，必须被 parity 门拦截）。

- **AC-6：diffusion 自有 loss/advantage/log_prob 的 SP 归约逐项等价。**
  - Positive Tests：
    - 逐项给出 `log_prob`、`sde_log_prob`（对空间维 mean）、kl/entropy、advantage window、mask/count、clip ratio 的 SP 归约方式（`sum + global_count` 或等价加权 mean，**不得**各 rank local-mean 再平均），并证明与非 SP 版本在 tolerance 内等价。
    - 样本级 DP round-robin 分区在开启 SP 后保持不变（SP 不改变样本分区）。
  - Negative Tests：
    - 采用"各 rank local mean 再平均"的错误归约，与非 SP 比对必须失败。

- **AC-7：RNG 三级一致性。**
  - Positive Tests：
    - 区分样本级 / token级 / dropout-noise 级随机性；同一 SP 组内的不同 token shard **不复用**相同随机数（明确 seed offset 或 generator 分区）；噪声/timestep 采样在 SP 组内可复现。
  - Negative Tests：
    - 不同 token shard 误用同一随机数、或 SP 组间 RNG 不一致导致同一样本噪声错位，均应被一致性测试捕获。

- **AC-8：权重同步——单代表 rank 发送 + checksum 覆盖语义（rollout 本期非 SP）。**
  - Positive Tests：
    - 每个 SP replica group 仅由一个代表 rank 向 rollout 发送权重（或发送前去重），无冗余 IPC；`get_weights_checksum` 校验覆盖 dtype、shape、shard/replica 语义，逐模块一致。
    - rollout 保持 `sp_degree=1`，权重同步在 `dp2×sp2`/`dp1×sp4` 训练拓扑下均通过 checksum。
  - Negative Tests：
    - 多个 sp_rank 各自发送相同完整参数（冗余 IPC）应被去重逻辑消除；checksum 只比对 bytes 而忽略 dtype/shape 语义的实现不被接受。

- **AC-9：10 步 RL perf 对照闸 + 通信分解 + go/no-go。**
  - Positive Tests：
    - 三档 ①4卡 DDP ②纯 FSDP(dp4) ③FSDP+SP(dp2×sp2 与 dp1×sp4)，各跑 10 步，产出指标：护栏（loss/grad 对 baseline 偏差<2%）、容量（最大序列长度 ③≥②2×、峰值显存↓≥40%）、效率（并行效率≥60%、SP 通信占步时<30%）。
    - perf 报告含**通信分解**：USP all-to-all/ring、FSDP reduce-scatter/all-gather、weight sync、rollout 等待四类分别计时。
    - `dp1×sp4` 结果明确标注为"容量测试"（DP=1 时 FSDP 分片收益退化，不作效率结论），效率结论以 `dp2×sp2` 为准。
  - Negative Tests：
    - 容量提升<1.5× 或显存下降<30% → 判定 SP 未解决根本问题，触发 Mooncake 专项评估（先做瓶颈归因，不直接堆 SP）。

## 路径边界（Path Boundaries）

### 上界（最大可接受范围）
完整实现五阶段：阶段0 隔离+防引用测试+纯 DP 回归门；阶段1 Option B 复合并行（dp×sp，两档拓扑）；阶段2 复用 sglang-d USP 算子并接入 diffusers，带 forward/backward/checkpoint/混精 parity 门；阶段3 全部 SP-aware loss 归约 + RNG 三级一致；阶段4 单代表 rank 权重同步 + checksum 覆盖语义；并交付 10 步 perf 闸（三档 + 通信分解 + go/no-go 报告）。物理删除 LLM 死代码在 SP 稳定后单独合入。

### 下界（最小可接受范围）
训练侧 SP 在 `dp2×sp2` 跑通：解锁 SP 维 + Option B + 复用 sglang-d USP 算子 + 核心 loss/log_prob 的 SP 归约正确（AC-3/4/5/6 的 parity 在 `dp2×sp2` 通过）+ 权重同步 checksum 通过（rollout 非 SP）+ 一份至少覆盖 `dp2×sp2` 的 perf 对照。LLM 死代码本期可仅"隔离+标记 deprecated"，不强制物理删除。

### 允许的选择（Allowed Choices）
- Can use：本地 sglang-d 的 USPAttention / `set_seq_parallel_pg` / `SequenceParallelGroupCoordinator`（= FastVideo 同源）；PyTorch≥2.4 的 `_templated_ring_attention`；FlashAttention 后端；yunchang 风格的 ulysses/ring 子组；`get_weights_checksum` 作回归门。
- Cannot use：自研另一套 USP/attention 实现（会偏离与 sglang-d 的精度对齐）；让 FSDP 直接吞多维复合 mesh（牵动 reshard/optimizer/checkpoint，无业界先例，风险高）；阶段0 一上来物理删除 LLM 代码；各 rank local-mean 再平均的 loss 归约；**写死固定卡数/拓扑或只支持单一 ulysses/ring 组合（必须可配置、覆盖 2~1000+ 卡）**；**防御性编程与生产/热路径上的冗余校验、兜错 try/except**。

> 说明：本方案是较确定性的设计（复合并行采用 Option B、USP 复用 sglang-d 已定），故上述边界偏窄；perf 闸的指标阈值（2%/2×/40%/60%/30%）为初稿，允许按真实硬件/模型校准（属优化方向而非硬性失败线，唯 go/no-go 的容量下限 1.5×/30% 用于触发 Mooncake 评估）。

## 可行性提示与建议（Feasibility Hints and Suggestions）

> 仅供参考的概念性建议，非强制实现方式。

### 概念性方案（Conceptual Approach）
1. 阶段0：给 `training_utils/{loss,data,cp_utils,log_utils}` 加模块级 deprecation + import-fail 测试；用现有 `cp_size=1` 配方跑回归基线（loss/grad/checksum）。
2. 阶段1：`parallel.py` 把 `(dp, cp)` 正名/扩展出 `sp_group`；保持 `apply_fsdp2(mesh=mesh["dp"])` 不变；`ParallelState` 增 `sp_rank/sp_size/sp_group/ulysses_group/ring_group`。
3. 阶段2：在 `actor.py` 模型加载后，把 diffusers transformer 的 attention 调用导向 sglang-d 的 `USPAttention`（优先显式 AttnProcessor 注入，见 DEC-1）；调用 sglang-d 的 `set_seq_parallel_pg` 建 ulysses/ring 子组。
4. 阶段3：在 `_forward_tile` 进 attention 前切序列、出来后按契约处理；改 `sde_log_prob` 的空间维 reduce 为跨 `sp_group` 的 `sum+global_count`；RNG 按三级分区。
5. 阶段4：在 `diffusion_update_weight_utils.connect_rollout_engines` / `update_weights` 增 SP replica 去重，仅代表 rank 发送；`get_weights_checksum` 扩展 dtype/shape/replica 语义校验。
6. perf：复用 `timer.py` / wandb `perf/*`（注意阶段0 若隔离了 `log_utils.py`，perf 埋点改挂 actor 自有指标）；包一层三档 sweep + 通信分解。

### 相关参考（Relevant References）
- `miles/backends/fsdp_utils/parallel.py` — 设备网格与 `ParallelState` 构造（SP 维起点）
- `miles/backends/fsdp_utils/actor.py` — `apply_fsdp2`、`_forward_tile`、自有 PPO-clip loss/advantage
- `miles/utils/arguments.py` — `context_parallel_size==1` 断言
- `miles/backends/sglang_diffusion_utils/monkey_patches/patch_usp_attention.py` — rollout 的 USP 桩（训练侧不可直接复用）
- `/sgl-workspace/sglang/.../multimodal_gen/runtime/layers/attention/layer.py` — sglang-d `USPAttention`（复用源，FastVideo 同源）
- `/sgl-workspace/sglang/.../multimodal_gen/runtime/distributed/parallel_state.py` — `set_seq_parallel_pg` / SP 组初始化
- `/sgl-workspace/sglang/.../multimodal_gen/runtime/loader/weights_updater.py` — rollout 接收端（DTensor-aware）
- `miles/backends/fsdp_utils/diffusion_update_weight_utils.py` — 训练侧权重打包/同步、`connect_rollout_engines`、`get_weights_checksum`
- `miles/utils/sde_log_prob.py` — log_prob 空间维 reduce（SP 归约点）
- `miles/rollout/diffusion_rollout.py` — `_make_generators` RNG
- `models/Wan-AI/Wan2.2-T2V-A14B-Diffusers/transformer/config.json` — heads=40/head_dim=128/layers=40

## 依赖与顺序（Dependencies and Sequence）

### Milestones
1. **里程碑 A — 地基净化（阶段0）**：隔离 LLM 死代码 + 防引用测试 + 纯 DP 回归基线。无前置依赖。
2. **里程碑 B — SP 基础设施（阶段1）**：解锁断言、Option B 复合并行、两档 rank 映射。依赖 A 的回归基线作对照。
3. **里程碑 C — USP 算子接入（阶段2）**：复用 sglang-d USP，接入 diffusers，单层/小 block 的 forward/backward/checkpoint/混精 parity。依赖 B。
4. **里程碑 D — 训练逻辑 SP-aware（阶段3）**：loss/log_prob 逐项 SP 归约 + RNG 三级一致；与 C 同批次落地（算子对、梯度不对则训练仍错）。依赖 B、C。
5. **里程碑 E — 权重同步（阶段4）**：单代表 rank 去重 + checksum 覆盖语义；rollout 保持非 SP。依赖 B。
6. **里程碑 F — perf 闸与决策**：三档 10 步对照 + 通信分解 + go/no-go。依赖 C、D、E（至少 `dp2×sp2` 跑通）。F 的结论决定是否进入 Mooncake 专项评估。
**贯穿全程 — Living 实现报告**：`.humanize/reports/usp-video-sp-impl-report.md` 是一份**活文档**，从里程碑 A 起就随每次系统设计/代码改动/遇到的明显 bug 持续更新（不是收尾一次性生成），收尾时补齐 parity 与 perf 结论、遗留问题与后续。独立于 RLCR loop 每轮自带的 `round-N-summary.md`（后者仅为循环内 Codex review 用的进度总结）。

物理删除 LLM 死代码（阶段0 的删除动作）依赖 C/D/E 全绿，单独合入。

## 任务分解（Task Breakdown）

| Task ID | 描述 | 目标 AC | Tag | 依赖 |
|---------|------|---------|-----|------|
| task1 | 给 LLM training_utils 套件加 deprecation 标记与 import-fail 测试 | AC-1 | coding | - |
| task2 | 建立 cp_size=1 纯 DP 回归基线（loss/grad/checksum） | AC-1 | coding | - |
| task3 | 解除 context_parallel_size==1 断言，正名 sequence_parallel_size | AC-2 | coding | task2 |
| task4 | 实现 Option B 复合并行：sp_group、ParallelState sp 字段、两档 rank 映射 + 启动校验 | AC-2 | coding | task3 |
| task5 | 接入 sglang-d USPAttention 到 diffusers attention（注入方式见 DEC-1） | AC-3,AC-4 | coding | task4 |
| task6 | 单层 attention + 小 DiT block 的 SP-vs-非SP parity 测试（forward/backward/checkpoint/混精） | AC-3 | coding | task5 |
| task7 | 序列切分契约 + RoPE 全局 offset 接口与校验 | AC-4 | coding | task5 |
| task8 | SP 梯度同步协议实现（次序见 DEC-2）+ DDP parity | AC-5 | coding | task4,task5 |
| task9 | diffusion loss/log_prob/kl/advantage 的 SP 归约（逐项等价证明）+ sde_log_prob 空间维 | AC-6 | coding | task5,task8 |
| task10 | RNG 三级一致性（样本/token/noise）实现与测试 | AC-7 | coding | task5 |
| task11 | 权重同步：SP replica 单代表 rank 去重 + checksum 覆盖 dtype/shape/replica | AC-8 | coding | task4 |
| task12 | 10 步 perf 闸三档 + 通信分解埋点 + go/no-go 报告 | AC-9 | coding | task6,task9,task11 |
| task13 | 若 perf 不达标：Mooncake 瓶颈归因专项分析（attention/FSDP/activation/weight-sync/rollout 哪类主导） | AC-9 | analyze | task12 |
| task14 | 确认 sglang 的 sp_degree 是否真建 SP NCCL 组（为未来 rollout 开 SP 铺垫） | AC-8 | analyze | - |
| task15 | **全程维护** living 实现报告（`.humanize/reports/usp-video-sp-impl-report.md`）：从 task1 起每次系统设计/代码改动/遇到明显 bug 即更新（非收尾一次性生成），收尾补齐 parity 与 perf 结论 | 全部 | coding | 贯穿（task1 起） |

## Claude-Codex 评议（Claude-Codex Deliberation）

### 一致同意（Agreements）
- 复合并行采用 Option B（SP 独立 group + FSDP/ZeRO 正交），对齐 USP/yunchang/Open-Sora/sglang-d。
- rollout 本期保持非 SP，先证明训练侧 SP 正确，显著降低 sglang 侧风险。
- 阶段0 改为"隔离+防引用测试"，物理删除推迟到 SP 稳定后。
- Wan2.2 `num_heads=40` 支持 `sp=2/4`；`dp2×sp2` 与 `dp1×sp4` 两档都测，后者仅作容量测试。
- 权重同步真正风险在 SP rank 冗余发送、rank 映射、checksum 语义覆盖。

### 已解决的分歧（Resolved Disagreements）
- "让 FSDP 吞多维复合 mesh" vs "Option B 独立 group"：选 Option B，理由是避免牵动 reshard_after_forward/optimizer/checkpoint 的多维 DTensor 行为，且业界无吞多维 mesh 的先例。
- "训练侧复用 patch_usp_attention" vs "复用 sglang-d USPAttention 本体"：选后者；patch_usp_attention 是 rollout 的 inference 桩（bypass SDPA），训练侧需真正的 USP forward。
- "复用 cp_utils 的 zigzag 切分" vs "对齐 sglang-d/FastVideo 的切分"：选后者（causal zigzag 不适合 DiT 双向注意力）。
- Codex 的 8 项 REQUIRED_CHANGES（mesh/rank 表、SP 梯度协议、backward/checkpoint 验证门、RoPE 接口、loss 逐项归约、RNG 三级、权重去重+checksum 语义、perf 通信分解）已分别落入 AC-2~AC-9 与 task 列表。

### 收敛状态（Convergence Status）
- Final Status: `converged`（经 Codex 首轮分析 + 第二轮复审；技术方向无 high-impact 分歧，细化要求纳入 AC/Task，对立意见转入下方 Pending User Decisions）。

## 决策记录（Resolved Decisions）

以下决策已确认（2026-06-01）。

- **DEC-1：diffusers attention 接入 USP 的方式。**
  - Decision：`显式 AttnProcessor 注入`（工程边界清晰、可控）；monkey-patch SDPA 仅作兜底。
  - Tradeoff Summary：AttnProcessor 需适配 diffusers 各模型但边界清楚；monkey-patch 快而易随版本漂移。
  - Decision Status：`RESOLVED — AttnProcessor 注入`

- **DEC-2：SP 梯度同步发生在 FSDP hook 前还是后。**
  - Decision：`FSDP reduce-scatter 后对 shard-grad 同步`（与 FSDP 通信对象一致、避免重复 all-gather），以 DDP parity 验证（task8 契约固定）。
  - Tradeoff Summary：次序错会导致数值近似但 optimizer 状态长期漂移。
  - Decision Status：`RESOLVED — reduce-scatter 后 shard-grad 同步`

- **DEC-3：ulysses/ring 的组合策略——不假定训练卡数。**
  - Decision：**实现不得写死任何拓扑或卡数（须覆盖 2 卡至 1000+ 卡）**。`ulysses_degree` 与 `ring_degree` 全程可配置，`sp = ulysses_degree × ring_degree`。推荐默认策略：**Ulysses 用于节点内**（受 `num_heads % ulysses == 0` 限制、通信量恒定、走 NVLink，典型 `ulysses ≤ 单节点 GPU 数`），**Ring 用于跨节点扩展**（P2P、可任意扩展、overlap 计算与通信）；即 `ulysses = 满足整除约束的节点内最大值`、`ring = sp / ulysses`。当前 4 卡 perf 闸的 `dp2×sp2` / `dp1×sp4` 仅是**当前硬件的验证点**，非架构假设。
  - Tradeoff Summary：Ulysses 受 head 数与节点内带宽约束、不可无限扩；Ring 可扩到任意节点数但 backward 更复杂。二者组合是 USP 在大规模下的必备能力（对齐 sglang-d/FastVideo/yunchang），不能因当前小规模而砍掉 ring 路径或写死 degree。
  - Decision Status：`RESOLVED — ulysses×ring 全程可配，不假定卡数；ring 路径为必备能力`

- **DEC-4：Mooncake 的目标瓶颈定位。**
  - Decision：Mooncake **不预设为 SP 的整体替代**；先由 task13 做瓶颈归因——仅当瓶颈是 rollout/权重同步/跨节点通信时才考虑 Mooncake；若瓶颈是训练侧 attention/activation，则继续优化 SP 或走 activation offload。
  - Tradeoff Summary：Mooncake（KVCache 中心）更贴合自回归推理；DiT 训练压力多在 activation/optimizer/attention 通信，盲目上 Mooncake 可能解决错问题。
  - Decision Status：`RESOLVED — 先瓶颈归因再定`

## 实现注记（Implementation Notes）

### 代码风格要求（Code Style Requirements）
- 实现代码与注释中**不得**出现计划专用术语，如 "AC-"、"里程碑/Milestone"、"阶段/Phase"、"task1" 等流程标记；这些仅用于本计划文档。
- 代码中用领域贴切的命名（如 `sequence_parallel_size`、`sp_group`、`ulysses_degree`、`shard_sequence`、`gather_sequence_loss`、`all_reduce_sp_grad` 等）。
- 序列并行相关命名统一用 `sp`/`sequence_parallel`；`cp`/`context_parallel` 仅作历史维名保留，新代码不要继续传播 CP 术语（可加 `sp` alias 向后兼容）。

### 代码精简原则（Code Minimalism — 硬性要求）
- **极简优先**：在保证精度与速度的前提下，能删则删、能简则简；不写"以防万一"的代码。
- **禁止防御性编程**：不加冗余的输入校验、不加 try/except 兜底吞错、不为不可能的分支写处理。让错误自然抛出，定位更快。
- **断言仅用于 debug**：`assert` 只作开发期排查用，不作为生产路径的控制流；不在热路径（每 step / 每 tile / attention 内层）堆校验。
- **只在最必要处报警**：仅保留"配错即静默错误或挂死"的启动期合法性校验（如 `num_heads % ulysses`、`dp×ulysses×ring == world`、SP 梯度同步缺失探测）；其余不滥用 warning/log。
- **区分实现与验证**：本计划的 parity 测试、checksum、通信分解属于**测试/验收层**（可丰富），不等于在实现代码里堆防御逻辑——实现代码本身保持精简。

--- Original Design Draft Start ---

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

**阶段 1 — 解锁 SP 维并建立 FSDP×SP 复合并行（对齐业界 = SP 独立 group + FSDP 正交）。**
解除 `miles/utils/arguments.py:1149` 的 `assert context_parallel_size == 1`。复合并行采用**业界一致的形态**：FSDP 仍只 wrap 在 shard 子 mesh（现有 `mesh["dp"]`，`actor.py:97` 不变），**SP 用独立的 process group**（复用 `mesh.get_group("cp")`，正名为 `sp_group`），参数在 SP 维**自然复制**、不进 FSDP 的 sharding。这一选择有三方背书：USP/yunchang 明确"SP 必须配合 ZeRO-1/2"、Open-Sora 用"Ulysses + ZeRO hybrid"、sglang-d/FastVideo 用独立的 `SequenceParallelGroupCoordinator`（ulysses_group + ring_group）而非把 SP 塞进 FSDP mesh。代价是 **SP 维的参数梯度需手动 all-reduce**（每个 SP rank 只见 1/sp 序列，参数梯度是局部贡献，须在 `sp_group` 内求和/平均），与 FSDP 的 reduce-scatter 协调好次序。放弃"让 FSDP 吞多维 mesh"的方案，因为它会牵动 reshard_after_forward / optimizer state / checkpoint 的多维 DTensor 行为，风险显著更高且无业界先例。

**阶段 2 — USP 注意力算子（直接复用 sglang-d / FastVideo 实现以保证精度对齐）。**
最强的对齐策略是**训练侧直接复用本地 sglang-d 的 USP 算子与 SP 进程组初始化**，而不是另写一套再去对齐：`/sgl-workspace/sglang/.../multimodal_gen/runtime/layers/attention/layer.py` 的 `USPAttention`（Ulysses all-to-all 按头切 + PyTorch 原生 `_templated_ring_attention`，layout `[B,S_local,H,D]`）整套是从 hao-ai-lab/FastVideo "copied and adapted" 来的，rollout 已在用——训练侧用同一份代码 = 天然精度一致。配套复用其 `parallel_state` 的 `set_seq_parallel_pg`/`SequenceParallelGroupCoordinator` 建 ulysses/ring 子组（`sp = ulysses_degree × ring_degree`，维度序 `tp-sp-pp-cfg-dp`）。注意点：(1) **训练侧模型是 diffusers**（`pipeline.transformer`），没有 `USPAttention` 类，需把 diffusers 的 attention 调用导向 sglang-d 的 USPAttention（AttnProcessor 或 SDPA 调用点 monkey-patch），而非沿用当前 `patch_usp_attention.py`（那是 rollout 的 inference 桩，bypass 成 SDPA）；(2) 硬约束 `num_heads % ulysses_degree == 0`，且 **Ulysses 不适合 GQA/MQA**，须先查 Wan2.2 的 head 数确认可切；(3) 序列在 **patchify 之后、attention 之前**切分，RoPE 需用全局 position offset（`sp_rank * S_local`）；(4) 它取代当前 `parallel.py:34-37` 仅在 `cp_size>1` 才挂、且与 transformers>=5.4 不兼容的 `ring_flash_attn`（纯 Ring）。

**阶段 3 — 让 diffusion 自有训练逻辑感知 SP。**
让 actor 自己的 `_forward_tile`、advantage/loss、log_prob 在序列被切分后仍正确：序列维切分、跨 `sp_group` 的损失/优势归约、以及 SP 组内一致的噪声 RNG。**这里需要重写一套 diffusion-native 的 SP 工具，而不是复用 causal 的 `cp_utils`**（理由见下）。

**阶段 4 — 权重同步（train→sglang）双侧适配。**
SP 不分片参数（每个 SP rank 持完整的 FSDP 分片副本），需分三层处理：(a) **训练侧打包**——因阶段 1 采用 Option B（FSDP 仍只在 dp 维 shard，参数 DTensor 的 `device_mesh` 仍是 1D），`update_weights` 的 `param.redistribute([Replicate()]*ndim).to_local()` 对 SP **透明、基本不用改**；唯一新增问题是多个 SP rank 各自持有相同完整参数，须去重以免冗余 IPC（见 (b)）。(b) **训练↔rollout rank 映射**——`connect_rollout_engines` 硬编码 `start_rank=i*rollout_num_gpus_per_engine`、`tp_rank=rank-start_rank`，并假设 train actor 与 rollout engine 的 GPU id 相同；dp×sp 拓扑下 rank 排布改变，须重新对齐该映射，并解决"多个 SP rank 持相同参数是否冗余 IPC、如何选 gather src"。这是最可能需要改的点。(c) **rollout 接收端**——`sglang_diffusion_engine.py` 的 `update_weights_from_tensor` 只转发完整张量；若 rollout 也开 `sp_degree>1`，sglang-diffusion 的 WeightsUpdater 能否把完整权重正确分发到各 SP rank，需到 sglang 侧确认（本仓库之外）。可复用已有的 `get_weights_checksum` 做同步后的逐模块校验作回归门。

### 客观证据

- **CP≡SP、且 cp 切的是序列维**：`miles/backends/training_utils/cp_utils.py` 全程沿 `dim=0` 切 token（`qkv_format="thd"`），采用 zigzag 负载均衡 2-chunk 切法（`chunk_0` 前段 + `chunk_1` 对称后段，`cp_utils.py:32-33`）——这是 ring-attention 类 SP 的标准切分。
- **LLM 套件是死代码**：全仓库 grep 显示 `training_utils/{loss,data,cp_utils,log_utils}.py` 仅彼此互相 import（`data.py:18`、`log_utils.py:17-18`、`loss.py:23`），无任何 diffusion 入口引用。
- **diffusion 自带 loss，不走 LLM 套件**：`fsdp_utils/actor.py:301`（reward 广播到去噪步）、`:685-688`（`-advantage_tile*ratio` → PPO-clip → `per_cell_loss.mean()`）、`_forward_tile`/`advantage_window`/`tstep_indices` 全是 diffusion 自有实现；`loss.py:27-70` 则是纯 LLM（`get_responses(logits[1,T,V], tokens, response_lengths)`、`rollout_temperature`）。
- **diffusion 实际依赖的 CP 触点仅 `parallel.py`**：`actor.py:44` 调 `create_fsdp_parallel_state`；实际只用 `dp_mesh`(`:97`)、`dp_size`(`:48`)、`dp_cp_rank/dp_cp_size/dp_src_rank/dp_cp_group_gloo`(`:212-236` gather metrics)，**未使用 `cp_rank/cp_size/cp_group`**——当前 `cp_size=1` 时 `dp_cp_*` 即 world。
- **断言阻断**：`miles/utils/arguments.py:1149` `assert args.context_parallel_size == 1`（在 `validate_args` 中无条件执行）。
- **USP 基础已就位**：`miles/backends/sglang_diffusion_utils/monkey_patches/patch_usp_attention.py` 已导入 sglang-diffusion 的 `USPAttention`（当前重定向到 SDPA）；`sglang_diffusion_engine.py:319` 已通过 `ServerArgs` 转发 `sp_degree`/ulysses/ring degree。
- **现有 SP 算子的脆弱性**：`parallel.py:32-33` 注释指明 `ring_flash_attn` 与 transformers>=5.4 在纯 DP 下不兼容——这正是用 USP 取代它的理由之一。
- **业界对齐（调研）**：sglang-d 的 USP（`/sgl-workspace/sglang/.../multimodal_gen/runtime/layers/attention/layer.py` 的 `USPAttention`、`runtime/distributed/parallel_state.py` 的 SP 组初始化）整套从 hao-ai-lab/FastVideo "copied and adapted"，依赖 PyTorch≥2.4 的 `_templated_ring_attention` + FlashAttention/SageAttention 后端、yunchang 风格的 `set_seq_parallel_pg`。USP 标准库 feifeibear/long-context-attention 与 Open-Sora 均采用"SP（独立 group）+ ZeRO/FSDP 正交"，且约束 `num_heads % ulysses_degree == 0`、Ulysses 不适合 GQA/MQA——与本方案 Option B 选择一致。
- **sglang 接收端已 DTensor-aware**：`/sgl-workspace/sglang/.../multimodal_gen/runtime/loader/weights_updater.py` 的 `load_weights_into_model` 对 DTensor 参数调 `distribute_tensor(loaded_weight, param.device_mesh, param.placements)` 按 rollout 自身 mesh 重新分片——所以 rollout 开 `sp_degree>1` 时权重接收**基本可行**，剩余未知是 sglang 的 `sp_degree` 是否真的建了 SP 的 NCCL 组。
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

主方向是一条有严格阶段依赖的单一路线，备选则环绕其布置：阶段 0 的清理是"地基净化"，必须先行且以纯 DP 逐位回归为验收门，否则 LLM 残留会持续误导 SP 实现（本次讨论中它就已造成"扩展现有 CP""3D 网格"等误判）；阶段 1-2（解锁 cp 维 + USP 算子）是核心交付，Alt-1 的精度框架应与之并行建好，作为"算子和 sglang-diffusion 对齐"的客观验收；阶段 3 必须与阶段 2 同批次落地，因为算子正确不等于梯度正确——序列分片下的 loss/优势归约与 RNG 一致性若不到位，训练数值就是错的，而这里要走 Alt-4 的选项 B（新写 diffusion-native 切分）而非复用 causal `cp_utils`。阶段 4（权重同步双侧适配）是容易被忽略却会"静默跑偏"的一面：SP 不分片参数，且 Option B 下 FSDP 参数 mesh 仍是 1D、训练侧打包对 SP 透明，真正的坑在 train↔rollout 的 rank 映射、SP 维参数复制的冗余 IPC 去重、以及 rollout 侧 `sp_degree>1` 的接收分发（sglang 接收端已 DTensor-aware，基本可行，但需确认 sglang 真的建了 SP 的 NCCL 组），全程须以 `get_weights_checksum` 把关。验证侧由两件仪器收口：Alt-1 精度框架确认"算子和 sglang-diffusion 对齐"，Alt-3 的**轻量 10 步 perf 闸**对照 ①什么都不开 / ②纯 FSDP / ③FSDP+SP 三档，用少量关键指标（指标与阈值待讨论）判断 SP 是否达预期；达不到即触发 Alt-2（Mooncake）这一明确保留的安全出口——仅当 USP 的集合通信开销无法扩展到目标拓扑上的视频序列长度时才启动。一句话：阶段 0 净化地基，阶段 1-4 是 FSDP+USP 的交付主体（含权重同步），Alt-1 与 Alt-3 是精度与性能的放行仪器，Alt-2 是回退出口，Alt-4 是阶段 3 的实现选型。

--- Original Design Draft End ---

---

## BitLesson Selection (REQUIRED FOR EACH TASK)

Before executing each task or sub-task, you MUST:

1. Read @/workspace/809a2940-8360-4812-81c2-c7383f3f43e7/miles_diffusion/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task to select relevant lesson IDs
3. Follow the selected lesson IDs (or `NONE`) during implementation

Include a `## BitLesson Delta` section in your summary with:
- Action: none|add|update
- Lesson ID(s): NONE or comma-separated IDs
- Notes: what changed and why (required if action is add or update)

Reference: @/workspace/809a2940-8360-4812-81c2-c7383f3f43e7/miles_diffusion/.humanize/bitlesson.md

---

## Goal Tracker Rules

Throughout your work, you MUST maintain the Goal Tracker:

1. **Before starting a round**: Re-anchor on the original plan and current round contract
2. **Before starting a task**: Mark the relevant mainline task as "in_progress" in Active Tasks
   - Confirm Tag/Owner routing is correct before execution
3. **Active Tasks** are MAINLINE tasks only - side issues do not belong there
4. **Blocking Side Issues** are reserved for issues that truly stop mainline progress
5. **Queued Side Issues** are non-blocking and must not take over the round
6. **After completing a mainline task**: Move it to "Completed and Verified" with evidence (but mark as "pending verification")
7. **If you discover the plan has errors**:
   - Do NOT silently change direction
   - Add entry to "Plan Evolution Log" with justification
   - Explain how the change still serves the Ultimate Goal
8. **If you need to defer a task**:
   - Move it to "Explicitly Deferred" section
   - Provide strong justification
   - Explain impact on Acceptance Criteria
9. **If you discover new issues**:
   - Add to "Blocking Side Issues" only if mainline progress is blocked
   - Otherwise add to "Queued Side Issues" or keep them as `[queued]` tasks/backlog

---

Note: You MUST NOT try to exit `start-rlcr-loop` loop by lying or edit loop state file or try to execute `cancel-rlcr-loop`

After completing the work, please:
0. If you have access to the `code-simplifier` agent, use it to review and optimize the code you just wrote
1. Finalize @/workspace/809a2940-8360-4812-81c2-c7383f3f43e7/miles_diffusion/.humanize/rlcr/2026-06-01_09-39-22/goal-tracker.md (this is Round 0, so you are initializing it - see "Goal Tracker Setup" above)
2. Write your round contract into @/workspace/809a2940-8360-4812-81c2-c7383f3f43e7/miles_diffusion/.humanize/rlcr/2026-06-01_09-39-22/round-0-contract.md
3. Commit your changes with a descriptive commit message
4. Write your work summary into @/workspace/809a2940-8360-4812-81c2-c7383f3f43e7/miles_diffusion/.humanize/rlcr/2026-06-01_09-39-22/round-0-summary.md
