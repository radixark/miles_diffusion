# USP 视频序列并行 — 实现报告（Living Document）

> **状态：进行中（活文档）。** 本报告从设计阶段开始维护，实现过程中随每次系统设计/代码改动/遇到的明显 bug 持续更新，收尾时补齐 parity 与 perf 结论。
> 配套：计划 `.humanize/plans/usp-video-sp-plan.md`，草稿 `.humanize/ideas/usp-video-sp-20260601-080512.md`。
> 分支：`feat/usp`。最近更新：2026-06-01（设计定稿，实现未开始）。

---

## 1. 目标概述

为 miles_diffusion（Wan2.2-T2V 视频 DiT 的 diffusion GRPO 训练，FSDP2 后端 + sglang-diffusion rollout）引入**序列并行（SP）**，注意力算子用 **USP**（Ulysses + Ring），与 sglang-diffusion 算子精度对齐；SP 不达标则评估回退 Mooncake。实现分五阶段：阶段0 隔离 LLM CP 死代码 → 阶段1 解锁 SP + Option B 复合并行 → 阶段2 复用 sglang-d USP 算子 → 阶段3 diffusion 训练逻辑 SP-aware → 阶段4 权重同步适配。

---

## 2. 系统设计决策（及理由）

| # | 决策 | 选择 | 理由 |
|---|------|------|------|
| 复合并行 | FSDP×SP 形态 | **Option B**：FSDP 仅在 dp 维 shard 参数 + SP 独立 process group + 参数 SP 维复制 + 手动 SP 梯度 all-reduce | 业界一致（yunchang 要求配 ZeRO-1/2、Open-Sora Ulysses+ZeRO、sglang-d 独立 SequenceParallelGroupCoordinator）；避免 FSDP 吞多维 mesh 牵动 reshard/optimizer/checkpoint |
| USP 算子 | 实现来源 | **直接复用本地 sglang-d 的 USPAttention**（= hao-ai-lab/FastVideo 同源） | 与 rollout 同一份代码 = 天然精度对齐；避免自研再对齐 |
| DEC-1 | diffusers attention 接入 | **AttnProcessor 注入**（monkey-patch 兜底） | 工程边界清晰、可控 |
| DEC-2 | SP 梯度同步时机 | **FSDP reduce-scatter 后对 shard-grad 同步** | 与 FSDP 通信对象一致、避免重复 all-gather；以 DDP parity 验证 |
| DEC-3 | ulysses/ring 组合 | **不假定卡数（2~1000+）；ulysses×ring 全程可配；ring 为必备能力**。默认 Ulysses 节点内（受 num_heads 限制、NVLink）、Ring 跨节点扩展 | 大规模必须二者组合；不能因当前小规模写死 degree |
| DEC-4 | Mooncake 定位 | **不预设为 SP 替代；先瓶颈归因再定** | KVCache 中心更贴合自回归推理，DiT 训练压力在 activation/optimizer/attention 通信 |
| 清理时机 | 阶段0 | **先隔离+防引用测试，SP 跑通后再物理删除** | 降低早期回归风险，保住验证窗口 |
| 拓扑 | 当前硬件 | 4 卡训练 + 1 卡独占 rollout(含 reward)；perf 测 dp2×sp2 与 dp1×sp4 | 仅当前验证点，非架构假设 |

**关键代码事实**（已查证）：CP≡SP（cp 维即序列维，被 `arguments.py` 的 `assert context_parallel_size==1` 锁死）；`training_utils/{loss,data,cp_utils,log_utils}.py` 是 LLM RL 死代码、diffusion 训练不调用（自带 PPO-clip loss 在 `actor.py`）；Wan2.2 num_heads=40/head_dim=128/layers=40，对 sp=2/4 满足 Ulysses 整除；sglang 接收端 `weights_updater.py` 已 DTensor-aware。

---

## 3. 设计演进与纠偏（讨论中纠正的误判，留作备忘）

- ❌ 起初把 CP 与 SP 当两个正交维（设计过 3D `(sp,dp,cp)` 网格）→ ✅ 纠正：**CP≡SP**，2D `(dp, cp)` 中 cp 维即 SP 维。
- ❌ "扩展现有 CP 路径" → ✅ 纠正：现有 CP 是被断言禁用的 LLM 死代码脚手架，应**清理**而非扩展。
- ❌ "让 FSDP 感知整个复合 mesh" → ✅ 纠正：业界用 Option B（SP 独立 group）。
- ❌ "复用 rollout 的 patch_usp_attention" → ✅ 纠正：那是 inference 桩（bypass SDPA），训练侧需真正 USPAttention forward。
- ❌ 基于"4 卡"做拓扑决定 → ✅ 纠正：**不假定卡数（2~1000+）**，ulysses/ring 全程可配。

---

## 4. 代码改动日志（按阶段/文件，实现中填充）

> 每次改动追加一条：`文件 — 改了什么 — 为什么 — 对应 AC/task`。

_（实现未开始）_

---

## 5. 遇到的明显 bug / 坑（实现中填充）

> 每条记录：现象 — 根因 — 修复 — 影响范围。

_（实现未开始）_

---

## 6. 算子 parity 与 perf 结论（实现中/收尾填充）

- USP↔sglang-d parity（forward/backward/checkpoint/混精）：_待填_
- 10 步 perf 三档（DDP / 纯FSDP / FSDP+SP，dp2×sp2 与 dp1×sp4）+ 通信分解：_待填_
- go/no-go 判定与 Mooncake 是否触发：_待填_

---

## 7. 遗留问题与后续

_（实现未开始）_
