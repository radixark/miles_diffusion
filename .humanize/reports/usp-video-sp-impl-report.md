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

## 4. 代码改动日志（按阶段/文件）

### 阶段0 — 隔离 LLM CP 死代码（里程碑 A，Round 0）
| 文件 | 改动 | 理由 | AC |
|---|---|---|---|
| `training_utils/{loss,data,cp_utils,log_utils}.py` | 顶部加模块 docstring + `__deprecated__ = True` | 标记为 LLM-RL 死代码，diffusion 训练不调用 | AC-1 |
| `tests/test_cp_deadcode_isolation.py`（新建） | 防引用守卫：AST 静态验证 4 模块有 `__deprecated__` + 全 `miles/`+`flow_grpo/` 无活引用 | AC-1 的 import-level 守卫；用 AST 而非 import（死代码已 import-broken，见 §5） | AC-1 |
| `tests/`（新建目录） | 仓库此前无测试目录 | 承载守卫测试 | AC-1 |

验证：`pytest tests/test_cp_deadcode_isolation.py` → **5 passed**。物理删除按 AC-1.1 推迟到 AC-2~6 全绿后。

### 阶段1 — 解锁 SP 维 + Option B 复合并行（里程碑 B，Round 0，进行中）
| 文件 | 改动 | 理由 | AC |
|---|---|---|---|
| `miles/backends/fsdp_utils/arguments.py` | 加 `sequence_parallel_size`/`ulysses_degree`/`ring_degree`；`context_parallel_size` 留作 SP 别名 | 暴露 SP 配置，不假定卡数 | AC-2 |
| `miles/utils/arguments.py` | 解除 `assert context_parallel_size==1`，改为 CP→SP 向后兼容 | 解锁 SP | AC-2 |
| `miles/backends/fsdp_utils/sp_mesh.py`（新建） | 纯函数 `resolve_sp_degrees`/`validate_sp_config`/`sp_subgroups`/`locate_rank`，对齐 sglang ulysses 连续/ring 跨步划分 | rank 映射/子组划分，不假定卡数（2~1000+），可独立单测 | AC-2 |
| `tests/test_sp_mesh.py`（新建） | 覆盖 2~1024 卡、各 ulysses×ring 组合的布局不变量 + sglang 对齐例 + 非法配置拒绝 + heads%ulysses 守卫 | AC-2 验证 | AC-2 |

验证：`pytest tests/` → **22 passed**。**待续**：`ParallelState` 扩 sp 字段 + `create_fsdp_parallel_state` 接真实 dist 组（复用 sglang `set_seq_parallel_pg_by_sp_groups`），需多 GPU 跑通验证。
关键确认：sglang.multimodal_gen 在训练环境**可直接 import**，为阶段2 复用 USPAttention 与本阶段复用 SP 组构建奠定基础。

---

## 5. 遇到的明显 bug / 坑

- **死代码已 import-broken**（Round 0 发现）：`log_utils.py` 有 `from miles.utils.flops_utils import calculate_fwd_flops`，但 `miles.utils.flops_utils` 在 diffusion fork 里**不存在** → `ModuleNotFoundError`。
  - 根因：这套 LLM 死代码从上游 miles 继承，依赖的 `flops_utils` 在 diffusion fork 被删，但死代码未清理。
  - 影响/处理：(1) 这是比"无 import 引用"更强的死代码证据——连 import 都失败，diffusion 绝不可能用；(2) `__deprecated__` 标记的检查改用 **AST 静态解析**而非 `importlib.import_module`（不 import broken 模块）。

---

## 6. 算子 parity 与 perf 结论（实现中/收尾填充）

- USP↔sglang-d parity（forward/backward/checkpoint/混精）：_待填_
- 10 步 perf 三档（DDP / 纯FSDP / FSDP+SP，dp2×sp2 与 dp1×sp4）+ 通信分解：_待填_
- go/no-go 判定与 Mooncake 是否触发：_待填_

---

## 7. 遗留问题与后续

_（实现未开始）_
