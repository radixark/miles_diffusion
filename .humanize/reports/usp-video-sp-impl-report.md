# USP 视频序列并行 — 实现报告（Living Document）

> **状态：进行中（活文档）。** 本报告从设计阶段开始维护，实现过程中随每次系统设计/代码改动/遇到的明显 bug 持续更新，收尾时补齐 parity 与 perf 结论。
> 配套：计划 `.humanize/plans/usp-video-sp-plan.md`，草稿 `.humanize/ideas/usp-video-sp-20260601-080512.md`。
> 分支：`feat/usp`。最近更新：2026-06-01（阶段2：Ulysses+Ring 算子 parity 全过（含 backward/ckpt））。

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
| `training_utils/parallel.py` | `ParallelState` 加 sp_rank/sp_size/sp_group/ulysses_degree/ring_degree/ulysses_group/ring_group | 承载 SP 状态；cp_* 作兼容别名 | AC-2 |
| `fsdp_utils/parallel.py` | `create_fsdp_parallel_state` 改写：建 (dp,sp) mesh、FSDP 仍只 wrap dp（Option B）、SP>1 时复用 sglang `set_seq_parallel_pg_by_sp_groups` 建 ulysses/ring 组；移除从未执行过的 ring_flash_attn 调用（USP 取代，阶段2） | Option B 接线 | AC-2 |
| `tests/sp_init_smoke.py`（新建） | torchrun 多卡 smoke：真实 NCCL 下校验 dp/sp/ulysses/ring 组成员与纯函数一致 | AC-2 多卡验证 | AC-2 |

验证：单测 `pytest tests/` → **22 passed**；多卡 smoke（8×B200）**5 配置全通过**：4卡{sp2, sp4(u4), sp4(u2r2)}、8卡{sp4, sp2}。
关键确认：sglang.multimodal_gen 训练环境**可直接 import**，已实际复用其 `set_seq_parallel_pg_by_sp_groups` 建组。**阶段1（AC-2）完成。**

### 阶段2 — USPAttention 接入 diffusers Wan2.2（里程碑 C，**Ulysses + Ring parity 全通过**）
| 文件 | 改动 | 理由 | AC |
|---|---|---|---|
| `fsdp_utils/parallel.py` | SP>1 时除 `set_seq_parallel_pg_by_sp_groups`（设 ULYSSES_PG/RING_PG）外，补建并注册 sglang `parallel_state._SP` coordinator | 解决难点#3：USPAttention 经 `get_sp_group()` 读 `_SP`，仅设 PG 不够 | AC-3 |
| `fsdp_utils/sp_attention.py`（新建） | `WanUSPAttnProcessor`（self-attn→USPAttention、cross-attn→SDPA）；`shard_sequence`/`gather_sequence`（可微 all-gather）；`apply_sequence_parallel`（rope 切 + block 切/聚 hook + 后端/精度/forward_context 初始化）；`init_sp_backend` | task5/task7 主体 | AC-3,4 |
| `fsdp_utils/actor.py` | load+FSDP 后 `sp_size>1` 调 `apply_sequence_parallel(..., compute_dtype=_forward_dtype)` | 生产接线 | AC-3,4 |
| `sglang .../layers/usp.py` | `_usp_all_to_all_single` 包成可微 autograd Function（`_AllToAllSingle`，even-split all-to-all 是对合、反向=同一 all-to-all） | **修 sglang 侧 bug**：原 `ft_c.all_to_all_single` 无 autograd kernel，训练反向静默断梯度 | AC-3 |
| `tests/sp_attention_parity.py`（新建） | 真实 Wan DiT 小配置 SP-vs-全序列FA 参考；forward/输入梯度/各 block self-attn 权重梯度；ckpt 开关；ulysses/ring | task6 | AC-3,4 |
| `tests/sp_init_smoke.py` | 补校验 6 个 sglang SP getter 经 `_SP` 返回正确 | 难点#3 回归 | AC-2 |

**parity 结果（4×B200，bf16，S=64 H=8 D=128，与全序列 FA 参考比）：**
- `dp1×sp2(u2)`、`dp1×sp4(u4)`，ckpt 开/关四档：forward **逐位一致**（rel=0）、输入梯度逐位一致、各 block `to_q/k/v/to_out` 权重梯度 rel 5e-3~8e-3、cos(1-) ~6e-6。**全过。**
- `dp1×sp4(u2×r2)` Ring × ckpt{on,off}：**full backward 通过**——forward rel 3.6e-3、输入梯度 rel 6.5e-3、权重梯度 rel 5.6e-3~8.4e-3（修 §5 坑5 后 `to_k` 从 0.59 回到 5.6e-3）。

**接口调研（实现前）：**
- diffusers `WanTransformer3DModel`（`models/transformers/transformer_wan.py`）：`WanAttnProcessor.__call__(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)`。流程 QKV→`norm_q/k`(RMSNorm)→reshape `[B,S,H,D]`→`apply_rotary_emb`（仅 self-attn）→`dispatch_attention_fn`。self-attn（`encoder_hidden_states is None`）走 RoPE、需 SP 切；cross-attn（对 512 text token）不切。layout `[B,S,H,D]` 与 USPAttention 期望一致。`set_attn_processor` 可注入自定义 processor。
- sglang `USPAttention.forward(q,k,v=[B,S_local,H,D], attn_mask, num_replicated_prefix/suffix, skip_sequence_parallel_override)`。
- 注入方案（DEC-1 AttnProcessor）：写 `WanUSPAttnProcessor` 复用 Wan 的 QKV/norm/RoPE，self-attn 调 USPAttention、cross-attn 走原 dispatch。

**关键难点（实现时必须处理）——均已落地：**
1. **序列切分点（AC-4 核心，仅换 processor 不够）** ✅：在 rope 产出处把 `(cos,sin)` 切到 `S_local`（每 block 复用同一份）；用 `blocks[0]` forward_pre_hook 切 `hidden_states` 到 `S_local`、`blocks[-1]` forward_hook 可微 all-gather 回 `S_full`。norm/MLP 在 `S_local` 跑，attention 内部 all-to-all 临时聚到 `S_full`。比重写 diffusers forward 更鲁棒（不随版本漂移）。
2. **RoPE 全局 offset（AC-4）** ✅：切 `hidden_states` 与切 rope 用同一 `shard_sequence`（连续 `[sp_rank*S_local:+S_local]`），与 USPAttention all-to-all 的重建序一致 → 全局 position 自然对齐。parity 逐位一致即证明 offset 正确。
3. **USPAttention 全局状态依赖** ✅（见难点#3 / §5 坑）：确认仅设 `ULYSSES_PG/RING_PG` 不够，须建 `_SP`；另需 `set_forward_context`（持久化以兼容 ckpt recompute）、强制 FA 后端、设 compute dtype=bf16（否则 FA 被降级 SDPA）。
4. cross-attn / I2V `add_k_proj` 不走 SP ✅：cross-attn（text KV 各 rank 复制）走本地 SDPA，无 SP 通信。

---

## 5. 遇到的明显 bug / 坑

- **死代码已 import-broken**（Round 0 发现）：`log_utils.py` 有 `from miles.utils.flops_utils import calculate_fwd_flops`，但 `miles.utils.flops_utils` 在 diffusion fork 里**不存在** → `ModuleNotFoundError`。
  - 根因：这套 LLM 死代码从上游 miles 继承，依赖的 `flops_utils` 在 diffusion fork 被删，但死代码未清理。
  - 影响/处理：(1) 这是比"无 import 引用"更强的死代码证据——连 import 都失败，diffusion 绝不可能用；(2) `__deprecated__` 标记的检查改用 **AST 静态解析**而非 `importlib.import_module`（不 import broken 模块）。
- **create_fsdp_parallel_state 依赖 gloo group 预初始化**（Round 0，多卡 smoke 发现）：`get_gloo_group()` 要求先 `init_gloo_group()`（正常训练在 `train_actor.py:84` 做）。非 bug，是既有依赖；smoke test 补调 `init_gloo_group()` 后通过。

### 阶段2（Round 1）

- **坑1 — 难点#3 实证：`set_seq_parallel_pg_by_sp_groups` 只设 PG、不建 `_SP`**。USPAttention.forward 经 `get_sp_group()`/`get_*_parallel_world_size()` 读模块全局 `_SP`（`SequenceParallelGroupCoordinator`），而 `set_seq_parallel_pg_by_sp_groups` 仅设 `PROCESS_GROUP.ULYSSES_PG/RING_PG`。`_SP` 只在 `initialize_model_parallel` 内建。处理：`parallel.py` 用 `init_parallel_group_coordinator(..., parallel_mode="sequence", ulysses_group=, ring_group=)` 单独建 `_SP` 并赋给 `parallel_state._SP`（与 rollout 同一组件）。
- **坑2 — sglang USP all-to-all 无 autograd kernel（训练反向静默断梯度）** ⚠️核心：`usp.py:_usp_all_to_all_single` 用 `torch.distributed._functional_collectives.all_to_all_single`，PyTorch 未给 `_c10d_functional::all_to_all_single`/`wait_tensor` 注册 autograd → 反向不过该算子。现象：`to_q/k/v.weight.grad` 全为 None（`to_out` 有 grad 因在 all-to-all 之后），输入梯度因残差/cross-attn 路径"看似对"实则缺 self-attn 贡献。根因：sglang 这套从 FastVideo "copied & adapted" 时为推理（no_grad）服务，丢了 FastVideo 原有的可微 `SeqAllToAll4D`。处理：在 sglang `usp.py` 加 `_AllToAllSingle` autograd Function（even-split all-to-all 是对合，反向=同一 all-to-all）。推理 no_grad 不调反向，对 rollout 透明。修后 ulysses 输入梯度逐位一致、权重梯度 rel<1e-2。
- **坑3 — compute dtype 默认 fp32 使 FA 被降级 SDPA**：`USPAttention.__init__` 用 `get_compute_dtype()` 选后端；训练进程没设 sglang mixed-precision state → 返回 `torch.get_default_dtype()`=fp32 → cuda 平台 `get_attn_backend_cls_str` 因 `dtype not in (fp16,bf16)` 把 FA 降级成 TORCH_SDPA。后果：ulysses 仍跑通（参考也 SDPA，故 parity 假性通过）、但 **Ring 启动即报错**（Ring 仅支持 FA/SAGE）。处理：`init_sp_backend` 调 `set_mixed_precision_policy(param_dtype=bf16,...)` + `global_force_attn_backend(FA)`；`apply_sequence_parallel` 传 `_forward_dtype`（非 FSDP master 的 fp32）。注：B200(sm100/Blackwell) 只支持 fa_ver=4（fa_ver=3 报 "only supported on sm90+"）。
- **坑4 — gradient checkpointing 下 `forward_context` 丢失**：原用 `with set_forward_context(...)` 包 forward，但 ckpt recompute 在 backward 阶段、`with` 已退出 → USPAttention `get_forward_context()` assert 失败。处理：`init_sp_backend` 改为持久化设 module-global `forward_context._forward_context`（一次设定、不退出），recompute 仍可读。
- **坑5 — Ring 训练反向 `dK/dV` 不正确 → 已修** ✅：现象（修前）`dp1×sp4(u2×r2)` forward 好（rel 3.6e-3）、`dQ`/`to_q.grad` 对，但 `to_k/to_v.weight.grad` rel≈0.59。根因：sglang `ring_attn` 直接调 torch `_templated_ring_attention`（**仅 forward 模板**）；环上 KV 旋转用 functional collective 不可微，故 dK/dV 拿不到 torch 另有的"反向环"梯度回传（`_templated_ring_attention_backward` 做反向 dKV 通信，正常由 torch `context_parallel` 的 autograd.Function 串接，sglang 绕过了它）。dQ 因 Q 不旋转而幸存。修法：在 `usp.py` 加 `_RingFlashAttention` autograd.Function，forward 调 `_templated_ring_attention`、backward 调 `_templated_ring_attention_backward`，op 用 torch 原生 aten flash（两模板都认的 canonical op）。`ring_attn` 按 `torch.is_grad_enabled()` 分流：训练走可微版、推理保持原 forward-only 路径**逐字不变**。修后 u2r2(含 ckpt) full backward：`to_k` rel 0.59→5.6e-3，全部权重梯度回到 bf16-reduction 级（~6-8e-3）。注：训练 ring 步用 torch aten flash（非 sglang FA kernel），ulysses 步仍用 sglang FA；rollout 本期非 SP，inference 路径不受影响。

---

## 6. 算子 parity 与 perf 结论（实现中/收尾填充）

- USP↔sglang-d parity（forward/backward/checkpoint/混精）：**Ulysses 已验证**——`sp2(u2)`/`sp4(u4)` × ckpt{on,off} 四档，forward+输入梯度逐位一致、self-attn 权重梯度 rel 5e-3~8e-3（bf16+FA）。**Ring（u2×r2，含 ckpt）full backward 亦通过**（坑5 已修）。
  - **权重梯度 5e-3 是 bf16 求和舍入、非 Ulysses 损失**：Ulysses all-to-all 只搬数据、无算术，故 forward / 输入梯度逐位一致（输入梯度各 rank 贡献不相交，跨 rank 求和=拼接、无舍入）。权重梯度则每 token 都贡献到同一矩阵：参考单进程对全 S 一次累加，SP 是每 rank 对 S/sp 个 token 算 bf16 偏导再 all-reduce(SUM)——分组求和的浮点非结合性，性质同 DDP/FSDP 梯度 all-reduce vs 单卡。**fp32 端到端复跑（`--fp32`，强制 SDPA）权重梯度 rel 降到 ~1e-6、forward 2.4e-7**，证实无损。生产中 SP 梯度 all-reduce 用 fp32（reduce_dtype）可进一步压低。
- 10 步 perf 三档（DDP / 纯FSDP / FSDP+SP，dp2×sp2 与 dp1×sp4）+ 通信分解：_待填（阶段F）_
- go/no-go 判定与 Mooncake 是否触发：_待填_

---

## 7. 遗留问题与后续

- **阶段3（AC-5/6/7）—— backward 的正式处理方案**：
  - **核心不变量**：序列分片区内的参数（transformer blocks、patch_embedding、condition_embedder）每个 sp rank 只见 S_local token → 偏梯度，须跨 sp `all_reduce(SUM)` 还原全量（参考已用此还原验证，rel 与全序列一致）。
  - **DEC-2 次序**：在 FSDP reduce-scatter **之后**对 sharded grad（DTensor local shard）做 sp all-reduce(SUM)，与 FSDP 通信对象一致、避免重复 all-gather；reduce 用 fp32。
  - **关键修正——让分片"均匀"以避免逐参数分类**：当前 gather 在 `blocks[-1]` 输出，导致 `norm_out`/`proj_out` 在**全序列**上复算（其梯度是全量、各 rank 相同，**不可再 SUM**，否则 ×sp）。阶段3 拟把 **gather 移到 `proj_out` 之后、unpatchify 之前**（norm_out/proj_out 是 token-wise，在 S_local 上跑等价）→ 整个 transformer 参数**一律偏梯度** → 单一均匀 `all_reduce(SUM)` 规则，无需区分。
  - **loss 归约（AC-6）**：现 PPO loss `per_cell_loss.mean()` 在 SP 下须改 `局部 sum + 跨 sp 全局 token 数`（不可 local-mean 再平均），与权重梯度 SUM 规则自洽。
  - parity 测试中"跨 sp sum 还原权重梯度"正是该协议的雏形与验证手段。
- **per-token timestep（ti2v）**：当前切分契约假定 `timestep_proj` 非逐 token（Wan2.2-T2V-A14B 成立，temb=[B,6,inner]）；若用逐 token timestep 需一并切 `timestep_proj`。
- **序列 padding**：`shard_sequence` 要求 `S % sp == 0`，否则报错；如遇不整除分辨率需在 patchify 前 padding（AC-4 已列）。
