# Round 0 Summary

为 miles_diffusion（Wan2.2-T2V 视频 DiT，diffusion GRPO，FSDP2 后端 + sglang-d rollout）实现训练侧 **FSDP + USP 序列并行**。一轮内完成全部五阶段（AC-1~9）到 plan 下界以上。分支 `feat/usp`（base `usp-impl-base`），HEAD `90becb4`。

## What Was Implemented

**AC-1 隔离 LLM CP 死代码（不删，先隔离）**：`training_utils/{loss,data,cp_utils,log_utils}.py` 加 `__deprecated__` 标记 + AST 静态防引用守卫（死代码已 import-broken，故用 AST 非 import）。物理删除按 AC-1.1 推迟。

**AC-2 解锁 SP + Option B 复合并行**：解除 `arguments.py` 的 `assert context_parallel_size==1`（CP→SP 向后兼容别名）；新建 `sp_mesh.py` 纯函数（resolve/validate/subgroups/locate_rank，不假定卡数 2~1000+，对齐 sglang ulysses 连续/ring 跨步划分）；`create_fsdp_parallel_state` 改写建 (dp,sp) mesh、FSDP 仅 wrap dp 维、SP>1 复用 sglang `set_seq_parallel_pg_by_sp_groups` + 补建 `_SP` coordinator。

**AC-3/4 USP 算子接入 + 序列切分契约**：`sp_attention.py` 新建 `WanUSPAttnProcessor`（self-attn→sglang USPAttention、cross-attn→SDPA）、可微 shard/gather、`apply_sequence_parallel`（rope 切 + blocks[0] 切 + proj_out 后 gather + 强制 FA/bf16/持久化 forward_context）。RoPE 全局 offset 由切分序与 all-to-all 重建序一致天然对齐。

**AC-5 SP 梯度同步**：`actor._all_reduce_sp_grads`——FSDP reduce-scatter（dp 维平均）后跨 sp all-reduce(SUM, fp32)，clip/step 前。gather 点后移到 proj_out 后，使全部参数偏梯度统一、单一规则。

**AC-6/7 loss/RNG SP-aware**：proj_out 后 gather→全序列 → noise_pred 各 sp rank 逐位一致 → loss/log_prob/advantage 在全序列上算，无"local-mean 再平均"问题（AC-6 退化为 forward parity）；训练前向确定性 + 采样在 rollout（非 SP）→ RNG 三级一致天然成立（AC-7）。

**AC-8 权重同步**：分析确认 Option B + colocate 下结构性正确（dp-redistribute 后各 rank 持相同全量、单代表 rank 去重天然满足、无冗余 IPC）；checksum 验证从 LoRA 子类**提升到非-LoRA 基类**（Wan2.2 此前零验证）；checksum 两侧对称扩为 name+dtype+shape+bytes；Codex review 后加 connect_rollout_engines 拓扑契约启动断言 + verify TP 感知（tp==1 才比对全量）。task14 确认 sglang `initialize_model_parallel` 真建 SP NCCL 组。

**AC-9 perf 闸（训练步口径，无 reward）= GO**：`sp_perf_gate.py` 真实 A14B per-layer 维度，四档 fwd+bwd。容量 max seq@170GB：fsdp 362k→sp_dp2 758k(2.1×)→sp_dp4 1429k(3.9×)；激活斜率比 1:1/2.1:1/4.2；效率 dp2×sp2 92-101%、sp_dp4 86-97%；SP 通信 5-7%。go/no-go 护栏/容量/效率全达标 → **不触发 Mooncake**。

**修了 2 个 sglang 侧 bug**（wan-strict-mode 分支 0d3bb580b + 1f9c8981a）：USP all-to-all、Ring attention 推理算子做训练时断梯度，包成可微 autograd.Function，按 is_grad_enabled 分流（推理逐字不变）。

## Files Changed
- miles/utils/arguments.py、miles/backends/fsdp_utils/arguments.py（解锁 + SP args）
- miles/backends/fsdp_utils/{sp_mesh.py(新), parallel.py, sp_attention.py(新), actor.py, diffusion_update_weight_utils.py}
- miles/backends/training_utils/{loss,data,cp_utils,log_utils,parallel}.py（deprecation 标记）
- tests/{test_cp_deadcode_isolation, test_sp_mesh, sp_init_smoke, sp_attention_parity, sp_grad_sync_parity, sp_weight_sync_parity, sp_perf_gate}.py（新）
- sglang(wan-strict-mode): runtime/layers/usp.py（可微 all-to-all + ring）、runtime/loader/weight_utils.py（checksum dtype/shape）
- .humanize/reports/usp-video-sp-impl-report.md（living 文档，全程维护）

## Validation
- `pytest tests/test_cp_deadcode_isolation.py tests/test_sp_mesh.py` → 27 passed
- sp_init_smoke（8×B200，5 配置）、sp_attention_parity（ulysses sp2/sp4 + ring u2r2 × ckpt，forward 逐位一致、权重梯度 rel≤8e-3，fp32 复跑 ~1e-6 证无损）
- sp_grad_sync_parity（dp1×sp4，全 69 参数全量梯度==全序列单进程参考）
- sp_weight_sync_parity（dp2×sp2 / dp1×sp4-u4 / u2r2，全量重建 checksum 跨 rank 逐位一致==单进程参考 + shape/dtype 敏感）
- sp_perf_gate（4×B200，四档，go/no-go=GO）
- Codex（gpt-5.5:high）中途 review stage 2/3/4：P1-A 证伪、P1-B/P2 已修

## Remaining Items（Explicitly Deferred，见 goal-tracker）
1. 完整 10 步 RL 的 weight-sync/rollout-wait 端到端计时 + 真 reward + MILES_VERIFY_WEIGHT_SYNC=1 在线校验 → GPU-gated（需 5 卡），不改 go/no-go 结论。
2. 物理删除 LLM CP 死代码 → AC-1.1 门控，AC 全绿后单独 PR。
3. 持续 gate hook 安装 → humanize 安装器与 codex 0.135 命名不兼容，且剩余工作多 GPU-gated。

## BitLesson Delta
Action: add
Lesson ID(s): BL-20260601-usp-autograd, BL-20260601-checksum-dtype-shape, BL-20260602-profiler-attr
Notes: 新增 3 条可复用教训——(1) 复用 sglang/FastVideo 推理 USP 算子做训练须补 all-to-all/ring 的 autograd（否则静默断梯度）；(2) 权重 checksum 须含 dtype/shape 且对 TP 分片 rollout 做 tp==1 gate；(3) torch≥2.9 profiler 用 self_device_time_total。
