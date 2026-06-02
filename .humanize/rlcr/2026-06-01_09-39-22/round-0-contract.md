# Round 0 Contract

## Mainline Objective
交付 miles_diffusion 训练侧 FSDP + USP 序列并行的**完整实现**：隔离 LLM CP 死代码 → 解锁 SP + Option B 复合并行 → 复用 sglang-d USP 算子接入 diffusers → diffusion 训练逻辑 SP-aware（梯度/loss/RNG）→ 权重同步 checksum → 轻量训练步 perf 闸给出 go/no-go。

## Target ACs
AC-1 ~ AC-9（全部）。本轮把五阶段一次性做到 plan 下界以上：训练侧 SP 在 dp2×sp2 与 dp1×sp4 跑通、parity 全过、权重同步 checksum 通过、perf 闸 GO。

## Blocking Side Issues In Scope
- sglang USP 算子推理→训练的可微化（all-to-all、ring backward）——不修则训练梯度错，必须在本轮解决（已修，BL-20260601-usp-autograd）。
- weight-sync 拓扑契约 + checksum 语义（Codex review P1-B/P2）——已修。

## Queued / Out of Scope (this round)
- 完整 10 步 RL 的 weight-sync/rollout-wait 端到端计时 + 真 reward：GPU-gated（需 5 卡，当前 4 空闲），不改 go/no-go 结论。
- 物理删除 LLM 死代码：计划 AC-1.1 明确推迟到 AC 全绿后单独合入。
- 持续 gate hook 安装：humanize 安装器与 codex 0.135 不兼容，且本项目剩余多 GPU-gated。

## Round Success Criteria
- AC-1~8 的 parity / 守卫测试全过（已：deadcode 隔离、sp_mesh、sp_attention_parity ulysses+ring×ckpt、sp_grad_sync_parity dp1×sp4、sp_weight_sync_parity 三档）。
- AC-9 训练步 perf 闸产出三档对照 + 通信分解 + go/no-go=GO（容量 sp_dp2 2.1×、效率 92-101%、SP 通信 5-7%）。
- Living 报告与 goal-tracker 反映真实状态；遗留项显式 deferred 并附理由。
- Codex review-phase 对 feat/usp vs usp-impl-base 全 diff 无遗留 [P0-9]（本轮 review 目标）。
