# BitLesson Knowledge Base

This file is project-specific. Keep entries precise and reusable for future rounds.

## Entry Template (Strict)

Use this exact field order for every entry:

```markdown
## Lesson: <unique-id>
Lesson ID: <BL-YYYYMMDD-short-name>
Scope: <component/subsystem/files>
Problem Description: <specific failure mode with trigger conditions>
Root Cause: <direct technical cause>
Solution: <exact fix that resolved the problem>
Constraints: <limits, assumptions, non-goals>
Validation Evidence: <tests/commands/logs/PR evidence>
Source Rounds: <round numbers where problem appeared and was solved>
```

## Entries

## Lesson: sglang-usp-collectives-no-autograd
Lesson ID: BL-20260601-usp-autograd
Scope: sglang multimodal_gen runtime/layers/usp.py（训练侧复用 rollout 推理算子）
Problem Description: 复用 sglang/FastVideo 的 USP（Ulysses all-to-all、Ring attention）做**训练**时，反向静默断梯度——to_q/k/v.weight.grad 为 None（all-to-all 后的 to_out 有梯度故"看似对"），Ring 的 dK/dV rel≈0.59。
Root Cause: 这些算子是为推理（no_grad）"copied & adapted"：(1) `torch.distributed._functional_collectives.all_to_all_single` 未注册 autograd kernel；(2) `ring_attn` 直接调 `_templated_ring_attention`（仅 forward 模板），KV 环旋转的 functional collective 不可微。
Solution: 包成 autograd.Function——`_AllToAllSingle`（even-split all-to-all 是对合，反向=同一 all-to-all）；`_RingFlashAttention`（fwd=_templated_ring_attention, bwd=_templated_ring_attention_backward，op 用 torch 原生 aten flash），按 `torch.is_grad_enabled()` 分流，推理路径逐字不变。
Constraints: `_templated_ring_attention_backward` 的 import 路径随 torch 版本变（2.9 在 `torch.distributed.tensor.experimental._attention`，2.11+ 移到 `_context_parallel._attention`）；bf16 下权重梯度 rel 收敛到 ~e-3（求和舍入，非损失，fp32 复跑 ~1e-6 证明）。
Validation Evidence: tests/sp_attention_parity.py（ulysses sp2/sp4 + ring u2r2 × ckpt 全过）；sglang commits 0d3bb580b + 1f9c8981a。
Source Rounds: 0

## Lesson: weight-checksum-bytes-only-misses-shape
Lesson ID: BL-20260601-checksum-dtype-shape
Scope: 训练↔rollout 权重同步校验（diffusion_update_weight_utils.py + sglang loader/weight_utils.py）
Problem Description: 纯 `name+bytes` 的 SHA-256 校验对"同 dtype、同总字节、不同 shape"（转置/reshape）不敏感——能通过校验但语义错；且 TP 分片 rollout（Column/RowParallelLinear, tp>1）的 materialized 权重是逐分片，与训练侧全量哈希必然不匹配，开校验会假性报错。
Root Cause: (1) 字节哈希丢了 shape 语义；(2) 比对双方分片语义不一致（全量 vs 分片）。
Solution: (1) 两侧**对称**把哈希扩为 `name+dtype+shape+bytes`；(2) train↔engine 全量比对仅在 tp==1 执行，tp>1 打日志跳过；跨 engine 一致性比对保留（探测 replica 发散）。
Constraints: 两侧必须逐字节同算法否则恒不匹配；校验是 opt-in（MILES_VERIFY_WEIGHT_SYNC）、属验证层不进热路径。
Validation Evidence: tests/sp_weight_sync_parity.py（3 档拓扑全量重建 checksum 跨 rank 逐位一致==单进程参考 + shape/dtype 敏感性）；commits ee9264f + 5835316。
Source Rounds: 0

## Lesson: torch-profiler-self-device-time
Lesson ID: BL-20260602-profiler-attr
Scope: torch.profiler 通信占比测量（tests/sp_perf_gate.py）
Problem Description: 用 `key_averages()` 汇总 CUDA 自时间做通信分解时全为 0。
Root Cause: 新版 torch（2.9）弃用 `self_cuda_time_total`（返回 0），改名 `self_device_time_total`。
Solution: 取 `getattr(e,"self_device_time_total",0) or getattr(e,"self_cuda_time_total",0)`；集合通信按 kernel 名分桶：`ncclDevKernel_*AllToAll*/SendRecv`=SP(ulysses/ring)，`*AllGather*/*ReduceScatter*`=FSDP。
Constraints: kernel 名匹配是近似；Ulysses all-to-all 还含 split_with_sizes_copy/cat 等非 nccl kernel（未计入通信，偏保守）。
Validation Evidence: sp_perf_gate.py 实测 SP 通信 5-7%、FSDP 0-8%。
Source Rounds: 0
