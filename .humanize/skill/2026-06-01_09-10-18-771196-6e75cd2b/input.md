# Ask Codex Input

## Question

你是资深分布式训练系统架构师，正在对一份已迭代的实现方案做收敛复审（第二轮）。请全程中文，按指定格式输出。

## 背景
miles_diffusion = diffusion GRPO 训练框架（FSDP2 训练 + sglang-diffusion rollout），目标：为视频 DiT（Wan2.2）训练加序列并行 SP，算子用 USP，与 sglang-diffusion 精度对齐；SP 不达标则回退 Mooncake。你在首轮已提出：复合 mesh 要明确参数复制维、阶段0先删风险高、USP 训练侧反向/checkpoint 支持存疑、精度对齐≠训练正确、权重同步可能静默错误、Mooncake 需重新论证瓶颈。以下是据此和代码调研迭代后的方案。

## 新增调研事实与用户决策

**调研对齐（sglang-d + 业界）：**
- sglang-d 的 USP 整套从 hao-ai-lab/FastVideo "copied and adapted"：USPAttention = Ulysses(all-to-all 切 head) + PyTorch 原生 `_templated_ring_attention`，layout `[B,S_local,H,D]`，`sp=ulysses×ring`，维度序 `tp-sp-pp-cfg-dp`，依赖 torch≥2.4 + FlashAttention 后端 + yunchang 风格 `set_seq_parallel_pg`。
- 业界一致用"SP 独立 process group + ZeRO/FSDP 正交"（yunchang 要求配 ZeRO-1/2；Open-Sora Ulysses+ZeRO；sglang-d 用独立 SequenceParallelGroupCoordinator）→ 采纳 Option B。
- sglang 接收端 `weights_updater.py` 已 DTensor-aware（`distribute_tensor` 按 rollout mesh 重分片）。
- 硬约束：`num_heads%ulysses==0`，Ulysses 不适合 GQA/MQA。
- **Wan2.2-T2V-A14B 实测：num_attention_heads=40, head_dim=128, num_layers=40, MoE 双 transformer。** → 40 对 sp=2(ulysses2)、sp=4(ulysses4 或 ulysses2×ring2) 均满足整除。
- diffusion 训练自有 loss（actor.py 的 PPO-clip `_forward_tile`/`advantage_window`），不走 LLM training_utils；latent `(B,T,C,H,W)`，T 是去噪步（flatten 进 batch），attention 序列是空间/时空 token。

**用户决策：**
1. 阶段0 清理：先隔离+标记 deprecated+防引用测试，SP 跑通后再物理删除（不一上来删）。
2. rollout 暂不开 sp_degree>1（保持非 SP），先把训练侧 SP 跑通——大幅降低阶段4 sglang 侧风险。
3. perf 两档都测：dp2×sp2 与 dp1×sp4。
4. 拓扑：4 卡训练 + 1 卡独占 rollout（含 reward），训练 NCCL world=4 卡。

## 最终候选方案 v2（5 阶段）
- **阶段0**：隔离 LLM 死代码（training_utils/{loss,data,cp_utils,log_utils}），纯 DP 逐位回归门；保留 parallel.py/ParallelState。
- **阶段1**：解除 cp_size==1 断言；Option B —— FSDP 仍 wrap dp 子 mesh，SP 用独立 sp_group（复用 mesh.get_group("cp")），参数 SP 维复制 + **手动 SP 梯度 all-reduce**（与 FSDP reduce-scatter 协调次序）。
- **阶段2**：训练侧直接复用 sglang-d 的 USPAttention + set_seq_parallel_pg（与 rollout 同一份代码 = 精度对齐）；训练模型是 diffusers，需把其 attention 导向 USPAttention（AttnProcessor/monkey-patch）；序列 patchify 后切，RoPE 全局 offset。
- **阶段3**：diffusion 自有 loss/advantage/log_prob 的 SP-aware：序列相关 reduce（sde_log_prob 对空间维 mean）跨 sp_group 归约；SP 组内一致 RNG；样本分区（DP round-robin）不受 SP 影响。
- **阶段4**：权重同步——Option B 下训练侧打包对 SP 透明；主要处理多 SP rank 冗余 IPC 去重 + connect_rollout_engines 的 rank 映射；rollout 暂不开 SP 故 sglang 侧风险延后；get_weights_checksum 把关。
- **perf 闸**：10 步 RL，三档 ①DDP ②纯FSDP ③FSDP+SP；指标（护栏：loss/grad 偏差<2%；容量：最大序列长度≥2×、显存↓≥40%；效率：并行效率≥60%、SP通信<30%），go/no-go 决定是否上 Mooncake。

## 输出格式（中文，每段列表）
- `AGREE:` 认可的点
- `DISAGREE:` 仍不合理的点及理由
- `REQUIRED_CHANGES:` 收敛前必须修正项
- `OPTIONAL_IMPROVEMENTS:` 非阻塞改进
- `UNRESOLVED:` 需用户决策的对立意见

## Configuration

- Model: gpt-5.5
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-06-01_09-10-18
- Tool: codex
