AGREE:
- 采纳 Option B 是正确收敛方向：SP 独立 process group，与 FSDP/ZeRO 正交，符合 USP/yunchang/sglang-d 的实际范式。
- rollout 暂不开 `sp_degree>1` 是合理降险：先证明训练侧 SP 正确，再处理 sglang rollout SP，能避免同时调 mesh、attention、权重同步三类问题。
- 阶段0 改为隔离+deprecated+防引用测试，比直接删除稳妥，尤其当前 diffusion loss 不走 LLM `training_utils`，但仍需防止隐式 import。
- Wan2.2 的 `num_attention_heads=40` 支持 `sp=2/4`，所以 `ulysses=2/4` 或 `ulysses2×ring2` 都有可行空间。
- perf 同测 `dp2×sp2` 和 `dp1×sp4` 是必要的：前者验证 FSDP+SP 正交，后者验证最大序列容量上限。
- 权重同步阶段把 rollout 保持非 SP，并让训练侧打包对 SP 透明，方向正确；真正风险集中在 SP rank 冗余发送、rank 映射和 checksum。

DISAGREE:
- “参数 SP 维复制 + 手动 SP 梯度 all-reduce”描述仍不够精确。关键不是是否 all-reduce，而是它发生在 FSDP reduce-scatter 前还是后、按 full grad 还是 local shard、loss scaling 如何处理。这个顺序若错，可能数值近似但优化器状态长期漂移。
- “直接复用 sglang-d 的 USPAttention = 精度对齐”仍然不能推出训练正确。sglang-d 主要服务推理/rollout路径，训练侧还要证明 backward、activation checkpoint、autocast、FSDP hook、RNG、RoPE offset 全部一致。
- “复用 mesh.get_group('cp')”命名上有歧义。若语义已经从 context parallel 变为 sequence parallel，应明确 `cp` 只是历史维名，或引入 `sp` alias，否则后续维护会把 CP/SP 混用。
- 阶段3 的“sde_log_prob 对空间维 mean 跨 sp_group 归约”需要重新核对数学语义。若原始 loss 是对完整 token/空间维求 mean，则 SP 后要么 `sum + global_count`，要么严格等价的加权 mean；不能默认各 rank local mean 再平均。
- `dp1×sp4` 不能代表完整 FSDP+SP 组合收益，因为 DP 维为 1 时 FSDP 基本退化，参数/优化器状态分片收益消失。它适合测序列容量，不适合作为 FSDP+SP 效率结论。
- Mooncake fallback 的 go/no-go 仍偏结果导向，缺少瓶颈归因。若 SP 未达标，需要先确认瓶颈是 attention 通信、FSDP 通信、activation、weight sync、rollout 等哪一类，否则 Mooncake 可能解决错问题。

REQUIRED_CHANGES:
- 明确定义训练 mesh：例如 `train_world=4` 下逻辑维度为 `dp × sp`，参数在 `sp` 维 replicated，在 `dp` 维由 FSDP sharded；同时写清 rank 映射、group 成员、`dp2×sp2` 和 `dp1×sp4` 的具体表。
- 把 SP 梯度同步设计写成可验证协议：同步对象、通信 op、平均/求和系数、与 FSDP backward hook 的次序、optimizer step 前的不变量。建议先做单层 attention + 小 DiT block 的 DDP parity 测试。
- 阶段2 必须增加 USP backward/checkpoint 验证门：同 seed、同输入、同参数，对比非 SP 与 SP 的 forward loss、参数 grad、activation checkpoint 开/关、bf16/fp32 混合精度。
- RoPE 全局 offset 要从方案文字升级成接口约束：每个 SP rank 的 sequence shard 必须知道 global token index，不能只按 local `S_local` 生成位置。
- loss 归约必须逐项列出：`log_prob`、`sde_log_prob`、entropy/kl、advantage window、mask/count、clip ratio 的 SP 归约方式，并证明与非 SP 版本等价。
- RNG 一致性要区分样本级、token级、dropout/noise级。SP 组内一致 RNG 不应导致不同 token shard 误用相同随机数；需要明确 seed offset 或 generator partition。
- 权重同步必须规定只由每个 SP replica group 的一个代表 rank 发送，或者发送前去重；checksum 要覆盖 dtype、shape、shard/replica语义，不能只看 tensor bytes。
- perf 闸增加通信分解：至少拆出 USP all-to-all/ring、FSDP reduce-scatter/all-gather、weight sync、rollout等待。否则“SP通信<30%”无法指导优化。

OPTIONAL_IMPROVEMENTS:
- 将 `cp` 维逐步重命名或封装为 `sp_group` API，保留向后兼容，避免新代码继续传播 CP 术语。
- 第一阶段可先只支持 `ulysses=sp, ring=1`，待正确性稳定后再打开 `ulysses2×ring2`，减少 ring attention backward 变量。
- 增加一个最小 Wan attention fixture：固定 `B,S,H,D`、mask、RoPE、bf16，作为 CI 中的 SP parity 单测。
- `dp1×sp4` 的指标单独标注为 capacity test；效率评估主看 `dp2×sp2` 与更大集群时的 `dpN×spM`。
- deprecated 死代码建议加 import-level fail test，而不是只靠目录标记，防止后续误引用。
- Mooncake 分支提前定义触发条件：例如 attention通信主导才继续 USP优化，参数/权重同步主导才考虑 Mooncake 或异步同步路径。

UNRESOLVED:
- 是否允许训练侧 monkey-patch diffusers attention，还是必须改成显式 AttnProcessor 注入。前者快但脆，后者工程边界更清楚。
- SP 梯度 all-reduce 最终采用 FSDP hook 前 full-grad 同步，还是 FSDP reduce-scatter 后 shard-grad 同步。两者都可行，但必须选一个并写进实现契约。
- `sp=4` 首选 `ulysses4` 还是 `ulysses2×ring2`。前者简单，后者可能更省通信/显存但训练 backward 风险更高。
- go/no-go 后 Mooncake 的目标问题仍需确认：是替代 SP，还是只解决 rollout/权重同步/跨节点通信瓶颈。当前方案还不能直接推出 Mooncake 是正确 fallback。
