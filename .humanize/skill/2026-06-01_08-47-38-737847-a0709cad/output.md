CORE_RISKS:
- `apply_fsdp2` 使用“完整复合 mesh”这个表述风险很高。FSDP 不应误把 SP 维当作参数 shard 维；目标应明确为参数在 FSDP shard 维分片、在 SP 维复制。否则会出现权重 shard 形态错误、优化器状态错误、权重同步重复或缺片。
- “CP≡SP”在概念上成立，但不能推出现有 `cp` 维可无痛复用。现有 `dp_cp_*`、gloo group、rank 映射、metric gather 可能隐含“dp×cp 是全训练 world”的假设，SP 打开后这些路径可能产生重复统计或错位聚合。
- 删除 LLM 残留作为阶段 0 的优先级值得质疑。它能降低认知噪音，但不直接降低 SP 交付风险；如果删除动作引入回归，会消耗早期验证窗口。更稳妥是先隔离、标记 deprecated、加禁止引用测试，等 SP 路径稳定后再物理删除。
- USPAttention 是否支持训练侧反向传播、activation checkpoint、FSDP mixed precision、torch compile/flash kernel 组合，是主风险之一。草稿把“rollout 侧已有 patch”视为训练可复用基础，这个假设偏强。
- 精度对齐不等于训练正确。即使单算子输出接近，SP 下 loss scaling、advantage 聚合、梯度归约、噪声 RNG、timestep/window 采样任一出错，RL 信号都会系统性偏移。
- 权重同步可能静默错误。多 SP rank 参数复制、FSDP shard gather、rollout SP/TP rank 映射三者叠加后，checksum 只能证明最终权重一致，不能证明“发送者选择、版本时序、模块名映射、dtype cast”全部正确。
- Mooncake 作为回退需要重新论证。KVCache 中心的解耦/卸载更自然适配自回归推理；DiT 视频训练的主要压力可能是 attention activation、MLP activation、optimizer state，而不是 KVCache。它未必能替代训练侧 SP。

MISSING_REQUIREMENTS:
- 需要明确 SP 切分的张量契约：latent/token 维具体在哪个 layout，视频维度是 `B,T,H,W,C`、`B,N,C` 还是 packed sequence，RoPE/pos embedding 如何随局部 shard 计算全局位置。
- 需要 head/sequence divisibility 约束：`num_heads % ulysses_degree == 0`、sequence padding、ring degree × ulysses degree 是否必须等于 `sp_degree`、非整除视频帧/分辨率如何处理。
- 需要定义 SP 下所有 reduce 的数学口径：per-token mean、per-sample mean、per-timestep mean、advantage normalization 是局部还是全局，PPO clip loss 的 denominator 是否保持与非 SP 完全一致。
- 需要 RNG 规范：噪声采样、dropout、timestep sampling、classifier-free guidance、augmentation、reward 相关随机性在 DP/SP rank 上如何保证可复现且不重复。
- 需要 rollout 与训练拓扑的资源模型：4+1 下 reward 独占卡是否参与 NCCL world，训练 world/ray actor/sglang router 的 rank 空间如何隔离，故障恢复时 group 如何重建。
- 需要明确 SP 是否只覆盖 attention，还是覆盖输入切分后的全 block。若 MLP、norm、residual、patchify/unpatchify 仍全量复制，显存收益会显著低于预期。
- 需要覆盖 variable length / padding mask。视频训练通常会有不同帧数、不同分辨率、裁剪 bucket；SP 工具必须支持 packed 或 padded batch，不只是固定长序列。
- 需要 optimizer/checkpoint/save-load 要求。FSDP×SP 多维 mesh 下 optimizer state dict、resume、EMA、模型导出、rollout 热更新都需要验收。
- 需要通信超时与诊断要求。USP all-to-all/ring、FSDP all-gather/reduce-scatter、gloo weight sync 混用时，必须有 NCCL debug、timeout、rank dump、collective 顺序检查。

TECHNICAL_GAPS:
- 草稿没有给出 FSDP2 mesh API 的具体 placement 设计。应明确 DTensor placements、FSDP shard mesh、SP process group、参数复制维，而不是笼统说“FSDP 感知整个复合 mesh”。
- diffusion-native SP 工具缺少最小接口设计。至少需要 `split_sequence`、`gather_sequence`、`reduce_sequence_loss`、`global_position_ids`、`pad/unpad`、`sp_group_info` 这类边界清晰的 API。
- USP 接入点不清楚。需要列出 DiT attention 的实际模块路径、QKV layout、RoPE/QK norm 顺序、attention mask 语义、dtype policy，否则“复用 patch_usp_attention”仍停留在方向层。
- 反向通信计划缺失。USP 前向可跑不代表反向高效；需要确认 backward 的 all-to-all/ring 顺序、显存峰值、是否支持 checkpoint recompute。
- 性能闸的 baseline 不够严谨。“什么都不开 = 4 卡 DDP”可能不是当前训练后端真实可用路径；若当前后端默认 FSDP2，DDP baseline 会混入实现差异。建议主对照用当前生产 DP/FSDP 配方。
- 10 步 RL perf 可能噪声过大。rollout、reward、训练 step 混合后，USP 性能会被非训练瓶颈掩盖；需要拆分 train-only microbench、end-to-end RL、weight sync 三类指标。
- checksum 验证粒度不足。还需要每个 rollout rank 的模块级 checksum、版本号、dtype、shape、更新耗时、发送源 rank 日志，避免只在一个 rank 上看起来正确。
- `dp=1, sp=4` 的训练统计风险没有讨论。RL advantage normalization、batch diversity、gradient noise scale 可能依赖 DP 样本数；这档配置虽能测容量，但未必能代表训练质量。

ALTERNATIVE_DIRECTIONS:
- 先做“最小 SP 垂直切片”，暂不删除 LLM 残留：只解锁 `cp_size`、建立 SP group、在一个 DiT attention block 上接 USP、跑 forward/backward 数值对照。优点是最快暴露核心风险；缺点是代码里仍有旧 CP 噪音。
- 将 `cp` 正名为 `sp` 但保留兼容 alias：配置层暴露 `sequence_parallel_size`，内部短期映射到 `context_parallel_size`。优点是减少概念债；缺点是迁移配置、脚本、日志需要小心。
- FSDP mesh 保守化：FSDP 只接收 shard 子 mesh，SP 独立 process group 由 attention 层使用，参数在 SP 维自然复制。优点是降低 FSDP2 多维 mesh 不确定性；缺点是 state dict/weight sync 需要手动理解两个 group。
- 训练和 rollout 先不强制共用同一 USP 实现，而是共用测试向量和误差门。优点是避免把 rollout 桩强行拉进训练 autograd；缺点是后续维护两套 kernel adapter。
- 对视频长序列先评估 activation checkpoint + tiled attention + selective recompute。优点是实现风险低，可能覆盖中等长度视频；缺点是吞吐可能下降，无法像 SP 一样扩展上限。
- Mooncake 不应只作为“SP 不行就上”的单一路线。更现实的回退分支是：训练侧用 checkpoint/offload，rollout 侧再评估 Mooncake/KVCache 解耦。两者解决的瓶颈可能不同。

QUESTIONS_FOR_USER:
- 目标视频模型的最小规格是什么：帧数、分辨率、latent patch size、num_heads、head_dim、训练 dtype？这决定 SP degree 和 USP 可行性。
- 4 张训练卡上首选拓扑是 `dp=2,sp=2` 还是 `dp=1,sp=4`？前者更像训练，后者更看容量。
- 是否接受阶段 0 从“删除”降级为“隔离 + 测试防回归”，把删除推迟到 SP 跑通之后？
- 精度对齐目标是逐算子对齐、单步 loss/grad 对齐，还是 rollout 轨迹统计对齐？不同目标对应完全不同的 tolerance。
- rollout 是否必须同时开启 `sp_degree>1`？如果 rollout 暂时保持非 SP，权重同步和 sglang 侧风险会小很多。
- Mooncake 的目标瓶颈到底是训练显存、rollout 显存、权重同步开销，还是端到端吞吐？需要先定瓶颈再定回退方案。

CANDIDATE_CRITERIA:
- `cp_size=1` 现有 diffusion 训练在固定种子下通过回归：loss、grad norm、reward、weight checksum 在设定 tolerance 内一致；不要求逐位一致，除非当前栈已证明确定性。
- `sp_size>1` 单 block attention forward/backward 对非 SP 参考实现通过：输出、输入梯度、QKV/proj 权重梯度均满足 dtype 对应 tolerance。
- 全 actor 训练 step 对照通过：同 batch、同 timestep、同噪声下，SP 与非 SP 的 loss/advantage/log_prob/grad norm 偏差在明确阈值内。
- 支持至少两种拓扑：`dp=2,sp=2` 和 `dp=1,sp=4`，并在启动时校验 head、sequence、world size、ulysses/ring 配置合法性。
- 权重同步验收：每次更新带版本号；所有 rollout rank 的模块级 shape/dtype/checksum 与训练源一致；SP rank 不发生重复无意义发送或有明确去重策略。
- 性能验收拆三层：train-only step、rollout-only、end-to-end RL。SP 必须证明更长序列容量，且在目标长序列下峰值显存显著下降、通信占比可解释。
- 可观测性验收：日志中记录 dp/sp rank、mesh、USP degree、padding 长度、通信耗时、weight sync 源 rank、checksum 摘要；NCCL hang 时能定位到 collective 阶段。
- 回退判据验收：若 SP 容量提升低于 1.5×、显存下降低于 30%、或端到端吞吐不可接受，则进入 Mooncake/卸载专项评估，而不是继续堆 SP 优化。
