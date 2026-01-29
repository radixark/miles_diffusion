# Flow-GRPO in slime: Prototype Plan

## 目标与范围
- 目标：在 `slime` 内部搭建一个最小可跑的 Flow-GRPO 训练原型，复现论文中单一模型 + 单一数据集的实验趋势即可。
- 范围：只做一个模型、一个数据集；忽略多模型/多任务兼容性；以可跑通闭环为第一优先。
- 训练后端：固定使用 FSDP（slime 的 `fsdp_utils` 路径），不考虑 Megatron 或其他后端。

## 选定的模型与数据集（用于最初版本）
- 模型：`stabilityai/stable-diffusion-3.5-medium`（与 flow_grpo 默认实验一致，diffusers 已有补丁实现 log-prob）。
- 数据集：`flow_grpo/dataset/pickscore_sfw`（本地 prompt 列表 + pickscore reward，可在本地完成评估）。
- Reward：`PickScore`（`flow_grpo/flow_grpo/rewards.py`），避免依赖远程 reward 服务。

## 方法论（为什么这样做）
- 以 `flow_grpo` 的训练范式为准绳，先抽象“最小环路”四件事：
  1) 生成轨迹并记录 log-prob；2) 计算 reward；3) 计算优势/裁剪；4) 更新策略。
- 以 `slime` 的结构为骨架，优先复用其数据流与训练编排（rollout manager + data buffer + train actor），只为 diffusion 增加专用的 rollout/train 插件，保证与现有系统自然融合、可扩展。
- 先做“可比对验证”的最小闭环，再扩展到更多模型/数据集，降低集成风险。

## 与 slime 结构结合的设计要点
- 复用 `slime` 的 **RolloutDataSource**（`slime/slime/rollout/data_source.py`）做 group sampling：
  - 利用 `n_samples_per_prompt` 实现 per-prompt 的组采样，与 Flow-GRPO 的 group advantage 对齐。
- 复用 `slime` 的 **Sample** 结构（`slime/slime/utils/types.py`）：
  - 在 `Sample.metadata` 或 `Sample.train_metadata` 内存储 diffusion 轨迹信息（latents/timesteps/log_probs/prev_latents_mean 等）。
  - 这样不破坏现有接口，也便于未来扩展到其他 diffusion 任务。
- 新增 **diffusion rollout** 与 **diffusion train actor**，固定挂载到 FSDP 训练链路（`slime/slime/backends/fsdp_utils`），保持与 `slime` 的 Ray/placement_group 体系一致，方便复用分布式和 tracking 体系。

## 分阶段计划（每阶段都有验证方式）

### 阶段 1：训练范式对齐与接口设计
**目标（更细化）**：形成一份可执行的“数据流与接口规范”，确保在 slime + FSDP 架构内，Flow-GRPO 的核心计算（log-prob / reward / advantage / ratio / clip）可以一一落地，不存在字段缺失或语义不一致。
- 具体工作（细化步骤）：
  - 1) 梳理 Flow-GRPO 最小闭环的“必需张量”：
    - 从 `flow_grpo/scripts/train_sd3.py` 抽出训练所需的最小张量清单（latents 轨迹、timesteps、log_prob、prev_latents_mean、reward、advantage）。
    - 从 `flow_grpo/flow_grpo/diffusers_patch/*_with_logprob.py` 明确 log-prob 的计算时点与形状。
  - 2) 建立 slime 侧的“字段映射表”：
    - 明确每个字段落在 `Sample.metadata` 或 `Sample.train_metadata`，以及是否需要序列化进 rollout buffer。
    - 给出每个字段的 shape/dtype/对齐维度（prompt 维度 vs 轨迹时间步维度）。
  - 3) 定义最小可跑的接口契约：
    - rollout 产出：`image`, `log_prob_old`, `timesteps`, `latents`, `prev_latents_mean`, `prompt_id/group_id`。
    - training 消耗：`reward`, `advantage`, `log_prob_new`, `log_prob_old`。
  - 4) 组采样与 advantage 归一化策略对齐：
    - 复用 `n_samples_per_prompt`，明确 “组内归一化/全局归一化”的开关与默认行为。
    - 指定 reward 聚合和 advantage 广播的规则（每条轨迹对齐到每个时间步）。
  - 5) 输出一份“接口规格表”与“最小数据流图”：
    - 字段名、shape、dtype、来源模块、使用模块、序列化要求。
    - 简化数据流图：Prompt -> Rollout -> Reward -> Advantage -> Policy Loss。
- 验证方式（更细化）：
  - 1) 公式覆盖验证：
    - 用纸面/表格校验 `ratio = exp(log_prob_new - log_prob_old)` 和 `clip` 计算所需字段齐全、维度可广播。
  - 2) 维度一致性验证：
    - 对每个字段的 shape 做矩阵检查，确保（batch, steps, ...) 与 advantage 广播一致。
  - 3) 语义一致性验证：
    - 对照 `flow_grpo/scripts/train_sd3.py` 的损失公式逐项匹配字段语义（“old log-prob”一定来自 rollout， “new log-prob”来自当前模型）。
  - 4) FSDP 兼容性确认：
    - 检查接口中所有可训练张量仅在训练 actor 中出现，rollout 侧只保留必要轨迹信息，避免跨进程共享大张量导致不可控内存开销。

### 阶段 2：Rollout 原型（生成 + log-prob + reward）
**目标**：在 slime 中完成 diffusion 轨迹采样并得到 reward。
- 具体工作：
  - 将 `flow_grpo/flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py` 作为核心生成器。
  - 创建 diffusion rollout 实现（对齐 `slime/slime/rollout/sglang_rollout.py` 的接口模式）。
  - 生成流程：prompt -> logprob 轨迹 -> 图像 -> reward。
  - reward 直接复用 `flow_grpo/flow_grpo/rewards.py`。
- 验证方式：
  - 小批量（例如 4 prompts * 2 samples）生成，检查：
    - 每个 sample 具备完整的轨迹信息（timesteps/log_probs/latents 数量一致）。
    - reward 有数值输出且分布合理（非全 0 / 非 NaN）。
  - 与 `flow_grpo/scripts/train_sd3.py` 在相同 seed 下比对单次生成结果的统计分布（不要求像素级一致）。

### 阶段 3：Flow-GRPO 训练最小闭环（FSDP 固定）
**目标**：实现核心 loss（重要性权重 + clip + advantage）并能跑一个 epoch。
- 具体工作：
  - 基于 slime 的 FSDPTrainRayActor（`slime/slime/backends/fsdp_utils/actor.py`）实现 diffusion 训练 actor（不需要 critic）。
  - 对照 `flow_grpo/scripts/train_sd3.py` 复刻：
    - ratio = exp(log_prob - old_log_prob)
    - clip 策略 + advantage（按 prompt 组归一化）
  - 先仅支持单机单卡，后续再扩展分布式。
- 验证方式：
  - 在固定 batch 上对比 flow_grpo 脚本的 loss 数值量级（同输入的数值差异应可解释）。
  - 单步训练后检查模型参数发生更新，loss 不为 NaN。

### 阶段 4：端到端原型验证
**目标**：运行一个小规模训练并验证 reward 改善趋势。
- 具体工作：
  - 配置最小训练循环（例如 1-2k steps）并记录 reward 曲线。
  - 输出固定 prompt 的采样图像用于 qualitative 对比。
- 验证方式：
  - reward 曲线应呈现总体上升趋势（即使幅度较小）。
  - 训练前后在 pickscore 评估集上对比平均分值变化。

### 阶段 4.5（正确性验证）：最小关键指标补齐
**目标**：先补齐最关键的正确性指标，验证训练信号与 Flow‑GRPO 对齐，再进入全面指标对齐。
- 具体工作：
  - **A 组：Reward 与 Advantage 正确性（基础信号）**
    - **Reward 结构对齐（multi_score）**：
      - 支持 `config.reward_fn` 的 dict 形式（如 `{"ocr": 1.0, "pickscore": 0.5}`）。
      - 计算加权平均得到 `reward_avg`，并保留各子 reward（如 `reward_ocr`）。
      - 说明：只有做了 multi_score 加权，`reward_avg` 才与 Flow‑GRPO 同义。
    - **组内归一化统计**：
      - 记录 `reward_std_mean`, `zero_std_ratio`, `group_size` 用于判断 advantage 是否有效。
  - **B 组：训练更新是否发生（核心日志）**
    - 记录 `approx_kl`, `clipfrac`, `loss`
    - 记录 `epoch`（暂以 rollout_id 代替），保证时间轴可对齐
  - **C 组：时间轴/命名对齐（防止误读）**
    - 统一关键指标的 W&B key 名，不加 `diffusion_train/` 前缀
    - 将日志 step 对齐到 rollout_id（或显式记录 `epoch` 字段）
- 验证方式：
  - 1) reward 量级回到与 Flow‑GRPO 相近（~0.3+ 起步）。
  - 2) `clipfrac` 与 `approx_kl` 出现非零波动。
  - 3) `reward_std_mean` 非极小，`zero_std_ratio` 不接近 1。

### 阶段 5（扩展）：指标体系对齐与日志可视化
**目标**：把 Miles 的 diffusion 训练日志对齐到 Flow‑GRPO 的指标体系，确保 W&B 面板可直接对比。
- 具体工作：
  - **训练侧指标补齐**（与 Flow‑GRPO key 名对齐）：
    - `loss`, `policy_loss`, `kl_loss`, `approx_kl`, `clipfrac`, `clipfrac_gt_one`, `clipfrac_lt_one`
    - `epoch`, `inner_epoch`, `actual_batch_size`
  - **rollout 侧指标补齐**：
    - `reward_avg`, `reward_ocr`, `reward_ori_avg`（reward dict 统一输出 avg/ocr 等）
    - `group_size`, `trained_prompt_num`, `reward_std_mean`, `zero_std_ratio`
  - **评估侧指标补齐**：
    - `eval_images`, `eval_reward_avg`, `eval_reward_ocr`
  - **图片日志对齐**：
    - 训练阶段图像统一用 `images`，评估阶段用 `eval_images`。
  - **命名规范**：
    - 停止使用 `diffusion_train/*` 作为前缀，直接采用 Flow‑GRPO 的 key 名。
- 验证方式：
  - 1) W&B Summary 里 `keys` 数量接近 Flow‑GRPO（~21+）。
  - 2) Flow‑GRPO 与 Miles 同步跑时，metric 名称与语义一一对应。

### 阶段 6（扩展）：clipfrac=0 的问题诊断与修复
**目标**：让 `clipfrac` 与 `approx_kl` 出现非零波动，确保策略发生可观更新。
- 具体工作：
  - 复刻 Flow‑GRPO 的 advantage 归一化策略：
    - per‑prompt 归一化 `(r - mean) / (std + eps)`
    - 输出 `reward_std_mean`, `zero_std_ratio` 与 `group_size`
  - 引入 reward dict（`multi_score`）：
    - 确保 reward 的尺度与 Flow‑GRPO 一致（特别是 `avg`/`ocr`）
  - 观察并必要时调整：
    - `diffusion_clip_range`（必要时减小到 0.1）
    - `lr`（必要时适度增大）
- 验证方式：
  - 1) `clipfrac` 偶尔 > 0；`approx_kl` 不再为 0。
  - 2) reward 分布有可解释变化。

## 扩展性设计（为后续多模型/多任务做准备）
- 统一的 diffusion rollout 接口，允许替换 pipeline（SD3 / Flux / QwenImage）而不改训练逻辑。
- reward 采用 registry 方式封装（复用 `flow_grpo/flow_grpo/rewards.py`），方便扩展到 OCR/Geneval。
- group sampling 与 per-prompt stats 复用 slime 数据源结构，不需另起系统。

## 最终交付的原型形态
- slime 内新增一个 diffusion RL 训练入口（参数最少、单机可跑），且固定 FSDP 后端。
- 能使用 SD3.5-M + pickscore_sfw 复现 Flow-GRPO 实验趋势。
- 训练/评估结果有可追踪的日志与 reward 曲线。
