# Diffusion 集成阶段问题与修复记录

> 目的：统一记录阶段性 bug、根因、修复方法、以及相关代码改动位置。

## 1) Rollout 在 CPU 上导致 fp16 失败
**现象**
- `Pipelines loaded with dtype=torch.float16 cannot run with cpu`

**根因**
- RolloutManager 默认 `num_gpus=0`，diffusion rollout 在 CPU 上执行。
- fp16 pipeline 无法在 CPU 上运行。

**修复方法论**
- 让 RolloutManager 在 diffusion rollout 模式下申请 1 张 GPU，保证推理运行在 GPU。

**代码改动**
- `miles/ray/placement_group.py`
  - `create_rollout_manager()` 中检测 `diffusion_rollout` 时给 RolloutManager 分配 GPU。

---

## 2) fp16 权重 vs fp32 latents 的 dtype mismatch
**现象**
- `Input type (float) and bias type (c10::Half) should be the same`

**根因**
- `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py` 内部强制 `latents.float()`，导致输入 fp32，但 pipeline 权重是 fp16。

**修复方法论**
- 先用参数规避：rollout 改 fp32，保证 dtype 一致（不修改 flow_grpo）。

**代码改动**
- `scripts/run-diffusion-grpo-ocr.sh`
  - 新增 `--diffusion-dtype fp32`。

---

## 3) sglang 引擎抢显存导致训练 OOM
**现象**
- 训练阶段 OOM，日志显示 `SGLangEngine` 与 `KV Cache is allocated`。

**根因**
- sglang engine 会按 GPU 余量预分配大块 KV cache，吞掉显存。
- diffusion rollout 不需要 sglang，但默认逻辑仍可能启动。

**修复方法论**
- 在代码层面强制 diffusion rollout 跳过 sglang 引擎初始化，避免误启动。

**代码改动**
- `miles/ray/rollout.py`
  - `RolloutManager.__init__` 中增加 `diffusion_rollout` 判断，不初始化引擎。
  - `init_rollout_engines()` 内增加 diffusion guard，直接返回 0。

---

## 4) 训练阶段 OOM（micro-batch 过大）
**现象**
- 训练一开始 OOM。

**根因**
- `diffusion_train_batch_size` 未设置时默认等于整批大小（如 128），激活太大。

**修复方法论**
- 降低训练 micro-batch，保持 rollout 样本数不变但分批训练。

**代码改动**
- `scripts/run-diffusion-grpo-ocr.sh`
  - 新增 `--diffusion-train-batch-size 1`。

---

## 4.5) Diffusion 仍依赖 hf-checkpoint 的 tokenizer
**现象**
- 不传 `--hf-checkpoint` 会直接失败，即使 diffusion rollout 实际不需要 tokenizer。

**根因**
- `RolloutDataSource` 里无条件加载 tokenizer（依赖 `args.hf_checkpoint`），导致 diffusion-only 模式也被迫提供文本模型。

**当前规避**
- 脚本里传 `--hf-checkpoint gpt2` 仅用于初始化，不影响 diffusion 采样。

**建议修复方向**
- 当启用 diffusion 模式时，允许 `RolloutDataSource` 跳过 tokenizer / processor 的加载。
- 或者提供一个不依赖 tokenizer 的 diffusion 专用 data source。

---

## 5) Scheduler device mismatch（CPU/GPU 混用）
**现象**
- `Expected all tensors to be on the same device, cuda:0 and cpu!`

**根因**
- 训练侧手动调用 `sde_step_with_logprob`，但 scheduler 的 `timesteps/sigmas` 仍在 CPU。

**修复方法论**
- 训练前显式把 scheduler 的 `timesteps/sigmas` 迁到 GPU。

**代码改动**
- `miles/backends/fsdp_utils/diffusion_actor.py`
  - `_compute_log_prob_new()` 内将 scheduler 的 `timesteps/sigmas` 移到 `latents.device`。

---

## 6) index_for_timestep 找不到索引（timesteps dtype 错）
**现象**
- `IndexError: index 0 is out of bounds for dimension 0 with size 0`

**根因**
- 训练侧把 `timesteps` 强制转成 `long`，而 scheduler 的 timesteps 是浮点序列，导致匹配为空。

**修复方法论**
- 保持 `timesteps` 为 float，并对齐到 scheduler 的 dtype；同时在训练前调用 `set_timesteps`。

**代码改动**
- `miles/backends/fsdp_utils/diffusion_actor.py`
  - `_compute_log_prob_new()` 内调用 `scheduler.set_timesteps(steps, device=device)`，并对齐 dtype。
  - `train()` 中将 `timesteps` 转为 `float32`（不再用 `long`）。

---

## 7) offload 时调用 pipeline.cpu() 报错
**现象**
- `AttributeError: 'StableDiffusion3Pipeline' object has no attribute 'cpu'`

**根因**
- diffusers 的 pipeline 没有 `.cpu()` 方法，只支持 `.to(device)`。

**修复方法论**
- 用 `pipeline.to(\"cpu\")` 替代 `pipeline.cpu()`。

**代码改动**
- `miles/backends/fsdp_utils/diffusion_actor.py`
  - `sleep()` 内将 `self.pipeline.cpu()` 改为 `self.pipeline.to(\"cpu\")`。

---

## 8) W&B init 缺失导致 wandb.log 报错
**现象**
- `wandb.errors.Error: You must call wandb.init() before wandb.log()`

**根因**
- Diffusion 训练 actor 没有在 rank0 初始化 W&B，但仍调用了 `tracking_utils.log`。

**修复方法论**
- 在 `DiffusionFSDPTrainRayActor.init()` 中对 rank0 调用 `init_tracking(args, primary=False)`。

**代码改动**
- `miles/backends/fsdp_utils/diffusion_actor.py`
  - `init()` 内新增 `init_tracking(args, primary=False)`。

---

## 补充：flow_grpo 不触发部分 bug 的原因
- flow_grpo 推理走完整 pipeline，内部自动管理 scheduler 的 device/dtype。
- Miles 训练端为重算 `log_prob_new`，需要手动调用 `sde_step_with_logprob`，因此必须显式处理 device/dtype。
