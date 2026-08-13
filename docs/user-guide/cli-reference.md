---
title: CLI Reference
description: Every flag train_diffusion.py accepts, grouped by subsystem.
---
miles-diffusion is configured entirely through flags on `train_diffusion.py`. They come from
three places:

1. **Training-backend flags** — the `FSDPArgs` dataclass in
   `miles/backends/fsdp_utils/arguments.py`. Every field becomes `--field-name` automatically.
2. **miles flags** — `miles/utils/arguments.py`, added on top as an `extra_args_provider`.
3. **sglang-d passthrough** — the rollout engine's own CLI arguments, re-registered with a
   `--sglang-` prefix (`--sglang-attention-backend`, `--sglang-dit-precision`,
   `--sglang-vae-slicing`, …). A short skip list covers the ones miles sets itself: `model_path`,
   ports, `base_gpu_id`, `random_seed`, and a few others.

`--config <file.yaml>` loads extra keys into the namespace, but only ones argparse did *not*
already define — it cannot override a flag you passed.

<Note>

**Reading the prefixes.** `diffusion-` marks the modality — denoising, SDE, CFG, latents, frames —
so that a CLI merged with miles LLM says what a flag is *for*. Generic ML/RL concepts (clipping,
KL, EMA, LoRA, batching) do not take it. `fsdp-` and `rollout-` mark the *side*: compare
`--fsdp-flow-shift` (training-side sigma grid) with `--diffusion-flow-shift` (the engine's
generation schedule). The prefix does not tell you which argument group a flag lives in — groups
follow concern, not name.

</Note>

```bash
python3 train_diffusion.py --help
```

is always the ground truth. This page has two passes: **Essentials** for the flags most runs
touch, then the **complete reference** by group.

---

## Essentials

### The one required flag

| Flag | What |
|---|---|
| `--hf-checkpoint` | The diffusers pipeline to train, as an HF repo id or a local directory. |

One value serves three readers, so they cannot disagree: the training side loads components and
scheduler from it, the sglang-d engine serves it, and the **model family** is matched from its
name. Override the last with `--diffusion-model-family` when your checkpoint's name carries no
family hint — which local weights usually do not.

### Cluster topology

| Flag | Default | What |
|---|---|---|
| `--actor-num-nodes` | `1` | Nodes for the training actor. |
| `--actor-num-gpus-per-node` | `8` | GPUs per actor node. Train world size is the product. |
| `--rollout-num-gpus` | `None` | GPUs for rollout-side work. Forced to the train world size under `--colocate`. |
| `--rollout-num-gpus-per-engine` | `1` | GPUs per sglang-d engine (its TP × SP). |
| `--num-gpus-per-node` | `8` | Total GPUs the job may use per node. **Set this when using fewer than 8.** |
| `--colocate` | off | Time-share GPUs between trainer and engines. Forces `--offload-train` and `--offload-rollout` on. |

### Batch sizing

One identity, checked at parse time — pass **one** of the two, never both:

```
global_batch_size = rollout_batch_size × n_samples_per_prompt ÷ num_steps_per_rollout
```

| Flag | Default | What |
|---|---|---|
| `--rollout-batch-size` | int, unset | Prompts per rollout. No default — every recipe sets it. |
| `--n-samples-per-prompt` | `1` | Samples per prompt (GRPO group size). |
| `--global-batch-size` | derived | **Samples** per optimizer step. Must divide by `dp_size`. |
| `--num-steps-per-rollout` | derived | Optimizer steps per rollout. |
| `--micro-batch-size` | `1` | **Train pairs** per DiT forward (flat cut). |
| `--micro-batch-size-sample` × `--micro-batch-size-tstep` | `None` | 2D tile per DiT forward. Set together; overrides `--micro-batch-size` with their product. |
| `--num-rollout` | from dataset | Total rollout iterations. |
| `--num-epoch` | `None` | Alternative to `--num-rollout`; ignored if both are set. |

`global_batch_size` counts samples; the micro-batch knobs count train pairs (one sample yields one
pair per trained denoising step). See
[the arithmetic](/user-guide/training-script-walkthrough#3-the-batch-size-arithmetic).

### Diffusion sampling

| Flag | Default | What |
|---|---|---|
| `--diffusion-num-steps` | `10` | Denoising steps per rollout sample. |
| `--diffusion-num-sde-steps` | `0` | How many steps become train pairs. `0` disables. |
| `--diffusion-step-strategy-path` | `None` | Which steps. Overrides the bare count. |
| `--diffusion-noise-level` | `0.7` | SDE noise injected during rollout. |
| `--diffusion-guidance-scale` | `4.0` | CFG scale. |
| `--diffusion-height` / `--diffusion-width` | `512` | Output resolution. |
| `--diffusion-output-num-frames` | `1` | Frames — `1` for images. |
| `--rollout-microgroup-size` | `1` | Samples per prompt sent in one `POST /rollout/generate`. |

### Precision

| Flag | Default | What |
|---|---|---|
| `--diffusion-forward-dtype` | `bf16` | DiT forward compute — engine, FSDP `param_dtype`, and the training-side input cast. |
| `--fsdp-master-dtype` | `fp32` | Sharded master weights and optimizer state. |
| `--fsdp-reduce-dtype` | `fp32` | Gradient reduce-scatter. |

See [Dtype Control](/advanced/dtype-control).

### Algorithm

| Flag | Default | What |
|---|---|---|
| `--loss-type` | `policy_loss` | `policy_loss` (Flow-GRPO), `nft`, `sft_loss`, `custom_loss`. |
| `--advantage-estimator` | `grpo` | Only `grpo` today. |
| `--diffusion-clip-range` | `1e-4` | PPO ratio clip. |
| `--diffusion-adv-clip-max` | `5.0` | Advantage clamp. |
| `--diffusion-kl-beta` | `0.0` | Reference-KL coefficient; `> 0` turns on a reference forward. |
| `--globalize-reward-std` | off | Batch-wide std instead of per-group. |
| `--globalize-reward-mean` | off | Batch-wide mean instead of per-prompt. |

---

## Complete reference

### Cluster

| Flag | Type / default | Notes |
|---|---|---|
| `--actor-num-nodes` | int, `1` | |
| `--actor-num-gpus-per-node` | int, `8` | |
| `--rollout-num-gpus` | int, `None` | For train-only SFT, leave unset to colocate encoders with training, or set it to reserve dedicated encoder GPUs. |
| `--rollout-num-gpus-per-engine` | int, `1` | Like sglang's `tp_size`. |
| `--num-gpus-per-node` | int, `8` | |
| `--colocate` | flag | Also sets `--offload`. |
| `--offload` | flag | `--offload-train` + `--offload-rollout`. |
| `--offload-train` / `--no-offload-train` | tri-state | Always on under `--colocate`. |
| `--offload-rollout` / `--no-offload-rollout` | tri-state | Always on under `--colocate`. |
| `--distributed-backend` | str, `nccl` | |
| `--distributed-timeout-minutes` | int, `10` | |

### Training backend

| Flag | Type / default | Notes |
|---|---|---|
| `--train-backend` | `fsdp` | Only value. |
| `--fsdp-master-dtype` | `fp32` \| `bf16` \| `fp16` | Load, shard, and optimizer-state precision. |
| `--fsdp-reduce-dtype` | `fp32` \| `bf16` \| `fp16` | `bf16` matches flow_grpo's all-bf16 policy but adds cross-rank add-noise. |
| `--diffusion-forward-dtype` | `bf16` \| `fp16` \| `fp32` | |
| `--fsdp-cpu-offload` | flag | Offloads params, grads, optimizer state; the optimizer then runs on CPU. |
| `--fsdp-cpu-backend` | str, `gloo` | CPU process group for the above. |
| `--dp-replicate-size` | int, `1` | Hybrid sharding: replica count. `dp_shard` takes the ranks left over from this and SP. |
| `--sequence-parallel-size` | int, `1` | USP = Ulysses × Ring. |
| `--ulysses-degree` | int, `0` | `0` = auto (Ulysses fills SP). Ring degree > 1 needs torch ≥ 2.11 and a ring-capable attention backend. |
| `--fsdp-attention-backend` | str, `None` | diffusers `set_attention_backend` value. |
| `--fsdp-flow-shift` | float, `None` | Training-side sigma grid shift, regenerated when no engine supplies scheduler meta (SFT). Distinct from `--diffusion-flow-shift`. |
| `--gradient-checkpointing` | flag | |
| `--deterministic-mode` | flag | See [Deterministic Training](/advanced/deterministic). |
| `--train-env-vars` | JSON, `{}` | Extra env for the training processes. |

### Optimizer and schedule

| Flag | Type / default | Notes |
|---|---|---|
| `--optimizer` | `adam` | AdamW. Only value. |
| `--lr` | float, `1e-6` | |
| `--adam-beta1` / `--adam-beta2` | `0.9` / `0.999` | β₂ matches flow_grpo, not the LLM-side `0.95`. |
| `--adam-eps` | `1e-8` | |
| `--weight-decay` | `0.0` | |
| `--clip-grad` | `1.0` | |
| `--lr-decay-style` | `constant` | |
| `--min-lr`, `--lr-warmup-init` | `0.0` | |
| `--lr-warmup-iters` | int, `0` | |
| `--lr-warmup-fraction` | float, `None` | |
| `--lr-decay-iters` | int, `None` | |
| `--lr-wsd-decay-iters`, `--lr-wsd-decay-style` | `None` | Warmup-stable-decay. |
| `--use-checkpoint-lr-scheduler` | flag, on | |
| `--override-lr-scheduler` | flag | |
| `--seed` | int, `1234` | |

### Rollout

| Flag | Type / default | Notes |
|---|---|---|
| `--hf-checkpoint` | str | **Required.** Pipeline to train and to serve; also the family source. |
| `--diffusion-model-family` | str, `None` | Registered family key: `sd3`, `wan2_2`, `ltx`, `qwen_image`. Overrides name matching. |
| `--rollout-function-path` | str | Use `miles.rollout.sglang_diffusion_rollout.generate_rollout`. |
| `--train-pipeline-config-path` | str, `None` | Your own `TrainPipelineConfig` for an unregistered family. Mutually exclusive with `--diffusion-model-family`. |
| `--model-backend-path` | str, `None` | Override the family's model loader. |
| `--diffusion-num-steps` | int, `10` | |
| `--diffusion-flow-shift` | float, `None` | Generation-schedule shift, sent to the engine. |
| `--rollout-microgroup-size` | int, `1` | |
| `--diffusion-fps` | float, `None` | Video only. |
| `--diffusion-output-num-frames` | int, `1` | |
| `--diffusion-guidance-scale` | float, `4.0` | |
| `--diffusion-guidance-scale-2` | float, `None` | Wan2.2 low-noise expert; **required** when training it. |
| `--diffusion-true-cfg-scale` | float, `None` | |
| `--diffusion-negative-prompt` | str, `None` | Defaults to `" "` on the engine when CFG is on. |
| `--diffusion-noise-level` | float, `0.7` | |
| `--diffusion-height` / `--diffusion-width` | int, `512` | Rollout output size; SFT center-crop size. |
| `--diffusion-sde-type` | `sde` \| `cps` \| `ode` | Selects the train-side SDE backend too. |
| `--sde-step-backend-path` | str, `None` | Custom dynamics. See [SDE backends](#sde-step-backends). |
| `--diffusion-num-sde-steps` | int, `0` | |
| `--diffusion-sde-window-range` | `"lo,hi"`, `None` | For `sde_window`. Defaults to `[0, num_inference_steps)`. |
| `--diffusion-sde-candidate-steps` | `"1,2,3"`, `None` | Required by `epoch_global_random_choice`. |
| `--diffusion-step-strategy-path` | str, `None` | Overrides the bare `--diffusion-num-sde-steps` selection. |
| `--diffusion-log-prob-no-const` | flag | Drop log-prob constants on the engine (pairs with the CPS backend). |
| `--diffusion-generator-device` | str, `cuda` | |
| `--rollout-patch-group` | str, `None` | Comma-separated numeric-parity patch groups, e.g. `sgld`, `ltx`. |
| `--update-weight-target-module` | str, `transformer` | Modules to train and sync. Wan2.2: `transformer,transformer_2`. |
| `--update-weight-buffer-size` | int, `512 MiB` | Weight-sync chunk size in bytes. |
| `--rollout-seed` | int, `42` | |
| `--over-sampling-batch-size` | int, `None` | Must equal `--rollout-batch-size` today. |
| `--sglang-server-concurrency` | int, `512` | Per-engine in-flight request cap. |
| `--use-miles-router` | flag | **Required** — the SGLang router is not supported here. |
| `--miles-router-timeout` | float, `None` | |
| `--miles-router-max-connections` | int, `None` | |
| `--miles-router-health-check-failure-threshold` | int, `3` | |

### Data and batching

| Flag | Type / default | Notes |
|---|---|---|
| `--prompt-data` | str | jsonl, one row per prompt. |
| `--input-key` | str, `input` | |
| `--metadata-key` | str, `metadata` | |
| `--data-source-path` | str | Defaults to `RolloutDataSourceWithBuffer`. |
| `--disable-rollout-global-dataset` | flag | Manage data yourself. |
| `--rollout-batch-size` | int, unset | |
| `--n-samples-per-prompt` | int, `1` | |
| `--global-batch-size` | int, derived | |
| `--num-steps-per-rollout` | int, derived | |
| `--micro-batch-size` | int, `1` | Flat: train-pair dicts per DiT forward, contiguous within an optimizer window. |
| `--micro-batch-size-sample` | int, `None` | 2D: samples per DiT-forward tile. |
| `--micro-batch-size-tstep` | int, `None` | 2D: SDE timesteps per tile. Set with the above. |
| `--diffusion-train-iter-order` | `sample_major` \| `timestep_major` | Tile visit order; only meaningful with the 2D pair. |
| `--train-dp-split-mode` | `contiguous` \| `stride` | `contiguous` lets a micro-batch reproduce a rollout microgroup exactly; `stride` deals round-robin. |
| `--num-rollout` | int, `None` | |
| `--num-epoch` | int, `None` | |
| `--start-rollout-id` | int, `None` | Resumed from `--load` when unset. |
| `--sft-encoder-checkpoint` | str, `None` | SFT only: tokenizer/text_encoder/vae source. |
| `--sft-frame-stride` | int, `1` | SFT encode temporal stride. |

### Evaluation

| Flag | Type / default | Notes |
|---|---|---|
| `--eval-interval` | int, `None` | Requires configured eval datasets. |
| `--eval-prompt-data` | `<name> <path>` … | Repeatable pairs. |
| `--eval-config` | str, `None` | OmegaConf YAML/JSON; overrides `--eval-prompt-data`. |
| `--eval-function-path` | str, `None` | Defaults to `--rollout-function-path`. |
| `--eval-input-key` | str, `None` | |
| `--n-samples-per-eval-prompt` | int, `1` | |
| `--diffusion-eval-num-steps` | int, `None` | Defaults to `--diffusion-num-steps`. |
| `--skip-eval-before-train` | flag | |
| `--eval-reward-key` | str, `None` | Defaults to `--reward-key`. |

### Algorithm and loss

| Flag | Type / default | Notes |
|---|---|---|
| `--loss-type` | `policy_loss` | `policy_loss` \| `nft` \| `sft_loss` \| `custom_loss`. |
| `--advantage-estimator` | `grpo` | |
| `--disable-grpo-std-normalization` | flag | Dr.GRPO. Also forced off when `n_samples_per_prompt == 1`. |
| `--globalize-reward-mean` | flag | Leave **off** for flow_grpo parity. |
| `--globalize-reward-std` | flag | **On** for flow_grpo's PickScore recipe. |
| `--diffusion-clip-range` | float, `1e-4` | |
| `--diffusion-adv-clip-max` | float, `5.0` | Under `nft` this also sets the advantage-to-`r` slope. |
| `--diffusion-recompute-old-log-prob` | flag | Recompute old log-probs with the trainer forward instead of trusting the rollout's. policy_loss only. |
| `--diffusion-kl-beta` | float, `0.0` | |
| `--ref-mode` | `none` \| `lora_base` \| `ema` | Auto: `lora_base` when KL > 0, `ema` under `nft`. |
| `--custom-prepare-train-batch-path` | str, `None` | Builds DiT inputs. |
| `--custom-loss-function-path` | str, `None` | Loss **formula** only — the DiT forward stays in the actor. |
| `--diffusion-nft-beta` | float, `1.0` | |
| `--diffusion-nft-timestep-fraction` | float, `0.99` | |
| `--no-diffusion-nft-adaptive-weight` | flag | |
| `--no-diffusion-nft-shuffle-timesteps` | flag | |

### Reward

| Flag | Type / default | Notes |
|---|---|---|
| `--rm-type` | str, `None` | `pickscore` \| `ocr`. Overridable per sample via `metadata.rm_type`. |
| `--reward-key` | str, `None` | When the reward is a dict. |
| `--group-rm` | flag | Score a whole prompt group at once. |
| `--custom-rm-path` | str, `None` | `async def rm(args, samples) -> list[float]`. Batched only. |
| `--custom-reward-post-process-path` | str, `None` | Replace advantage normalisation. |
| `--colocate-reward` | flag | Reward actors onto rollout GPUs (train 0.7 + rollout 0.25 + reward 0.05). Requires `--colocate`. |
| `--pickscore-model-path` | str | Required for `--rm-type pickscore`. |
| `--pickscore-processor-path` | str | Required for `--rm-type pickscore`. |
| `--pickscore-num-workers` | int, `1` | |
| `--pickscore-num-gpus-per-worker` | float, `1.0` | Fractional values allowed. |
| `--pickscore-batch-size` | int, `8` | |
| `--pickscore-num-frames` | int, `None` | Frames scored per video; `None` = all. |
| `--ocr-num-workers` | int, `4` | |
| `--rollout-parser-num-workers` | int, `1` | Ray actors deserializing rollout responses. Raise when trajectory tensors are large. |

### Rollout customization hooks

Every one takes a dotted path.

| Flag | Replaces |
|---|---|
| `--custom-generate-function-path` | The inner `generate` call in the diffusion rollout. |
| `--dynamic-sampling-filter-path` | Per-group keep/drop (DAPO-style). |
| `--rollout-sample-filter-path` | Per-sample loss mask; set `sample.remove_sample = True`. Samples still count for advantage normalisation. |
| `--buffer-filter-path` | Buffer selection. |
| `--custom-convert-samples-to-train-data-path` | The whole samples → train-data conversion. |
| `--custom-expand-samples-to-train-pairs-path` | Just the sample → train-pair expansion. |
| `--custom-rollout-log-function-path` | Train rollout logging. |
| `--custom-eval-rollout-log-function-path` | Eval rollout logging. |

### LoRA

| Flag | Type / default | Notes |
|---|---|---|
| `--use-lora` | flag | |
| `--lora-rank` / `--lora-alpha` | int, `64` / `64` | |
| `--lora-target-modules` | str…, `None` | Defaults per model family. |
| `--lora-init-weights` | str, `gaussian` | `kaiming-uniform` maps to PEFT's default; other PEFT schemes pass through. |
| `--lora-ipc-weight-sync` | flag | Push only `lora_A`/`lora_B`; the engine merges locally. Requires `--use-lora`. |

### EMA

| Flag | Type / default | Notes |
|---|---|---|
| `--use-ema` | flag | Maintains an EMA copy as πₒₗd. Needs a consumer (`--ref-mode ema` or `--ema-rollout-policy ema`). |
| `--ema-rollout-policy` | `live` \| `ema` | Which weights get pushed to rollout. |
| `--ema-decay-init` | float, `0.001` | Decay during the flat period. |
| `--ema-decay-ramp` | float, `0.001` | Per-step increase after the flat period; the ramp restarts from zero. |
| `--ema-decay-max` | float, `0.5` | Ceiling. |
| `--ema-decay-flat-steps` | int, `0` | |

### Checkpointing

| Flag | Type / default | Notes |
|---|---|---|
| `--save` | str, `None` | |
| `--save-interval` | int, `None` | Requires `--save`. |
| `--no-save-optim` | flag | Smaller checkpoints, but no resumption. |
| `--load` | str, `None` | |
| `--ckpt-step` | int, `None` | Defaults to `latest_checkpointed_iteration.txt`. |
| `--no-load-optim` / `--no-load-rng` | flag | |

### Logging

| Flag | Type / default | Notes |
|---|---|---|
| `--use-wandb` | flag | |
| `--wandb-project` | str, `None` | |
| `--wandb-group`, `--wandb-run-id`, `--wandb-team`, `--wandb-host`, `--wandb-key` | str | |
| `--wandb-mode` | `online` \| `offline` \| `disabled` | Overrides `WANDB_MODE`. |
| `--wandb-dir` | str, `None` | Defaults to `./wandb`. |
| `--disable-wandb-random-suffix` | flag | |
| `--wandb-log-num-images` | int, `0` | Images/videos per rollout; `0` disables. |
| `--wandb-log-image-interval` | int, `1` | Send media every N rollouts. |
| `--use-miles-dashboard` | flag | Async phase/trajectory telemetry. |
| `--miles-dashboard-workspace` | str, `./miles_dashboard` | |

### Fault tolerance

| Flag | Type / default | Notes |
|---|---|---|
| `--use-fault-tolerance` | flag | Restart dead engines during rollout. |
| `--rollout-health-check-interval` | float, `30.0` | Raise it for video models — a request can take minutes. |
| `--rollout-health-check-timeout` | float, `30.0` | |
| `--rollout-health-check-first-wait` | float, `0` | Grace period for compilation/init. |

### Debugging

| Flag | Type / default | Notes |
|---|---|---|
| `--debug-rollout-only` | flag | Rollout, no training. |
| `--train-only` | flag | No engines, no weight sync, no eval. (`--debug-train-only` is a legacy alias.) |
| `--debug-skip-optimizer-step` | flag | No backward/step — weights never drift. Use it to measure pure forward divergence from the engine. |
| `--save-debug-rollout-data` | path template, `None` | `.format(rollout_id)`. |
| `--load-debug-rollout-data` | path template, `None` | Replays a rollout with **no engines started**. |
| `--load-debug-rollout-data-subsample` | float, `None` | |
| `--save-debug-train-data` | path template, `None` | |
| `--dump-details` | dir, `None` | Sets the debug dump paths for post-hoc analysis. |
| `--diffusion-debug-mode` | flag | Engine returns per-step debug tensors; enables the `model_output_*_abs_diff` metrics. |

`--debug-skip-optimizer-step` + `--diffusion-debug-mode` is the standard train/rollout alignment
probe: frozen weights, and the metrics report exactly how far the two forwards drift.

### CI

`--ci-test`, `--ci-metric-checker-key`, `--ci-metric-checker-threshold`. Used by `tests/e2e`;
not meant for manual runs.

---

## SDE step backends

`--diffusion-sde-type` picks the train-side backend automatically:

| `--diffusion-sde-type` | Backend | Dynamics |
|---|---|---|
| `sde` | `DiffusersSdeStepBackend` | Flow-matching SDE over diffusers scheduler sigmas. |
| `ode` | `DiffusersSdeStepBackend` | Same class; deterministic rollout (used by NFT). |
| `cps` | `CpsSdeStepBackend` | CPS kernel, σ = timestep ÷ divisor, log-prob without constants. |

`--sde-step-backend-path` overrides the mapping with your own `SdeStepBackend` subclass.
