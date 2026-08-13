---
title: LTX-2
description: Video GRPO on LTX-2.3 — native model package, CPS-SDE, unguided forward.
---
## 1. Model introduction

[LTX-2](https://github.com/Lightricks/LTX-2) is Lightricks' audio-video DiT. LTX-2.3 is the
variant miles-diffusion trains today, and it is the framework's most unusual recipe: it is the
only family that does **not** go through diffusers, the only one that trains **unguided**, and
the only one on the **CPS** SDE kernel.

Family key `ltx`, resolved from any `--hf-checkpoint` containing `ltx`. Config lives in
`miles/backends/fsdp_utils/configs/ltx.py`; the model package is
`miles/backends/fsdp_utils/models/ltx/`.

## 2. Supported variants

| Model | HF ID | Notes |
|---|---|---|
| LTX-2.3 | [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3) | The recipe below. Video branch only. |

The audio branch is loaded but never trained — `optimizer_state_allowed_missing = ["audio"]`.

## 3. Environment setup

`ltx-core` is pinned in `requirements.txt` and baked into the Docker image; a source install must
have it.

```bash
export WANDB_API_KEY=...
```

Dataset: `rockdu/miles-diffusion-datasets`, subset `flowgrpo_pickscore`.

## 4. Launch

```bash
python3 scripts/run_diffusion_grpo_ltx23_sglang.py
```

Uses 5 GPUs (`cuda_visible_devices="0,1,2,3,4"`): 4 for train+rollout colocate, 1 for the
PickScore actor.

## 5. Recipe configuration

### 5.1 Media and schedule

```bash
--hf-checkpoint Lightricks/LTX-2.3
--diffusion-output-num-frames 57 --diffusion-fps 24
--diffusion-height 512 --diffusion-width 768
--diffusion-num-steps 24
--diffusion-sde-type cps
--diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_random_choice
--diffusion-num-sde-steps 3 --diffusion-sde-candidate-steps 0,1,2,3,4,5,6,7,8,9
```

57 frames at 24 fps, 24 denoising steps, of which 3 are trained. The step strategy draws those 3
once per epoch from candidates 0-9 and shares them across every sample in the epoch — with only
8×8 = 64 samples per rollout, per-request randomization would add variance the batch is too small
to average out.

`--diffusion-sde-type cps` selects `CpsSdeStepBackend` on the training side: σ comes straight
from the carried rollout timesteps (σ×1000, hence the family's `sde_timestep_divisor = 1000.0`)
rather than from a scheduler lookup, and the log-prob drops its constants.

### 5.2 Unguided training

```bash
--diffusion-guidance-scale 1.0
```

`LTXTrainPipelineConfig` sets `supports_cfg_training = False`. Passing anything other than
`--diffusion-guidance-scale 1.0`, or passing `--diffusion-negative-prompt`, is a hard error at
parse time. The training forward is a single unguided velocity pass — there is no negative branch
to combine.

### 5.3 Batch shape

```bash
--rollout-batch-size 8 --n-samples-per-prompt 8   # 64 samples per rollout
--num-steps-per-rollout 2                         # → global_batch_size 32
--rollout-microgroup-size 1
--train-dp-split-mode stride
--micro-batch-size-sample 1 --micro-batch-size-tstep 1 --diffusion-train-iter-order sample_major
--rollout-parser-num-workers 8
```

64 samples × 3 SDE steps = 192 train pairs, 48 per DP rank, 24 per optimizer step per rank.
The DiT-forward tile is 1 sample × 1 timestep — one 57-frame video per forward is already all the
activation memory there is.

`--rollout-microgroup-size 1` keeps one video per request. `--rollout-parser-num-workers 8`
matters here specifically: 57-frame trajectory tensors are large, and one deserializer actor
becomes the bottleneck that starves the engines.

### 5.4 Algorithm

```bash
--advantage-estimator grpo --globalize-reward-std
--diffusion-clip-range 1e-5 --diffusion-kl-beta 0.0
--lr 2e-4 --adam-beta2 0.999 --weight-decay 1e-4
--use-lora --lora-rank 64 --lora-alpha 128 --lora-init-weights gaussian
```

The clip range is `1e-5` — an order of magnitude tighter than the image recipes' `1e-4`.
Default LoRA targets for this family are attention plus FFN: `to_q`, `to_k`, `to_v`, `to_out.0`,
`net.0.proj`, `net.2`.

Note that `--lora-ipc-weight-sync` is **not** enabled here; weights sync through the regular
merged-tensor path.

### 5.5 Precision — bf16 end to end

```bash
--fsdp-master-dtype bf16 --fsdp-reduce-dtype bf16 --diffusion-forward-dtype bf16
--sglang-dit-precision bf16
--fsdp-attention-backend sdpa_math --sglang-attention-backend torch_sdpa
```

Unlike the image recipes, LTX runs bf16 for master weights and gradient reduction as well as the
forward. Both sides use a math-SDPA attention backend, which is deterministic by construction and
identical across the two implementations.

### 5.6 Reward

```bash
--rm-type pickscore --pickscore-num-frames 3
```

`--pickscore-num-frames 3` scores three evenly-spaced frames per video and averages. Scoring all
57 would dominate rollout time for a reward signal that barely changes.

### 5.7 Timeouts

```bash
--rollout-health-check-interval 120
--miles-router-health-check-failure-threshold 30
```

A 57-frame video request takes minutes. The image-recipe defaults (30 s interval, 3 failures)
would kill healthy engines mid-generation.

## 6. Model-specific behaviour

**Native model package, not diffusers.** `model_backend_path` is `MilesModelBackend` and
`model_package` is `miles.backends.fsdp_utils.models.ltx`, which supplies loading, modeling,
attention, and the FSDP plan directly against `ltx_core`. FSDP wraps `BasicAVTransformerBlock`.

**Velocity, converted.** LTX's DiT predicts velocity; the trainer needs a noise prediction.
`forward_velocity` reconstructs the denoised sample with `ltx_core.utils.to_denoised` in fp32 and
divides back out — algebraically an identity for text-to-video, but the fp32 rounding path is
what the end-to-end metrics were recorded against, so it is kept explicitly.

**Boundary dtype policy.** `input_dtype_policy = {"latents": "default", "cond": "default",
"timestep": None}`. `forward_velocity` anchors its element-wise math on `latents.dtype`, and
rollout runs it in bf16, so latents and conditioning are cast at the boundary while the timestep
passes through untouched. See [Dtype Control](/advanced/dtype-control).

**Rollout patch group `ltx`.** `--rollout-patch-group ltx` applies two engine-side patches before
model construction: `patch_ltx2_rollout_cond_kwargs` and `patch_ltx2_disable_av_cross` (audio-video
cross-attention off, matching a video-only training forward).

## 7. Limitations

- **No sequence parallelism.** `sequence_parallel_plan` raises `NotImplementedError`; leave
  `--sequence-parallel-size` at `1`.
- **Video branch only.** The audio stream is not trained.

## 8. Pairs well with

- [Dtype Control](/advanced/dtype-control) — the boundary-dtype policy is why LTX has one.
- [Deterministic Training](/advanced/deterministic) — `sdpa_math` is the deterministic-safe
  backend this recipe relies on.
