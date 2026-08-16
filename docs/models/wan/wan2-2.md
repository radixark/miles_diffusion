---
title: Wan2.2-T2V-A14B
description: Dual-expert MoE video model — Flow-GRPO + PickScore recipe, LoRA SFT recipe, and the high/low-noise expert boundary.
---
## 1. Model introduction

[Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) is a text-to-video model with a **dual-expert
MoE DiT**: a high-noise expert (`transformer`) denoises timesteps `t ≥ boundary` and a low-noise expert
(`transformer_2`) handles the rest. Conditioning comes from a UMT5 text encoder; latents go through the Wan VAE (4×
temporal compression).


**Key highlights for RL training:**

- **Two experts, one boundary.** `boundary_ratio = 0.875` — which expert a train pair updates depends only on its
  timestep. Single- and dual-expert training are both supported: `--update-weight-target-module` names the experts to
  load, train, and sync.
- **Two guidance scales.** Rollout denoises low-noise steps with `guidance_scale_2` and there is **no fallback** —
  training asserts `--diffusion-guidance-scale-2` is set explicitly, because a silent mismatch against rollout would
  corrupt the ratio.
- **USP-ready.** Wan was enabled for Ulysses × Ring sequence parallelism.

## 2. Supported variants

| Model | HF ID | Notes |
|---|---|---|
| Wan2.2-T2V-A14B | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | Default in both Wan recipes |


## 3. Family config

Registered in `miles/backends/fsdp_utils/configs/wan2_2.py`:

| Property | Value | Notes |
|---|---|---|
| Expert routing | `t ≥ 0.875 × num_train_timesteps` → `transformer`, else `transformer_2` | `component_for_timestep` |
| Guidance routing | high-noise → `--diffusion-guidance-scale`, low-noise → `--diffusion-guidance-scale-2` (required) | `select_guidance_scale` |
| Timestep scaling | None — Wan DiT takes raw scheduler timesteps (0..1000) | |
| Condition inputs | `encoder_hidden_states` only (fixed-length UMT5 embeds) | Concat-collate, no padding needed |
| CFG combine | `neg + scale × (pos − neg)` | Standard |
| CFG batching | Off | |
| fp32 param islands | `scale_shift_table`, `time_embedder`, `norm2` kept fp32 under FSDP mixed precision | `models/diffusers/wan2_2/parallel_plan.py` |

## 4. Launch
### 4.1 Flow-GRPO + PickScore (4 train GPUs + 1 reward GPU)

Canonical recipe: `scripts/run_diffusion_grpo_wan22_pickscore_5gpu.py`

```bash
python3 scripts/run_diffusion_grpo_wan22_pickscore_5gpu.py
```


### 4.2 LoRA SFT on (video, prompt) pairs (4 GPUs, no rollout engines)

Recipe: `scripts/run_diffusion_sft_wan22.py`

```bash
MILES_SCRIPT_DATA_JSONL=/abs/data.jsonl python3 scripts/run_diffusion_sft_wan22.py
```

## 5. Pairs well with

- [Single-Prompt Multi-Generation](/advanced/single-prompt-multi-gen) — the microgroup mechanics behind
  `--rollout-microgroup-size 8`.
- [LoRA Training and Weight Sync](/advanced/lora) — IPC merge used by the GRPO recipe.
- [SDE Step Backend](/advanced/sde-backend) — how the trained SDE step is scored train-side.
- [Rewards](/user-guide/rewards) — PickScore worker pool configuration.
