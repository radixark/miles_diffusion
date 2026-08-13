---
title: Qwen-Image
description: Flow-GRPO with PickScore on Qwen-Image — the flow_grpo-aligned 5-GPU recipe.
---
## 1. Model introduction

[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) is Alibaba's MMDiT text-to-image model. In
miles-diffusion it is the **reference image recipe**: the launcher is aligned flag-for-flag with
flow_grpo's `pickscore_qwenimage` config, so a run here is directly comparable to the published
baseline.

Family key `qwen_image`, resolved from any `--hf-checkpoint` whose name contains `qwen-image`.
Training-side adapters live in `miles/backends/fsdp_utils/configs/qwen_image.py`.

## 2. Supported variants

| Model | HF ID | Notes |
|---|---|---|
| Qwen-Image | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) | The recipe below. |

Any checkpoint matching `qwen-image` resolves to the same config. For a differently-named local
directory — which is the usual case — name the family explicitly:

```bash
--hf-checkpoint /weights/my-qwen-image-ft --diffusion-model-family qwen_image
```

## 3. Environment setup

The launcher's `prepare()` downloads the dataset; the model is pulled by HF on first use.

```bash
# 5 GPUs on one node, HF cache warm
export HF_TOKEN=...        # not gated, but keeps hub rate limits sane
export WANDB_API_KEY=...   # omit and all wandb flags are dropped
```

Dataset: `rockdu/miles-diffusion-datasets`, subset `flowgrpo_pickscore`
(`train.jsonl` + `test.jsonl`, one prompt per row under the `input` key).

## 4. Launch

```bash
python3 scripts/run_diffusion_grpo_pickscore_5gpu_flowgrpo_aligned.py
```

Overrides go through the dataclass CLI:

```bash
python3 scripts/run_diffusion_grpo_pickscore_5gpu_flowgrpo_aligned.py \
    --num-rollout 50 \
    --cuda-visible-devices "0,1,2,3,4" \
    --extra-args "--diffusion-kl-beta 0.02"
```

## 5. Recipe configuration

### 5.1 GPU layout

```bash
--actor-num-gpus-per-node 4      # FSDP DP=4
--rollout-num-gpus 4
--rollout-num-gpus-per-engine 1  # 4 single-GPU sglang-d engines
--num-gpus-per-node 5
--colocate
```

With `cuda_visible_devices="4,5,6,7,1"`: the first four GPUs carry trainer **and** engines
time-multiplexed; the fifth is a dedicated PickScore worker.

### 5.2 Sampling and the SDE window

```bash
--diffusion-num-steps 10 --diffusion-eval-num-steps 50
--diffusion-height 512 --diffusion-width 512
--diffusion-guidance-scale 4.0 --diffusion-true-cfg-scale 4.0
--diffusion-noise-level 1.2
--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window
--diffusion-num-sde-steps 2 --diffusion-sde-window-range 3,5
```

The window range `3,5` with size 2 yields effective SDE indices `[3, 4]`. flow_grpo hard-codes
the range `(0, num_steps//2)` but in practice only trains steps 3-4; this recipe mirrors the
behaviour rather than the constant.

### 5.3 Batch shape

```bash
--rollout-batch-size 32 --n-samples-per-prompt 16   # 512 samples per rollout
--num-steps-per-rollout 2                           # → global_batch_size 256
--rollout-microgroup-size 8
--train-dp-split-mode stride
--micro-batch-size-sample 8 --micro-batch-size-tstep 1 --diffusion-train-iter-order sample_major
```

512 samples × 2 SDE steps = 1024 train pairs, 256 per DP rank, 128 per optimizer step per rank.
The 256-samples-per-optimizer-step figure is what matches flow_grpo's 32-GPU run
(batch 4 × 32 GPU × 2 accumulation).

The DiT-forward tile is 8 samples × 1 timestep — 8 pairs, so 16 forwards per optimizer step.
The 2D form is used rather than a flat `--micro-batch-size 8` because it says the intent
directly: one timestep from each of eight samples, not "whatever eight pairs come next".
`--train-dp-split-mode stride` deals pairs round-robin across the four DP ranks.

### 5.4 Algorithm

```bash
--advantage-estimator grpo --globalize-reward-std --diffusion-clip-range 1e-4
--lr 3e-4 --adam-beta2 0.999 --weight-decay 1e-4
```

Per-prompt mean with a batch-wide std (`--globalize-reward-std` on, `--globalize-reward-mean`
off) is exactly flow_grpo's `PerPromptStatTracker` with `global_std=True`. No KL —
`--diffusion-kl-beta` stays at its `0.0` default.

### 5.5 LoRA

```bash
--use-lora --lora-rank 64 --lora-alpha 128 --lora-init-weights gaussian --lora-ipc-weight-sync
```

`gaussian` is `N(0, 1/r)` for `lora_A` and zeros for `lora_B`, matching flow_grpo. Default target
modules for this family are attention QKV/out plus the image and text projections:

```
to_q  to_k  to_v  to_out.0
add_q_proj  add_k_proj  add_v_proj  to_add_out
img_mlp.net.0.proj  img_mlp.net.2
txt_mlp.net.0.proj  txt_mlp.net.2
```

### 5.6 Precision and rollout parity

```bash
--fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 --diffusion-forward-dtype bf16
--sglang-attention-backend torch_sdpa
--rollout-patch-group sgld
--gradient-checkpointing --deterministic-mode
```

`--rollout-patch-group sgld` applies the diffusers-op-parity patches on the engine so its forward
matches the trainer's. It costs a little rollout throughput and buys a much smaller
`train/log_prob_mean_abs_diff`.

## 6. Model-specific behaviour

Three things this config does that the generic path does not — each exists because of a concrete
train/rollout mismatch:

**Timestep scaling.** `process_timestep_as_input` divides trajectory timesteps by 1000 before the
DiT sees them, matching how sglang-d's Qwen-Image pipeline rescales them.

**CFG with norm rescale.** `cfg_combine` computes `uncond + scale·(cond − uncond)` and then
rescales the result back to `‖cond‖` — but **only when `true_cfg_scale > 1.0`**, mirroring
sglang-d's `QwenImagePipelineConfig.postprocess_cfg_noise`. The recipe's
`--diffusion-true-cfg-scale 4.0` puts it in the rescaling branch.

**RoPE cache rebuilt on CUDA.** diffusers builds `QwenEmbedRope`'s `pos_freqs`/`neg_freqs` on CPU
and only `.to(device)`s them at forward time, while sglang-d rebuilds them on CUDA. CPU and CUDA
`torch.pow` differ by fp32 ULPs, so the two caches byte-differ, RoPE output differs, and every
block drifts — frozen-weight `noise_pred` mean |Δ| lands around 2e-2.
`postprocess_model_after_materialize` rebuilds the caches on the model's CUDA device after FSDP
wrapping to close that gap.

Variable-length text is also handled here rather than generically: `collate_cond_for_sample_batch`
pads `encoder_hidden_states` to the batch max and derives the attention mask from `txt_seq_lens`
(the mask itself is never transmitted from rollout). It honours `pad_to_len` so the legacy tiling
path can reproduce its whole-window padding width bitwise.

Finally, `optimizer_state_allowed_missing` covers `transformer_blocks.59`'s text-branch
parameters — the parent `transformer.forward` discards the last block's text outputs, so those
parameters never receive gradient and never acquire optimizer state.

## 7. Pairs well with

- [LoRA weight sync](/advanced/lora) — `--lora-ipc-weight-sync` is on in this recipe.
- [Dtype Control](/advanced/dtype-control) — why fp32 master + bf16 forward.
- [Deterministic Training](/advanced/deterministic) — `--deterministic-mode` is on.
