---
title: SDE Step Backend
description: Train-side SDE dynamics for Flow-GRPO — when it applies, which flags to set, and how to check train/rollout alignment.
---
Flow-matching RL algorithms fall into two paradigms:

- **Coupled (Flow-GRPO)** — training re-scores the same `(x_t → x_{t+1})`
  transitions that rollout produced, so it needs tractable log-probs via
  `SdeStepBackend`.
- **Decoupled (DiffusionNFT, …)** — training re-noises the final image at the
  rollout's sampling sigmas; rollout dynamics are irrelevant, so this backend is unused.

## 1. When it applies

Flow-GRPO (`--loss-type policy_loss`, the default) calls
`SdeStepBackend.sde_step_logprob` on each recorded transition. The backend
mirrors sglang-diffusion's rollout stepping so train-side log-probs match
rollout-side values.

| Objective | Miles flag | Uses `SdeStepBackend`? | Paper |
|---|---|---|---|
| Flow-GRPO | `--loss-type policy_loss` | **Yes** | [Flow-GRPO](https://arxiv.org/abs/2505.05470) |
| DiffusionNFT | `--loss-type nft` | **No** | [DiffusionNFT](https://arxiv.org/abs/2509.16117) |

Canonical Flow-GRPO recipe: `scripts/run_diffusion_grpo_sd3_ocr_sglang.py`.

Implementation: `miles/backends/fsdp_utils/sde_step_backend.py`. Override with
`--sde-step-backend-path` if you need a custom kernel.

## 2. Dynamics types

| Formulation | `--diffusion-sde-type` | Backend | Notes |
|---|---|---|---|
| Flow-SDE | `sde` (default) | `FlowSdeStepBackend` | Transition noise std η·√(σ/(1−σ))·√(−dt) with the rollout's recorded σ (the grid's second σ replaces the denominator's σ at σ = 1); [Flow-GRPO](https://arxiv.org/abs/2505.05470) |
| CPS | `cps` | `CpsSdeStepBackend` | Transition noise std σ_next·sin(η·π/2); log-prob is −(x−μ)², without the Gaussian constants or the 1/(2σ²) scale (forces `--diffusion-log-prob-no-const`); [FlowCPS](https://arxiv.org/abs/2509.05952) |
| ODE | `ode` | `FlowSdeStepBackend` (placeholder; NFT never calls it) | Deterministic; NFT (or a custom loss formula) only — with the Flow-GRPO loss it is rejected at parse time |

`--diffusion-noise-level` (η) sets the SDE noise scale. DanceGRPO-style constant-η
schedules use the same `sde` backend with a tuned η
([DanceGRPO](https://arxiv.org/abs/2505.07818)).

## 3. Key flags

These CLI flags configure sglang-diffusion rollout stepping and must stay
consistent with the train-side backend:

| CLI flag | Default | Role |
|---|---|---|
| `--diffusion-sde-type` | `sde` | Dynamics family (auto-selects backend) |
| `--diffusion-noise-level` | `0.7` | η in the SDE kernel |
| `--diffusion-num-steps` | `10` | Denoise schedule length |
| `--diffusion-num-sde-steps` | `0` | Steps scored by the step strategy |
| `--diffusion-sde-window-range` | None | e.g. `0,10` for a full window |
| `--diffusion-step-strategy-path` | None | Which steps enter training |

Example (SD3 Flow-GRPO OCR — full-window SDE):

```bash
--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window \
--diffusion-num-sde-steps 10 \
--diffusion-sde-window-range 0,10 \
--diffusion-noise-level 0.7
```

Step strategies live in `miles/rollout/step_strategy_hub.py`. `sde_window` picks a
contiguous window; `epoch_global_random_choice` picks a subset of
`--diffusion-sde-candidate-steps`, shared by each block of `--rollout-batch-size` consecutive prompt groups (one rollout, unless a dynamic-sampling filter drops groups). Partial windows follow the MixGRPO /
TempFlow-GRPO idea ([MixGRPO](https://arxiv.org/abs/2507.21802),
[TempFlow-GRPO](https://arxiv.org/abs/2508.04324)).

## 4. Train / rollout alignment

The train-side kernel must reproduce rollout dynamics. Mismatched σ resolution
or noise level shows up as a rising `train/log_prob_mean_abs_diff`.

Checklist:

1. Both sides read `--diffusion-sde-type` and `--diffusion-noise-level`, and the default backend is derived from the
   former, so only a custom `--sde-step-backend-path` can drift from the engine's dynamics.
2. Train-side scoring takes σ from the trajectory the rollout recorded (`sigma`, `next_sigma`, `sigma_max` per
   transition), not from a scheduler, so `--diffusion-num-steps` needs no train-side copy.
3. Match input dtypes between FSDP forward and the rollout engine for fp32 runs.
4. Watch `train/log_prob_mean_abs_diff` — near zero before the first optimizer step.

## 5. Score centering

**Experimental.** The flag, its metrics and its requirements may still change.

Even when the kernels agree, the engine and the trainer rarely compute the exact same
transition mean, and a one-step-stale rollout under `train_diffusion_async.py` never does.
The engine samples `x' ~ N(μ_q, s²)` while the trainer scores it under `N(μ_θ, s²)`, so the
PPO gradient carries a drift term, `E[A] · ∇θ E_q[clip_A(r)]`, that pulls the actor toward
the engine instead of toward reward
([Score Centering Stabilizes Off-policy RL](https://arxiv.org/abs/2609.20807)).

`--diffusion-score-centering` subtracts that term exactly. Flow-GRPO's per-element mean
log-prob makes `r = (p_θ/q)^(1/D)`, so under the engine `log r` is normal with mean
`−κ/(2D)` and variance `κ/D²`, where `κ = |μ_θ − μ_q|² / s²`. The expected clipped ratio
`E_q[clip_A(r)]` therefore has a closed form in `κ`, and the loss adds
`A · (center − center.detach())`: the logged loss and the clip mask are unchanged, only the
gradient moves, and it is an exact no-op when `μ_q = μ_θ`. The clip is integrated rather than
applied per sample, which is what makes the correction exact for the clipped objective.

```bash
--diffusion-score-centering --diffusion-debug-mode
```

- Requires `--loss-type policy_loss`, `--diffusion-sde-type sde` with the default backend,
  `0 < --diffusion-clip-range < 1`, and `--diffusion-debug-mode`: the engine returns `μ_q`
  (`rollout_prev_sample_means`) and the step's noise std (`rollout_noise_std_devs`) only there.
- The PPO denominator must be the engine's own log-prob, so `--diffusion-recompute-old-log-prob`
  is rejected. The loss also raises if the two sides' transition stds differ or if the stored
  log-probs are not the Gaussian density of the recorded means at the sampled latents.
- `train/prev_mean_diff_over_noise_std` reports `|μ_q − μ_θ|` in units of the step's noise std
  (RMS over latent elements) whenever debug tensors are present; drift grows with it.
- `train/score_center_expected_clipped_ratio` is the center itself: `1.0` means nothing to
  remove, below `1` for positive advantages and above for negative ones under drift.
  `train/score_center_density_abs_diff` must stay near `0` (hard limit `5e-5`).

DiffusionNFT has no SDE log-prob, so score centering does not apply to it.

## 6. Pairs well with

- [Customization](../user-guide/customization.md) — `--diffusion-step-strategy-path` and `--sde-step-backend-path`.
- [SD3 model guide](../models/sd3/sd3.md) — Flow-GRPO vs NFT recipe flags.
- [Quick Start](../getting-started/quick-start.md) — SD3.5 Flow-GRPO OCR walkthrough.
