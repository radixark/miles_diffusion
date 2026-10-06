---
title: SDE Step Backend
description: Train-side SDE dynamics for Flow-GRPO — when it applies, which flags to set, and how to check train/rollout alignment.
---
Flow-matching RL algorithms fall into two paradigms:

- **Coupled (Flow-GRPO)** — training re-scores the same `(x_t → x_{t+1})`
  transitions that rollout produced, so it needs tractable log-probs via
  `SdeStepBackend`.
- **Decoupled (DiffusionNFT, …)** — training samples its own timesteps from the
  final image; rollout dynamics are irrelevant, so this backend is unused.

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
| Flow-SDE | `sde` (default) | `DiffusersSdeStepBackend` | η·√(σ/(1−σ)) with the scheduler's σ; [Flow-GRPO](https://arxiv.org/abs/2505.05470) |
| CPS | `cps` | `CpsSdeStepBackend` | [FlowCPS](https://arxiv.org/abs/2509.05952) |
| ODE | `ode` | `DiffusersSdeStepBackend` (η=0) | Deterministic; NFT rollout only |

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
contiguous window; `epoch_global_random_choice` picks a per-epoch subset via
`--diffusion-sde-candidate-steps`. Partial windows follow the MixGRPO /
TempFlow-GRPO idea ([MixGRPO](https://arxiv.org/abs/2507.21802),
[TempFlow-GRPO](https://arxiv.org/abs/2508.04324)).

## 4. Train / rollout alignment

The train-side kernel must reproduce rollout dynamics. Mismatched σ resolution
or noise level shows up as a rising `train/log_prob_mean_abs_diff`.

Checklist:

1. Match `--diffusion-sde-type` and `--diffusion-noise-level` on train and rollout.
2. Use the same `--diffusion-num-steps` schedule.
3. Match input dtypes between FSDP forward and the rollout engine for fp32 runs.
4. Watch `train/log_prob_mean_abs_diff` — near zero before the first optimizer step.

## 5. Score centering

Even when the kernels agree, the engine and the trainer rarely compute the exact same
transition mean. The engine samples `x' ~ N(μ_q, s²)`, while the trainer scores it under
`N(μ_θ, s²)`. Once `μ_q ≠ μ_θ`, the expected score under the engine is no longer zero, and
the policy gradient picks up a drift term: the expected advantage times that expected score.
Drift distills the trainer toward the engine. The engine is then re-synced from the trainer,
so the bias compounds over training
([Score Centering](https://arxiv.org/abs/2609.20807)).

`--diffusion-score-centering` subtracts the expected score from every pair's gradient. For a
Gaussian step with a θ-independent std, `∇θ E_q[log p_θ(x')] = ∇θ log p_θ(μ_q)`, so the
correction is exact and needs only the engine's mean `μ_q`, with no top-k approximation as
in the LLM case. It is a no-op when `μ_q = μ_θ`, scales with each pair's PPO gradient weight
(zero where the clip binds), and leaves the logged loss unchanged. Both built-in backends
satisfy the Gaussian form. A custom backend whose `log_prob` is not quadratic in
`prev_sample − prev_mean` must override `SdeStepBackend.expected_log_prob`.

```bash
--diffusion-score-centering --diffusion-debug-mode --diffusion-recompute-old-log-prob
```

- The engine returns `μ_q` (`rollout_prev_sample_means`) only under `--diffusion-debug-mode`,
  so the flag requires it. Debug mode ships every per-step debug tensor, which is heavier
  than score centering needs.
- Pair it with `--diffusion-recompute-old-log-prob`. The PPO log-prob is a per-element mean,
  so the rollout ratio `exp(log p_θ − log q)` is not an importance weight and does not cancel
  drift. A train/rollout mismatch near `--diffusion-clip-range` instead makes the clip mask a
  mismatch-correlated subset of pairs. Recomputing keeps the ratio about policy movement and
  leaves the mismatch to score centering.
- `train/prev_mean_diff_over_noise_std` reports `|μ_q − μ_θ|` in units of the step's noise std (RMS over
  latent elements). Drift grows with it; a value of 0 means score centering has nothing to do.

DiffusionNFT has no SDE log-prob, so score centering does not apply to it.

## 6. Pairs well with

- [Customization](../user-guide/customization.md) — `--diffusion-step-strategy-path` and `--sde-step-backend-path`.
- [SD3 model guide](../models/sd3/sd3.md) — Flow-GRPO vs NFT recipe flags.
- [Quick Start](../getting-started/quick-start.md) — SD3.5 Flow-GRPO OCR walkthrough.
