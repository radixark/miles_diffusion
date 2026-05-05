# Rollout Parallel Accuracy

sglang-diffusion supports several rollout-side parallel strategies. These
strategies are important for throughput and memory, but they can also change the
numeric path of the diffusion forward pass and rollout log-prob computation.
For RL post-training, those differences matter: the trainer consumes rollout
trajectories, rewards, and log-probs, so a parallel configuration should be
chosen with a clear understanding of its accuracy behavior.

This document summarizes the currently relevant rollout parallel strategies and
their observed precision impact.

## Parallel Strategies

| Strategy | Meaning | Typical purpose |
| --- | --- | --- |
| SP / Ulysses | Sequence parallelism over latent/image tokens. Each rank handles a shard of the sequence dimension and uses collectives inside attention. | Increase max resolution or reduce per-GPU activation memory. |
| TP | Tensor parallelism inside the DiT / transformer layers. | Split model compute and parameters across GPUs. |
| CFGP | Classifier-free-guidance parallelism. Conditional and unconditional branches are computed on different ranks and then combined. | Reduce wall-clock cost of CFG when both branches are required. |

The main tensors to watch are:

| Tensor | Why it matters |
| --- | --- |
| `model_output` | Direct output of the DiT denoiser. Differences here affect the denoising trajectory. |
| `prev_sample_mean` | Scheduler mean update before adding SDE/CPS variance noise. |
| `variance_noise` | Random noise used by SDE/CPS rollout. |
| `noise_std_dev` | Scheduler noise scale. |
| `rollout_log_probs` | Per-step rollout log-prob consumed by RL training. This is the most important rollout-side scalar for policy-gradient correctness. |

## Tested Scope

The rollout-parallel accuracy checks were run on:

| Model | Resolution | Steps | GPUs | Reference |
| --- | ---: | ---: | ---: | --- |
| `Qwen/Qwen-Image` | 1024 x 1024 | 50 | 1-2 | diffusers, single GPU, TP1 SP1, no CFGP |
| `Tongyi-MAI/Z-Image-Turbo` | 1024 x 1024 | 9 | 1-2 | diffusers, single GPU, TP1 SP1, no CFGP |

## Accuracy Summary

| Parallel strategy | `rollout_log_probs` vs single-GPU reference | DiT-side tensors vs single-GPU reference | Practical interpretation |
| --- | --- | --- | --- |
| SP / Ulysses | Bit-exact in the tested Qwen-Image and Z-Image-Turbo runs. | Bit-exact in the tested Qwen-Image and Z-Image-Turbo runs. | Safest tested rollout parallel mode for accuracy-sensitive log-prob replay. |
| TP | Bit-exact in the tested rollout log-prob path. | Qwen-Image was bit-exact in the tested TP2-SP1 run; Z-Image-Turbo showed DiT-side drift in `model_output` / `prev_sample_mean`. | Log-prob can remain exact even when the model forward path has small architecture-dependent reduction-order drift. |
| CFGP | Bit-exact in the tested rollout log-prob path. | `model_output` / `prev_sample_mean` can drift from the serial CFG reference because cond/uncond branches are combined through CFG-parallel collectives. | Useful for CFG throughput, but do not assume full tensor bit-exactness vs serial CFG. |

## Detailed Results

### SDE Rollout

| Parallel strategy | `variance_noise` | `noise_std_dev` | `rollout_log_probs` | `model_output` / `prev_sample_mean` |
| --- | --- | --- | --- | --- |
| SP / Ulysses | 0 max abs diff | 0 max abs diff | 0 max abs diff | 0 max abs diff in tested runs |
| TP | 0 max abs diff | 0 max abs diff | 0 max abs diff | Model-dependent: exact for tested Qwen-Image; drift observed for tested Z-Image-Turbo |
| CFGP | 0 max abs diff | 0 max abs diff | 0 max abs diff | Drift observed in CFG-parallel `model_output` / `prev_sample_mean` |

### CPS Rollout

| Parallel strategy | `variance_noise` | `noise_std_dev` | `rollout_log_probs` | `model_output` / `prev_sample_mean` |
| --- | --- | --- | --- | --- |
| SP / Ulysses | 0 max abs diff | 0 max abs diff | 0 max abs diff | 0 max abs diff in tested runs |
| TP | 0 max abs diff | 0 max abs diff | 0 max abs diff | Model-dependent: exact for tested Qwen-Image; drift observed for tested Z-Image-Turbo |
| CFGP | 0 max abs diff | 0 max abs diff | 0 max abs diff | Drift observed in CFG-parallel `model_output` / `prev_sample_mean` |

### ODE Rollout

| Parallel strategy | `rollout_log_probs` | `model_output` / deterministic update |
| --- | --- | --- |
| SP / Ulysses | 0 max abs diff | 0 max abs diff in tested runs |
| TP | 0 max abs diff | Model-dependent drift can appear in the DiT forward path |
| CFGP | 0 max abs diff | Drift observed in CFG-parallel `model_output` / `prev_sample_mean` |

ODE has a special precision contract: the rollout branch should preserve
bit-exactness with the non-rollout deterministic scheduler step. For this
reason, SGLang keeps the ODE branch dtype-preserving instead of applying the
same fp32 entry cast used by SDE/CPS.

## Practical Guidance

- Prefer SP / Ulysses when the main goal is scaling rollout resolution while
  preserving rollout log-prob accuracy. It is the cleanest tested path for
  bit-exact log-prob replay.
- Use TP when model memory or compute requires it, but validate DiT-side tensor
  drift for the specific backbone. The tested Qwen-Image path was bit-exact;
  the tested Z-Image-Turbo path still showed model-output drift.
- Use CFGP when CFG throughput matters, but treat it as a numerically different
  forward path from serial CFG for `model_output` and `prev_sample_mean`.
  `rollout_log_probs` were still bit-exact in the tested rollout path.
- For SDE/CPS, expect fp32 rollout log-prob computation. For ODE, preserve the
  native deterministic scheduler path.
