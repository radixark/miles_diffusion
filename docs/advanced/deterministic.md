---
title: Deterministic Training
description: What --deterministic-mode covers, which attention backends it accepts, and what it deliberately does not fix.
---

`--deterministic-mode` configures the **training actor's** forward and backward for repeatable execution with the same
hardware, topology, software stack, and inputs. Passing argument validation does not prove that an external kernel
honors the request; validate repeatability on the target GPU, build and input shapes.

## What it turns on

### At actor spawn (`miles/ray/actor_group.py`)

```bash
NCCL_DETERMINISTIC=1
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

NCCL reads its variable at `init_process_group` and cuBLAS at the first matmul — both before the
actor body executes, so setting them from Python would be too late. Both are `setdefault`, so an
explicit value in `--train-env-vars` wins.

`:4096:8` rather than `:16:8`: both are deterministic, but the larger workspace keeps cuBLASLt
from being workspace-limited. Costs roughly 32 MiB per handle.

### In the actor (`miles/backends/fsdp_utils/actor.py`)

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)
```

`warn_only=False` is load-bearing: SDPA's deterministic backward is gated on `!warnOnly` inside
`aten`'s `attention_backward.cu`, so `warn_only=True` would silently be a no-op on the native
backend — the most common one.

## Attention is the hard part

`torch.use_deterministic_algorithms` only governs torch-native ops. A custom attention kernel is
opaque to it and would run nondeterministically **without any warning**, so backends are
classified explicitly:


| `--fsdp-attention-backend`             | How determinism is obtained                                                                     |
| -------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `*native*`, `*math*` (SDPA)           | torch's global flag covers it. `math` backends are deterministic by construction.               |
| `flash`, `flash_varlen` (Diffusers FA2) | miles patches the selected entry point with `deterministic=True`.                            |
| `_flash_3` (Diffusers dense FA3)      | miles registers a direct public-autograd adapter and forces its deterministic request.        |
| `sage`, `xformers`, `flex`, `aiter`, … | No hook exists. **Rejected.**                                                                   |


Explicit Diffusers backend checks run **driver-side, before any actor launches**
(`validate_attention_args`). When the CLI backend is unset, the Diffusers actor resolves the active
backend, including `DIFFUSERS_ATTN_BACKEND`, and applies its deterministic integration. Native model
families use their separate package hooks, including spellings such as `flash_attention_3`.

The selected flash entry point must expose `deterministic`; installing an unrelated FA2 kernel
does not satisfy FA3 validation. Backend validation does not replace a GPU repeatability test.

### How the flash integrations work

Diffusers FA2 `flash` and `flash_varlen` retain a
`functools.partial(fn, deterministic=True)` on their selected entry point.

For dense FA3, the pinned Diffusers dispatcher drops the deterministic argument and uses a
forward-only custom op. Miles instead registers an adapter that calls FA3's public autograd
function directly with `num_splits=1`. Explicit `_flash_3` selection installs it even with
deterministic mode off, preserving backward support in both modes. The effective kernel flag is
the OR of the actor's forced mode, the per-call request and PyTorch's global deterministic mode;
a per-call `False` cannot override a forced request.

The adapter rejects masks, nonzero dropout, split counts other than one and Diffusers context
parallelism. Miles pure Ulysses wraps the local attention call; Ring and FA3 varlen/hub paths are
outside this integration. The registry is process-wide, so install it before model execution
and keep models needing different forced modes in separate processes.

The measured H100 cases passed output/gradient repeatability checks for BF16 SP=1/2/4 and
FP16 SP=1. This establishes the tested configurations, not every build or shape.
See [FA3 validation](../developer/fa3-validation.md) for the exact environment, reproduction and
the separate Krea2 DP=2/SP=1 short E2E result.

## What it does not cover

- **Rollout determinism.** SGLang diffusion has its own attention and inference paths. This
training flag does not configure or establish their determinism; validate the rollout stack separately.
- **Train/rollout agreement.** Determinism removes run-to-run variance; it does not make the two
forwards equal. That is a precision problem — see [Dtype Control](dtype-control.md).

## Cost

Real but usually acceptable: `cudnn.benchmark=False` gives up kernel autotuning, deterministic
kernels are chosen over faster nondeterministic ones, and cuBLAS reserves ~32 MiB per handle.
Recipes that need reproducibility more than throughput keep it on permanently.

## In CI

The e2e suite is built on this flag. Each test under `tests/e2e/short/` runs a real recipe with
`--deterministic-mode` for its registered short run (currently two or four rollouts) and compares every registered
metric series — reward statistics, old/new log-probs, grad norm — **bit for bit** against a standard committed under
`tests/ci/fixtures/e2e_standards/` (strict unless the test registers a per-metric tolerance).
Determinism is what makes strict comparison viable: a tolerance wide enough to absorb run-to-run
noise would also absorb small regressions. Standards are re-recorded by the PR author
(`python tests/ci/e2e_metrics_registry.py record --test <test-file>`), never by CI.

Because reward metrics are compared too, the CI recipes also pin the engine side — e.g.
`--sglang-dit-precision fp16`, plus the batch-invariant-op environment above — matching the
"what it does not cover" list: bit-stable curves need both halves.

## Related

- [Dtype Control](dtype-control.md) — the other half of train/rollout numeric agreement.
