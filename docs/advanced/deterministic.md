---
title: Deterministic Training
description: What --deterministic-mode covers, which attention backends it accepts, and what it deliberately does not fix.
---
`--deterministic-mode` makes the **training actor's** forward and backward bit-reproducible across
runs. It is on in most of the shipped recipes, because when you are chasing a train/rollout
numeric gap you need run-to-run noise to be zero before any measurement means anything.

The flag name is kept identical to Megatron's, and it maps to `FSDPArgs.deterministic_mode`.

## What it turns on

Determinism is set in two places, because some of it must be in the process environment *before*
the actor's first line runs.

### At actor spawn (`miles/ray/actor_group.py`)

```bash
NCCL_DETERMINISTIC=1
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

NCCL reads its variable at `init_process_group` and cuBLAS reads its at the first matmul — both
before the actor body executes, so setting them from Python would be too late. Both are
`setdefault`, so an explicit value in `--train-env-vars` wins.

`:4096:8` rather than `:16:8`: both are deterministic, but the larger workspace keeps cuBLASLt
from being workspace-limited and avoids the throughput hit. Costs roughly 32 MiB per handle.

### In the actor (`miles/backends/fsdp_utils/actor.py`)

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)
```

`warn_only=False` is load-bearing, not caution. SDPA's deterministic backward is gated on
`!warnOnly` inside `aten`'s `attention_backward.cu`, so `warn_only=True` would silently be a no-op
on the native backend — the most common one.

## Attention is the hard part

`torch.use_deterministic_algorithms` only governs torch-native ops. A custom attention kernel is
opaque to it and would run nondeterministically **without any warning**. So miles-diffusion
classifies backends explicitly:

| `--fsdp-attention-backend` | How determinism is obtained |
|---|---|
| unset, `*native*`, `*math*` (SDPA) | torch's global flag covers it. `math` backends are deterministic by construction. |
| `*flash*` (flash-attn, FA3) | torch's flag cannot reach it — miles patches `deterministic=True` onto the kernel entry points. |
| `sage`, `xformers`, `flex`, `aiter`, … | No hook exists. **Rejected.** |

The check runs **driver-side, before any actor launches** (`validate_attention_args`), so a
misconfiguration fails in seconds instead of after a multi-node startup:

```
ValueError: deterministic_mode cannot guarantee a deterministic backward for attention
backend 'sage': it is a custom kernel opaque to torch.use_deterministic_algorithms with
no deterministic hook here. Use a flash (flash/_flash_3) or native (SDPA) backend.
```

A flash backend that is installed but exposes no `deterministic` parameter is also rejected,
with a message naming which kernels were found.

### How the flash patch works

For diffusers-backed families, miles wraps the dispatch functions diffusers routes flash through:

```
flash_attn_func         flash_attn_varlen_func
flash_attn_3_func       flash_attn_3_varlen_func
```

Each is inspected for a `deterministic` parameter and, if present, replaced with a
`functools.partial(fn, deterministic=True)`. Entry points without the parameter are skipped.

Native model packages (`MilesModelBackend`, e.g. LTX-2) declare their own entry points via
`modeling.flash_attention_entrypoints(backend)`. A package that declares none raises
`NotImplementedError` under deterministic mode rather than pretending — use a native/math backend
for that model instead.

When a package also declares `required_flash_kernel_label(backend)`, the named kernel *must* end
up patched; if it did not, startup fails with the list of what was patched.

## What it does not cover

<Warning>

`--deterministic-mode` is **train-actor only**. It does not make the rollout engine deterministic.

</Warning>

- **The rollout engine.** sglang-d determinism is a separate concern, controlled through
  `--sglang-attention-backend`, `--sglang-dit-precision`, and the batch-invariant-op environment
  the engine actors are launched with. Rollout *sampling* is seeded by `--rollout-seed`.
- **Different rank counts.** With `--fsdp-reduce-dtype bf16`, gradient sums are non-associative
  across ranks, so a 4-rank and an 8-rank run will not match even in deterministic mode. Use
  `fp32` reduce when you need that.
- **Train/rollout agreement.** Determinism removes run-to-run variance; it does not make the two
  forwards equal. That is a precision problem — see [Dtype Control](/advanced/dtype-control).

## Cost

Real but usually acceptable: `cudnn.benchmark=False` gives up kernel autotuning, deterministic
kernels are chosen over faster nondeterministic ones, and cuBLAS reserves ~32 MiB per handle. The
recipes that need reproducibility more than throughput keep it on permanently.

## Typical use

Recipes that ship with it on include the Qwen-Image PickScore recipe, the SD3.5 OCR recipe, and
the LTX-2.3 recipe.

The standard alignment investigation combines it with the debug flags:

```bash
python3 scripts/run_diffusion_grpo_sd3_ocr_sglang.py --debug-alignment
```

which adds `--diffusion-debug-mode --debug-skip-optimizer-step`: weights never update, the engine
returns per-step debug tensors, and `train/model_output_*_abs_diff` reports pure forward-path
divergence with no run-to-run noise underneath it.

## Related

- [Dtype Control](/advanced/dtype-control) — the other half of train/rollout numeric agreement.
- `--rollout-patch-group sgld` — engine-side op-parity patches. Note that attention is
  deliberately **not** patched there: overriding `USPAttention.forward` breaks bitwise
  SP-invariance because kernel choice depends on head and batch shape. Align attention through
  backend selection instead.
