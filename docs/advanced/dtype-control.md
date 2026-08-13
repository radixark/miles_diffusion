---
title: Dtype Control
description: The three dtype knobs, the model-boundary cast policy, and per-parameter dtype overrides under FSDP2.
---
Diffusion RL is unusually sensitive to precision. The PPO ratio is
`exp(log_prob_new − log_prob_old)`, where `log_prob_old` came from the **rollout engine** and
`log_prob_new` from the **trainer**. Any numeric difference between the two forwards shows up as a
ratio drifting off 1.0 — an update signal made of nothing but rounding.

miles-diffusion therefore separates precision into three layers, each with its own control.

## 1. The three flags

| Flag | Default | Applies to |
|---|---|---|
| `--fsdp-master-dtype` | `fp32` | The FSDP-sharded master copy: load dtype, shard dtype, optimizer state. |
| `--fsdp-reduce-dtype` | `fp32` | `MixedPrecisionPolicy.reduce_dtype` — the gradient reduce-scatter. |
| `--diffusion-forward-dtype` | `bf16` | The DiT forward compute. |

`--fsdp-master-dtype fp32` with a lower `--diffusion-forward-dtype` is ordinary mixed-precision
training. Setting both to `bf16` (as the LTX-2 recipe does) is a deliberate choice to match a
bf16-throughout reference.

`--fsdp-reduce-dtype fp32` keeps multi-rank gradient sums numerically stable. `bf16` matches
flow_grpo's all-bf16 policy, at the cost of bf16 add-non-associativity noise across ranks — two
runs on different rank counts will not agree.

### `--diffusion-forward-dtype` is used three times

This is the flag that matters most, because one value is applied in three separate places:

1. the **sglang-d rollout engine**'s forward,
2. FSDP's `MixedPrecisionPolicy.param_dtype` on the training side,
3. the **training-side input cast** at the model boundary.

All three at one value is what makes `log_prob_new` comparable to `log_prob_old`. Split them and
the ratio drifts.

### fp16 needs the grad scaler

fp16 policy gradients are small enough to underflow. The actor always constructs a
`ShardedGradScaler`, enabled only when the forward dtype is fp16, which keeps the `found_inf`
decision synchronized across FSDP ranks. It is a no-op for bf16 and fp32 — nothing to configure.

## 2. The model-boundary cast policy

Autocast is not enough on its own. It governs op *interiors* — matmuls, convolutions — but
element-wise ops run at whatever dtype their inputs already carry. So the dtype of the tensors
handed to the DiT is itself a semantic choice, and it has to match what the family's sglang-d
pipeline feeds its DiT.

Each family therefore declares an `input_dtype_policy` over three boundary inputs:

```python
input_dtype_policy = {"latents": ..., "cond": ..., "timestep": ...}
```

| Value | Meaning |
|---|---|
| `"default"` | Cast to the run's `--diffusion-forward-dtype`. |
| `"fp32"` / `"bf16"` / `"fp16"` | Cast to that dtype specifically. |
| `None` | Passthrough — keep whatever dtype rollout handed us. |

The base class defaults to passthrough on all three, so families opt into casts explicitly. LTX-2
is the current example:

```python
# miles/backends/fsdp_utils/configs/ltx.py
input_dtype_policy = {"latents": "default", "cond": "default", "timestep": None}
```

`forward_velocity` anchors its element-wise math on `latents.dtype` and rollout runs it in bf16 —
so latents and conditioning are cast at the boundary while the timestep is left alone.

Non-float tensors (integer masks) and non-tensors are never cast, whatever the policy says.
An unknown key in the dict is a hard error rather than a silent passthrough — a typo here would
otherwise be invisible.

**Division of labour:** `input_dtype_policy` owns the boundary; autocast owns the interior. That
split also keeps gradient-checkpointing recompute consistent, since the recomputed forward sees
the same ambient autocast as the original.

## 3. Per-parameter dtype overrides

Some parameters must stay fp32 even in a bf16 forward — timestep embedders, RoPE frequency
buffers, `scale_shift_table`-style modulation parameters. FSDP2's stock `MixedPrecisionPolicy`
is per-wrap, not per-parameter, so miles-diffusion adds a targeted patch.

### Declaring the rule

A model's `FSDPParallelPlan` carries FQN glob patterns, matched against **root-relative** names:

```python
# miles/backends/fsdp_utils/models/diffusers/wan2_2/parallel_plan.py
FSDP_PARALLEL_PLAN = FSDPParallelPlan(
    param_dtype_patterns={
        "*scale_shift_table": "fp32",
        "*time_embedder*": "fp32",
        "*.norm2.*": "fp32",
    },
)
```

Semantics:

- Patterns apply **in declaration order**, and a later pattern overrides an earlier one — a narrow
  rule can carve a parameter back out of a broad one.
- A pattern matching **nothing** is an error, not a no-op. Rules do not rot silently when a model
  is renamed.
- An assignment equal to the group default compiles to nothing (which is also how a carve-out
  works).

SD3 and Qwen-Image declare an empty plan; only Wan2.2 currently needs overrides.

### How it is compiled

`compile_param_dtype_maps` (`miles/backends/fsdp_utils/mixed_precision.py`) does two passes:

```
pass 1   expand patterns against root FQNs → one dtype per matched parameter
pass 2   walk the wraps in fully_shard() call order, claiming parameters
         first-wrap-wins (the rule FSDP2 itself applies), and re-key each
         override to the FQN local to its owning wrap
```

Anything no wrap claims lands in `root_map` under its root FQN. The result is one dtype map per
`fully_shard` call, keyed the way that call will look parameters up.

Because the runtime map is keyed by *wrap-local* FQN, two parameters in one wrap group that share
a local FQN but want different dtypes cannot be told apart — that is detected and raised, rather
than silently applying one dtype to both.

### The runtime patch

When any override exists, the actor swaps in `ParamDtypeMixedPrecisionPolicy` (a
`MixedPrecisionPolicy` with an extra `param_dtype_map`) and applies
`apply_param_dtype_map_patch()`.

<Warning>

The patch is **version-gated on `torch==2.11.0`** and raises on any other version. This is
deliberate: it reaches into FSDP2 internals, and silently running against a different torch would
be worse than failing. `requirements.txt` pins the matching torch.

</Warning>

Wraps with no overrides get a plain `MixedPrecisionPolicy` and never touch the patched path.

Both policies set `cast_forward_inputs=False` — input casting belongs to `input_dtype_policy`,
not to FSDP.

At startup the actor logs what it compiled:

```
FSDP: wrapping 40 modules of type ('WanTransformerBlock',), param_dtype=torch.bfloat16,
reduce_dtype=torch.float32, param_dtype_overrides=123 (4,567,890 parameters)
```

If that count is zero when you expected overrides, your patterns did not match.

## 4. Verifying the result

Precision work is only real if you can measure it. Run with:

```bash
--diffusion-debug-mode --debug-skip-optimizer-step
```

The engine then returns per-step debug tensors and the trainer never updates weights, so any
divergence is pure forward-path difference. Watch these metrics:

| Metric | What it tells you |
|---|---|
| `train/model_output_mean_abs_diff` | Average trainer-vs-rollout `noise_pred` gap. |
| `train/model_output_max_abs_diff` | Worst element. |
| `train/model_output_rel_max` | The worst element relative to the rollout output's scale. |
| `train/log_prob_mean_abs_diff` | The gap that actually feeds the PPO ratio. |
| `train/ratio_abs_minus_1` | How far the ratio sits from 1.0 in a live run. |

For a sense of scale: the Qwen-Image RoPE cache bug — CPU-built vs CUDA-built frequency tables
differing by fp32 ULPs — produced a frozen-weight `noise_pred` mean |Δ| of about **2e-2**. Small
absolute numbers here are not automatically fine; compare against a known-good run.

## 5. Related knobs

| Flag | Why it matters for precision |
|---|---|
| `--rollout-patch-group sgld` | Engine-side op-parity patches (RMSNorm, LayerNormScaleShift, MulAdd, QK-norm RoPE). Costs rollout throughput, buys forward agreement. |
| `--fsdp-attention-backend` / `--sglang-attention-backend` | Different attention kernels give different results at the same dtype. Match them. |
| `--deterministic-mode` | Removes run-to-run nondeterminism so a dtype change is the only variable. See [Deterministic Training](/advanced/deterministic). |
| `--diffusion-recompute-old-log-prob` | Sidesteps the question: recompute `log_prob_old` with the trainer's own forward, making the ratio implementation-consistent by construction. |
