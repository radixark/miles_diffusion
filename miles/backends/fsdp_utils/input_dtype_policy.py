"""Family-declared dtypes for the model-boundary inputs (latents, cond, timestep).

The trainer does not hard-cast its forward inputs; each family declares, per input, a dtype name
("fp32"/"bf16"/"fp16"), "default" for the run's forward dtype, or None to pass the rollout dtype
through. The boundary dtype is what element-wise ops see before any weight is involved, so it must
match what the family's sglang-d pipeline feeds the DiT for log-prob alignment; compute inside the
model is owned by the trainer's autocast (see model_loader.apply_fsdp2).
"""

from __future__ import annotations

import torch

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

INPUT_DTYPE_POLICY_KEYS = ("latents", "cond", "timestep")


def apply_input_dtype_policy(
    policy: dict,
    *,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    conds: tuple,
    default_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """Cast float boundary inputs per family policy ("default"/dtype name/None=passthrough);
    autocast alone would leave element-wise ops running at the raw input dtype."""
    # A typo'd key would silently mean passthrough.
    unknown = set(policy) - set(INPUT_DTYPE_POLICY_KEYS)
    if unknown:
        raise ValueError(f"input_dtype_policy has unknown keys {sorted(unknown)}; known: {INPUT_DTYPE_POLICY_KEYS}")

    def _dtype(key: str) -> torch.dtype | None:
        dtype_name = policy.get(key)
        if dtype_name is None:  # passthrough: keep whatever dtype rollout handed us
            return None
        if dtype_name == "default":  # the run's forward dtype
            return default_dtype
        if dtype_name not in _DTYPES:
            raise ValueError(f"input_dtype_policy[{key!r}] has unknown dtype {dtype_name!r}")
        return _DTYPES[dtype_name]

    def _cast(value, dtype: torch.dtype | None):
        if dtype is None or not torch.is_tensor(value) or not value.is_floating_point():
            return value  # passthrough inputs, int masks, and non-tensors stay untouched
        return value.to(dtype)

    cond_dtype = _dtype("cond")
    return (
        _cast(latents, _dtype("latents")),
        _cast(timesteps, _dtype("timestep")),
        tuple(cond and {key: _cast(value, cond_dtype) for key, value in cond.items()} for cond in conds),
    )
