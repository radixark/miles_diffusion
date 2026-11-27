"""Family-declared dtypes for the model-boundary inputs (latents, cond, timestep).

Each family declares, per input, a dtype name ("fp32"/"bf16"/"fp16"), "default" for the run's
forward dtype, or None to pass the rollout dtype through. The trainer resolves the policy once per
run (``InputDtypes.resolve``) and casts each stage's inputs with it as it sends them to the model
(``PreparedBatch.model_inputs``). The boundary dtype is what element-wise ops see before any weight
is involved, so it must match what the family's sglang-d pipeline feeds the DiT for log-prob
alignment; compute inside the model is owned by the trainer's autocast (see model_loader.apply_fsdp2).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

INPUT_DTYPE_POLICY_KEYS = ("latents", "cond", "timestep")


@dataclass(frozen=True)
class InputDtypes:
    """A family's ``input_dtype_policy`` resolved for one run: the dtype each boundary input is sent in, None to
    keep the rollout dtype."""

    latents: torch.dtype | None
    timestep: torch.dtype | None
    cond: torch.dtype | None

    @classmethod
    def resolve(cls, policy: dict, default_dtype: torch.dtype) -> InputDtypes:
        # A typo'd key would silently mean passthrough.
        unknown = set(policy) - set(INPUT_DTYPE_POLICY_KEYS)
        if unknown:
            raise ValueError(
                f"input_dtype_policy has unknown keys {sorted(unknown)}; known: {INPUT_DTYPE_POLICY_KEYS}"
            )

        def _dtype(key: str) -> torch.dtype | None:
            dtype_name = policy.get(key)
            if dtype_name is None:  # passthrough: keep whatever dtype rollout handed us
                return None
            if dtype_name == "default":  # the run's forward dtype
                return default_dtype
            if dtype_name not in _DTYPES:
                raise ValueError(f"input_dtype_policy[{key!r}] has unknown dtype {dtype_name!r}")
            return _DTYPES[dtype_name]

        return cls(latents=_dtype("latents"), timestep=_dtype("timestep"), cond=_dtype("cond"))


def cast_input(value, dtype: torch.dtype | None):
    """``value`` in ``dtype``; passthrough inputs, int masks and non-tensors stay untouched."""
    if dtype is None or not torch.is_tensor(value) or not value.is_floating_point():
        return value
    return value.to(dtype)
