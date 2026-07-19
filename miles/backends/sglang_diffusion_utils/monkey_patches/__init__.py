"""sgl-d numerical-parity monkey patches for miles training alignment.

Patch groups align sgl-d rollout ops/models with the training-side forward.
The engine parent lists selected group names in ``MILES_ROLLOUT_PATCH_GROUPS``;
the sglang scheduler grandchild (spawn: fresh imports) re-reads it and applies
those groups before model construction.

- ``sgld``: diffusers / SD3 op parity (RMSNorm, LayerNormScaleShift, MulAdd,
  ...). Op-layer patches: they apply to every sgl-d DiT built from these
  generic classes. Attention is NOT patched: overriding USPAttention.forward
  breaks bitwise SP-invariance (kernel choice depends on head/batch shape) —
  align the attention kernel via the attention-backend selection instead.
- ``ltx``:  LTX rollout cond kwargs + AV cross-off (video-only train parity).

Patch modules are imported inside ``apply_*`` only, so CPU-only Ray actors
can import this package without pulling sglang triton kernels. Adding a
group = one ``@register_rollout_patch_group("<name>")``-decorated apply fn.
"""

from __future__ import annotations

import os
from collections.abc import Callable

# Comma-separated group names selected by the engine parent, e.g. "sgld,ltx".
ROLLOUT_PATCH_GROUPS_ENV = "MILES_ROLLOUT_PATCH_GROUPS"

_ROLLOUT_PATCH_APPLIERS: dict[str, Callable[[], None]] = {}


def register_rollout_patch_group(name: str):
    """Decorator: register a patch group's apply fn under a group name."""

    def wrapper(fn: Callable[[], None]) -> Callable[[], None]:
        _ROLLOUT_PATCH_APPLIERS[name] = fn
        return fn

    return wrapper


@register_rollout_patch_group("sgld")
def apply_sgld_monkey_patches() -> None:
    from miles.backends.sglang_diffusion_utils.monkey_patches import (
        patch_layernorm_scale_shift,
        patch_mul_add,
        patch_qk_norm_rope,
        patch_rmsnorm,
        patch_scale_residual_layernorm,
    )

    patch_rmsnorm.apply()
    patch_layernorm_scale_shift.apply()
    patch_scale_residual_layernorm.apply()
    patch_mul_add.apply()
    patch_qk_norm_rope.apply()


@register_rollout_patch_group("ltx")
def apply_ltx2_rollout_patches() -> None:
    from miles.backends.sglang_diffusion_utils.monkey_patches import (
        patch_ltx2_disable_av_cross,
        patch_ltx2_rollout_cond_kwargs,
    )

    patch_ltx2_rollout_cond_kwargs.apply()
    patch_ltx2_disable_av_cross.apply()


def apply_env_selected_rollout_patches() -> None:
    """Apply every group named in the env list (runs in the scheduler grandchild)."""
    for name in filter(None, os.environ.get(ROLLOUT_PATCH_GROUPS_ENV, "").split(",")):
        applier = _ROLLOUT_PATCH_APPLIERS.get(name)
        if applier is None:
            raise ValueError(f"Unknown rollout patch group {name!r}; known: {list(_ROLLOUT_PATCH_APPLIERS)}")
        applier()
