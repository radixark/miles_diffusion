"""sgl-d numerical-parity monkey patches for miles training alignment.

Rollout engines select a patch group via ``resolve_rollout_patch_group(args)``;
the scheduler child reads ``MILES_ROLLOUT_PATCH_GROUP`` and calls
``apply_rollout_patch_group``.

- ``sgld``: diffusers / SD3 op parity (RMSNorm, RoPE, attention, …).
- ``ltx``:  LTX-2 ltx_core parity + AV-off (rollout uses official gs=1 path).

Patch modules are imported inside ``apply_*`` only so ``RolloutManager`` (a
CPU-only Ray actor) can import this package without pulling sglang triton kernels.
"""

from __future__ import annotations

import os

ROLLOUT_PATCH_GROUP_ENV = "MILES_ROLLOUT_PATCH_GROUP"
PATCH_GROUP_SGLD = "sgld"
PATCH_GROUP_LTX = "ltx"

# Propagated into Ray rollout workers (see miles/ray/rollout.py).
LTX_ROLLOUT_PATCHES_ENV = "MILES_APPLY_LTX2_LTXCORE_PARITY"


def resolve_rollout_patch_group(args) -> str | None:
    """Return the rollout patch group for this engine, or None."""
    if getattr(args, "apply_sgld_monkey_patches", False):
        return PATCH_GROUP_SGLD

    from miles.backends.sglang_diffusion_utils.configs.ltx import is_ltx_model

    if is_ltx_model(args) and os.environ.get(LTX_ROLLOUT_PATCHES_ENV, "1") == "1":
        return PATCH_GROUP_LTX

    return None


def apply_rollout_patch_group(group: str | None) -> None:
    if group == PATCH_GROUP_SGLD:
        apply_sgld_monkey_patches(include_ltx2_ltxcore=False)
    elif group == PATCH_GROUP_LTX:
        apply_ltx2_rollout_patches()


def apply_sgld_monkey_patches(*, include_ltx2_ltxcore: bool | None = None) -> None:
    from miles.backends.sglang_diffusion_utils.monkey_patches import (
        patch_layernorm_scale_shift,
        patch_mul_add,
        patch_qk_norm_rope,
        patch_rmsnorm,
        patch_scale_residual_layernorm,
        patch_usp_attention,
    )

    patch_rmsnorm.apply()
    patch_layernorm_scale_shift.apply()
    patch_scale_residual_layernorm.apply()
    patch_mul_add.apply()
    patch_usp_attention.apply()
    patch_qk_norm_rope.apply()

    if include_ltx2_ltxcore is None:
        include_ltx2_ltxcore = os.environ.get(LTX_ROLLOUT_PATCHES_ENV, "1") == "1"
    if include_ltx2_ltxcore:
        from miles.backends.sglang_diffusion_utils.monkey_patches import (
            patch_ltx2_ltxcore_parity,
        )

        patch_ltx2_ltxcore_parity.apply()


def apply_ltx2_rollout_patches() -> None:
    """LTX-2 ltx_core parity + video-only train alignment."""
    from miles.backends.sglang_diffusion_utils.monkey_patches import (
        patch_ltx2_disable_av_cross,
        patch_ltx2_ltxcore_parity,
        patch_ltx2_rollout_cond_kwargs,
    )

    patch_ltx2_ltxcore_parity.apply()
    patch_ltx2_disable_av_cross.apply()
    patch_ltx2_rollout_cond_kwargs.apply()
