"""sgl-d → diffusers numerical-parity monkey patches.

Patches sgl-d's generic op classes (RMSNorm, LayerNormScaleShift, MulAdd,
USPAttention, etc.) to match diffusers' bf16 cast/op order. Apply once at
sglang-d scheduler startup so DiT forwards on the rollout side agree with
diffusers-style training-side forwards down to bf16 ULPs.

Patches are at the op layer, not the model layer — they apply to every sgl-d
DiT that uses these generic classes. Adding alignment for a new op = drop a
new ``patch_<op>.py`` file and add it to ``apply_sgld_monkey_patches``.
"""

from miles.backends.sglang_diffusion_utils.monkey_patches import (
    patch_layernorm_scale_shift,
    patch_mul_add,
    patch_qk_norm_rope,
    patch_rmsnorm,
    patch_scale_residual_layernorm,
    patch_usp_attention,
)


def apply_sgld_monkey_patches() -> None:
    patch_rmsnorm.apply()
    patch_layernorm_scale_shift.apply()
    patch_scale_residual_layernorm.apply()
    patch_mul_add.apply()
    patch_usp_attention.apply()
    patch_qk_norm_rope.apply()
