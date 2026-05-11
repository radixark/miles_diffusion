import importlib

import torch

from sglang.multimodal_gen.runtime.layers import layernorm as _layernorm_mod

# sgl-d DiT modules that import apply_qk_norm_with_optional_rope by name.
# Each one needs the name re-bound so monkey-patching layernorm alone isn't enough.
_REBIND_MODULES = (
    "sglang.multimodal_gen.runtime.models.dits.qwen_image",
    "sglang.multimodal_gen.runtime.models.dits.flux",
    "sglang.multimodal_gen.runtime.models.dits.flux_2",
    "sglang.multimodal_gen.runtime.models.dits.zimage",
)


def _patched_apply_qk_norm_with_optional_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm,
    k_norm,
    head_dim: int,
    cos_sin_cache=None,
    *,
    is_neox: bool = False,
    positions=None,
    position_offset: int = 0,
    allow_inplace: bool = True,
):
    # Replace sgl-d's fused qk-norm-rope CUDA kernel (which bypasses the
    # patched RMSNorm.forward) with: patched q_norm/k_norm + diffusers'
    # complex ROPE formula from apply_rotary_emb_qwen(use_real=False).
    q_normed = q_norm(q)
    k_normed = k_norm(k)
    if cos_sin_cache is None:
        return q_normed, k_normed

    # Layout: [cos_half | sin_half] along last dim; each half = head_dim/2.
    half = cos_sin_cache.shape[-1] // 2
    freqs_cis = torch.complex(cos_sin_cache[..., :half], cos_sin_cache[..., half:])

    def _apply(x: torch.Tensor) -> torch.Tensor:
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        f = freqs_cis.unsqueeze(1).to(x.device)
        if f.dim() < x_c.dim():
            f = f.unsqueeze(0)
        return torch.view_as_real(x_c * f).flatten(3).type_as(x)

    return _apply(q_normed), _apply(k_normed)


def apply() -> None:
    _layernorm_mod.apply_qk_norm_with_optional_rope = _patched_apply_qk_norm_with_optional_rope
    for mod_path in _REBIND_MODULES:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        mod.apply_qk_norm_with_optional_rope = _patched_apply_qk_norm_with_optional_rope
