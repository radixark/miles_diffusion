"""Wan-exact eager parity for the fused norm/residual kernels.

diffusers' WanTransformerBlock computes every norm/residual site in fp32 and
rounds once via .type_as (sole exception: the cross-attention residual is a
plain bf16 add). The fused kernels round to bf16 mid-chain instead, ~3e-3 rel
per site on a bit-exact input. The sgld group mirrors SD3's bf16-eager shape
and is anti-parity for Wan; these mirror the Wan fp32-once shape.
"""

import torch
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift, ScaleResidualLayerNormScaleShift


def _ensure_broadcast(mod: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if mod.dim() == ref.dim() - 1:
        return mod.unsqueeze(-2)
    return mod


def _layer_norm_f32(norm: torch.nn.LayerNorm, x_f32: torch.Tensor) -> torch.Tensor:
    weight = norm.weight.float() if norm.weight is not None else None
    bias = norm.bias.float() if norm.bias is not None else None
    return F.layer_norm(x_f32, norm.normalized_shape, weight, bias, norm.eps)


def _lnss_forward(self, x: torch.Tensor, shift=None, scale=None):
    # diffusers: (norm1(x.float()) * (1 + scale) + shift).type_as(x) -- one rounding.
    normed = _layer_norm_f32(self.norm, x.float())
    if shift is None and scale is None:
        return normed.type_as(x)
    scale = _ensure_broadcast(scale, normed).float()
    shift = _ensure_broadcast(shift, normed).float()
    return (normed * (1 + scale) + shift).type_as(x)


def _residual_f32(residual: torch.Tensor, x: torch.Tensor, gate):
    # diffusers: (residual.float() + x * gate).type_as(residual); the ungated
    # cross-attention residual is a plain same-dtype add.
    if isinstance(gate, int):
        assert gate == 1
        return residual + x
    if gate.dim() == 4:
        num_frames = gate.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        gated = (x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)).float() * gate.float()).flatten(1, 2)
    else:
        gated = x.float() * gate.float()
    return (residual.float() + gated).type_as(residual)


def _srlnss_forward(self, residual: torch.Tensor, x: torch.Tensor, gate, shift, scale):
    residual_out = _residual_f32(residual, x, gate)
    normed = _layer_norm_f32(self.norm, residual_out.float())
    if shift is None and scale is None:
        return normed.type_as(residual_out), residual_out
    scale = _ensure_broadcast(scale, normed).float()
    shift = _ensure_broadcast(shift, normed).float()
    return (normed * (1 + scale) + shift).type_as(residual_out), residual_out


def _mul_add_forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0):
    # diffusers ffn residual: (c.float() + a.float() * (k + b.float())).type_as(c)
    if b.dim() == 4:
        num_frames = b.shape[1]
        frame_seqlen = a.shape[1] // num_frames
        gated = (a.unflatten(dim=1, sizes=(num_frames, frame_seqlen)).float() * (k + b.float())).flatten(1, 2)
    else:
        gated = a.float() * (k + b.float())
    return (c.float() + gated).type_as(c)


def _rms_norm_forward(self, x: torch.Tensor, residual=None):
    # Wan norm_q/norm_k are torch.nn.RMSNorm on the train side. NOT sgld's
    # patch_rmsnorm semantics: diffusers-generic RMSNorm rounds before the
    # weight mul, torch.nn.RMSNorm does not.
    assert residual is None, "wan attention rmsnorm never fuses a residual"
    return F.rms_norm(x, (x.shape[-1],), self.weight, self.variance_epsilon)


def _rope_fp32(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # diffusers WanAttnProcessor: fp32 rotation over interleaved pairs with
    # half-size tables, rounded once. The fused kernels rotate in bf16.
    cos = cos.float().unsqueeze(-2)
    sin = sin.float().unsqueeze(-2)
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2).type_as(x)


def _patched_flashinfer_rope_qk_inplace(q, k, cos_sin_cache, is_neox=False):
    # wanvideo builds the cache as cat([cos, sin], dim=-1), each [tokens, D/2].
    assert not is_neox
    half = cos_sin_cache.shape[-1] // 2
    cos, sin = cos_sin_cache[..., :half], cos_sin_cache[..., half:]
    return _rope_fp32(q, cos, sin), _rope_fp32(k, cos, sin)


def _patched_apply_rotary_emb(x, cos, sin, is_neox_style, interleaved=False):
    assert not is_neox_style
    if interleaved and cos.shape[-1] == x.shape[-1]:
        cos = cos[..., ::2]
        sin = sin[..., ::2]
    return _rope_fp32(x, cos, sin)


def _upcast_head_table_pre_hook(module, args, kwargs=None):
    """Keep the root `scale_shift_table + temb` identical on both sides.

    Container fp32: the trainer adds a fp32 temb, the engine a bf16 one -- with a bf16 table the
    engine's sum rounds where the trainer's promotes.

    Values re-rounded to bf16 every call: weight sync writes the fp32 master into this param, but
    the trainer's own forward computes with its bf16 copy.
    """
    table = getattr(module, "scale_shift_table", None)
    if table is None:
        return
    if table.dtype != torch.float32:
        module.scale_shift_table = torch.nn.Parameter(table.data.float(), requires_grad=table.requires_grad)
        table = module.scale_shift_table
    table.data.copy_(table.data.to(torch.bfloat16).to(torch.float32))


def apply() -> None:
    import importlib

    from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm

    LayerNormScaleShift.forward = _lnss_forward
    ScaleResidualLayerNormScaleShift.forward = _srlnss_forward
    MulAdd.forward = _mul_add_forward
    RMSNorm.forward = _rms_norm_forward
    # The wan DiT modules bind the rope entry points at import time.
    for mod_path in (
        "sglang.multimodal_gen.runtime.models.dits.wanvideo",
        "sglang.multimodal_gen.runtime.models.dits.causal_wanvideo",
    ):
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        if hasattr(mod, "apply_flashinfer_rope_qk_inplace"):
            mod.apply_flashinfer_rope_qk_inplace = _patched_flashinfer_rope_qk_inplace
        if hasattr(mod, "_apply_rotary_emb"):
            mod._apply_rotary_emb = _patched_apply_rotary_emb
        for cls_name in ("WanTransformer3DModel",):
            cls = getattr(mod, cls_name, None)
            if cls is not None and not getattr(cls, "_wan_head_table_hooked", False):
                orig_init = cls.__init__

                def _init(self, *a, _orig=orig_init, **kw):
                    _orig(self, *a, **kw)
                    self.register_forward_pre_hook(_upcast_head_table_pre_hook)

                cls.__init__ = _init
                cls._wan_head_table_hooked = True
