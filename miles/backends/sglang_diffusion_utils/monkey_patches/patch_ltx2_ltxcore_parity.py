"""LTX-2 DiT parity patches: align sglang ltx_2.py with miles/ltx_core.

TODO(upstream): remove once sgl-d LTX rollout matches ltx_core AdaLN / temb /
velocity-to-x0 paths natively (train/rollout alignment checks pass without patch).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

_ORIGINALS: dict[str, Any] = {}
_APPLIED = False


def expand_temb_for_hidden(temb: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """Broadcast batch-level temb ``[B, 1, D]`` to ``[B, T, D]`` when uniform."""
    if temb.ndim == 3 and temb.shape[1] == 1 and hidden_states.ndim == 3 and hidden_states.shape[1] > 1:
        return temb.expand(-1, hidden_states.shape[1], -1)
    return temb


def _ltx_pytorch_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
    mask = attn_mask
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def _linear_out(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, module.weight, module.bias)


def _ltxcore_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    try:
        from ltx_core.utils import rms_norm as ltx_rms_norm

        return ltx_rms_norm(x, eps=eps)
    except ImportError:
        return F.rms_norm(x, normalized_shape=(x.shape[-1],), eps=eps)


def _ltxcore_apply_split_rotary_emb(
    x: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    cos, sin = freqs
    try:
        from ltx_core.model.transformer.rope import apply_split_rotary_emb as ltx_apply

        return ltx_apply(x, cos, sin)
    except ImportError:
        return _pytorch_apply_split_rotary_emb(x, cos, sin)


def _pytorch_apply_split_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x_dtype = x.dtype
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        b = x.shape[0]
        _, h, t, _ = cos.shape
        x = x.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    last = x.shape[-1]
    split_x = x.reshape(*x.shape[:-1], 2, last // 2)
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]

    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    first_out = out[..., :1, :]
    second_out = out[..., 1:, :]
    first_out.addcmul_(-sin_u, second_x)
    second_out.addcmul_(sin_u, first_x)

    out = out.reshape(*out.shape[:-2], last)
    if needs_reshape:
        out = out.swapaxes(1, 2).reshape(b, t, -1)
    return out.to(dtype=x_dtype)


def _patched_get_ada_values(
    self,
    scale_shift_table: torch.Tensor,
    batch_size: int,
    timestep: torch.Tensor,
    indices: slice,
) -> tuple[torch.Tensor, ...]:
    num_ada_params = int(scale_shift_table.shape[0])
    ada_values = (
        scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
        + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
    ).unbind(dim=2)
    return ada_values


def _patched_ltx2_adaln_single_forward(
    self,
    timestep: torch.Tensor,
    hidden_dtype: torch.dtype | None = None,
):
    """Match ltx_core AdaLayerNormSingle embedding path."""
    from ltx_core.model.transformer.timestep_embedding import get_timestep_embedding

    t = timestep.reshape(-1).to(dtype=torch.float32)
    t_freq = get_timestep_embedding(
        t,
        256,
        flip_sin_to_cos=True,
        downscale_freq_shift=0,
    )
    if hidden_dtype is not None:
        t_freq = t_freq.to(dtype=hidden_dtype)

    te = self.emb.timestep_embedder
    x = F.silu(_linear_out(te.linear_1, t_freq))
    embedded_timestep = _linear_out(te.linear_2, x).to(dtype=self.linear.weight.dtype)
    out = _linear_out(self.linear, F.silu(embedded_timestep))

    if timestep.ndim == 0:
        batch = 1
    elif timestep.ndim == 1:
        batch = 1
    else:
        batch = timestep.shape[0]
    out = out.view(batch, -1, out.shape[-1])
    embedded_timestep = embedded_timestep.view(batch, -1, embedded_timestep.shape[-1])
    return out, embedded_timestep


def _make_patched_ltx2_attention_forward(orig_forward: Callable[..., torch.Tensor]):
    def _patched_forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        skip_sequence_parallel_override: bool = False,
        gather_context_kv_for_sp: bool = False,
    ) -> torch.Tensor:
        from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
        from sglang.multimodal_gen.runtime.models.dits.ltx_2 import apply_interleaved_rotary_emb

        if get_tp_world_size() > 1 or gather_context_kv_for_sp or self.use_local_attention:
            return orig_forward(
                self,
                x,
                context=context,
                mask=mask,
                pe=pe,
                k_pe=k_pe,
                perturbation_mask=perturbation_mask,
                all_perturbed=all_perturbed,
                skip_sequence_parallel_override=skip_sequence_parallel_override,
                gather_context_kv_for_sp=gather_context_kv_for_sp,
            )

        gate_input = x
        context_ = x if context is None else context
        v = _linear_out(self.to_v, context_)
        use_attention = not all_perturbed

        if use_attention:
            q = _linear_out(self.to_q, x)
            k = _linear_out(self.to_k, context_)

            if self.qk_norm:
                assert self.q_norm is not None and self.k_norm is not None
                q = self.q_norm(q)
                k = self.k_norm(k)

            if pe is not None:
                cos, sin = pe
                k_cos, k_sin = pe if k_pe is None else k_pe
                if cos.dim() == 3:
                    q = apply_interleaved_rotary_emb(q, (cos, sin))
                    k = apply_interleaved_rotary_emb(k, (k_cos, k_sin))
                else:
                    q = _ltxcore_apply_split_rotary_emb(q, (cos, sin))
                    k = _ltxcore_apply_split_rotary_emb(k, (k_cos, k_sin))

            out = _ltx_pytorch_sdpa(q, k, v, self.local_heads, mask)

            if perturbation_mask is not None:
                if perturbation_mask.ndim == out.ndim - 1:
                    perturbation_mask = perturbation_mask.unsqueeze(-1)
                out = out * perturbation_mask + v * (1 - perturbation_mask)
        else:
            out = v

        if self.to_gate_logits is not None:
            gate_logits = _linear_out(self.to_gate_logits, gate_input)
            b, t = out.shape[:2]
            out = out.view(b, t, self.local_heads, self.dim_head)
            out = out * (2.0 * torch.sigmoid(gate_logits).unsqueeze(-1))
            out = out.view(b, t, self.local_heads * self.dim_head)

        return _linear_out(self.to_out[0], out)

    return _patched_forward


def _make_patched_ltx2_block_forward(orig_forward: Callable[..., tuple[torch.Tensor, torch.Tensor]]):
    def _patched_forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        args = list(args)
        if len(args) >= 6 and isinstance(args[0], torch.Tensor) and isinstance(args[4], torch.Tensor):
            args[4] = expand_temb_for_hidden(args[4], args[0])
        if len(args) >= 7 and isinstance(args[1], torch.Tensor) and isinstance(args[5], torch.Tensor):
            args[5] = expand_temb_for_hidden(args[5], args[1])
        if "temb" in kwargs:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            if isinstance(hidden_states, torch.Tensor):
                kwargs = dict(kwargs)
                kwargs["temb"] = expand_temb_for_hidden(kwargs["temb"], hidden_states)
        if "temb_audio" in kwargs:
            audio_hidden_states = kwargs.get("audio_hidden_states")
            if audio_hidden_states is None and len(args) >= 2:
                audio_hidden_states = args[1]
            if isinstance(audio_hidden_states, torch.Tensor):
                kwargs = dict(kwargs)
                kwargs["temb_audio"] = expand_temb_for_hidden(kwargs["temb_audio"], audio_hidden_states)
        return orig_forward(self, *args, **kwargs)

    return _patched_forward


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from sglang.multimodal_gen.runtime.models.dits import ltx_2 as ltx2_mod

    if "rms_norm" not in _ORIGINALS:
        _ORIGINALS["rms_norm"] = ltx2_mod.rms_norm

    def _patched_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
        return _ltxcore_rms_norm(x, eps=eps)

    ltx2_mod.rms_norm = _patched_rms_norm

    if "apply_split_rotary_emb" not in _ORIGINALS:
        _ORIGINALS["apply_split_rotary_emb"] = ltx2_mod.apply_split_rotary_emb

    def _patched_apply_split_rotary_emb(
        x: torch.Tensor,
        freqs: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return _ltxcore_apply_split_rotary_emb(x, freqs)

    ltx2_mod.apply_split_rotary_emb = _patched_apply_split_rotary_emb

    adaln_cls = ltx2_mod.LTX2AdaLayerNormSingle
    if "LTX2AdaLayerNormSingle.forward" not in _ORIGINALS:
        _ORIGINALS["LTX2AdaLayerNormSingle.forward"] = adaln_cls.forward
    adaln_cls.forward = _patched_ltx2_adaln_single_forward

    block_cls = ltx2_mod.LTX2TransformerBlock
    if "LTX2TransformerBlock.get_ada_values" not in _ORIGINALS:
        _ORIGINALS["LTX2TransformerBlock.get_ada_values"] = block_cls.get_ada_values
    block_cls.get_ada_values = _patched_get_ada_values

    if "LTX2TransformerBlock.forward" not in _ORIGINALS:
        _ORIGINALS["LTX2TransformerBlock.forward"] = block_cls.forward
    block_cls.forward = _make_patched_ltx2_block_forward(block_cls.forward)

    attn_cls = ltx2_mod.LTX2Attention
    if "LTX2Attention.forward" not in _ORIGINALS:
        _ORIGINALS["LTX2Attention.forward"] = attn_cls.forward
    attn_cls.forward = _make_patched_ltx2_attention_forward(attn_cls.forward)

    _APPLIED = True
