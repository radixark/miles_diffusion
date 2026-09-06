"""FlashAttention-3 bindings owned by the trainer.

diffusers wraps FA3 in a torch custom op (``_diffusers_flash_attn_3``) that registers
no autograd formula and hardcodes ``deterministic=False``, so a model trained with
``--fsdp-attention-backend _flash_3`` cannot backward through it and could not opt
into FA3's deterministic backward if it did. The trainer therefore binds
``flash_attn_interface`` itself: a diffusers-compatible dense attention that the
model's own processors dispatch to, and the fused forward/backward pair torch's
ring templates drive under USP.
"""

import torch

try:
    import flash_attn_interface as _fa3
except ImportError:  # pragma: no cover - exercised only on hosts without the FA3 wheel
    _fa3 = None


def is_available() -> bool:
    return _fa3 is not None


def _require():
    if _fa3 is None:
        raise RuntimeError("FlashAttention-3 (flash_attn_interface) is not installed")
    return _fa3


def flash3_attention(query, key, value, *, scale=None, causal=False, deterministic=False, return_lse=False):
    """Differentiable FA3 on [B, S, H, D] tensors.

    ``deterministic`` selects FA3's deterministic backward (its forward always is).
    With ``return_lse`` the row logsumexp comes back too, in FA3's [B, H, S] layout.
    """
    fa3 = _require()
    return fa3.flash_attn_func(
        query,
        key,
        value,
        softmax_scale=scale,
        causal=causal,
        deterministic=deterministic,
        return_attn_probs=return_lse,
    )


def ring_forward_op(query, key, value, *, is_causal, scale):
    """Local step of torch's ring template: [B, H, S, D] q/k/v -> (out [B, H, S, D], lse [B, H, S])."""
    out, lse = flash3_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        scale=scale,
        causal=is_causal,
        return_lse=True,
    )
    return out.transpose(1, 2), lse


def ring_backward_op(*, grad_out, query, key, value, out, logsumexp, is_causal, scale, deterministic):
    """Backward step of torch's ring template, all tensors [B, H, S, D] (lse [B, H, S]).

    ``out``/``logsumexp`` are the ring-merged results over every KV shard: FA3 recovers
    the probabilities from ``exp(qk - lse)`` and the row term from ``dot(grad_out, out)``,
    so each KV step's partial dQ/dK/dV comes out exactly as in the undistributed backward.
    """
    fa3 = _require()
    q, k, v, o, do = (t.transpose(1, 2).contiguous() for t in (query, key, value, out, grad_out))
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    # Positional layout mirrors flash_attn_interface.FlashAttnFunc.backward: the six
    # None slots are the varlen cu_seqlens/seqused/max_seqlen arguments.
    fa3._flash_attn_backward(
        do,
        q,
        k,
        v,
        o,
        logsumexp.contiguous(),
        None,
        None,
        None,
        None,
        None,
        None,
        dq,
        dk,
        dv,
        scale,
        is_causal,
        deterministic=deterministic,
    )
    return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2)


def _diffusers_backend(deterministic):
    def flash_attention_3(
        query,
        key,
        value,
        attn_mask=None,
        scale=None,
        is_causal=False,
        return_lse=False,
        _parallel_config=None,
    ):
        if attn_mask is not None:
            raise ValueError("`attn_mask` is not supported for flash-attn 3.")
        result = flash3_attention(
            query, key, value, scale=scale, causal=is_causal, deterministic=deterministic, return_lse=return_lse
        )
        if not return_lse:
            return result
        out, lse = result
        return out, lse.permute(0, 2, 1)  # diffusers returns lse as [B, S, H]

    flash_attention_3._miles_deterministic = deterministic
    return flash_attention_3


def install_diffusers_backend(*, deterministic: bool) -> None:
    """Re-register diffusers' ``_flash_3`` backend on the trainer's binding.

    Keeps upstream's input constraints; only the kernel callable changes, so the model's
    processors (self- and cross-attention alike) get a differentiable FA3 whose backward
    honors ``deterministic``. Idempotent per flag value.
    """
    import diffusers.models.attention_dispatch as ad

    _require()
    name = ad.AttentionBackendName._FLASH_3
    registry = ad._AttentionBackendRegistry
    current = registry._backends.get(name)
    if getattr(current, "_miles_deterministic", None) == deterministic:
        return
    registry.register(name, constraints=registry._constraints.get(name))(_diffusers_backend(deterministic))
