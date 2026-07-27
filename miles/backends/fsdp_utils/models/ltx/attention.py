"""LTX attention backend selection."""

from __future__ import annotations

import torch


def set_attention_backend(model: torch.nn.Module, backend: str) -> None:
    from ltx_core.loader.attention_ops import set_attention_module_op
    from ltx_core.model.transformer.attention import AttentionFunction, MaskedAttentionFunction

    aliases = {
        "fa3": "FLASH_ATTENTION_3",
        "fa4": "FLASH_ATTENTION_4",
        "sdpa": "PYTORCH",
        "native": "PYTORCH",
        "math": "SDPA_MATH",
        "sdpa_math": "SDPA_MATH",
    }
    name = aliases.get(backend.strip().lower(), backend.strip().upper())
    if name not in AttentionFunction.__members__:
        valid = ", ".join(m.name.lower() for m in AttentionFunction)
        raise ValueError(
            f"LTX --fsdp-attention-backend='{backend}' is not an ltx_core backend; "
            f"choose one of {{{valid}}} (aliases: fa3, fa4, sdpa)."
        )
    masked = MaskedAttentionFunction[name] if name in MaskedAttentionFunction.__members__ else None
    set_attention_module_op(attention=AttentionFunction[name], masked_attention=masked).mutator(model)
