"""Ensure LTX rollout denoising_env carries text/audio embeds for miles train replay.

TODO(upstream): remove once sgl-d LTX rollout returns full cond kwargs in the
standard denoising_env schema without miles-side postprocessing.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False


def _first_batch_tensor(batch: Any, attr: str) -> Any | None:
    value = getattr(batch, attr, None)
    if value is None:
        return None
    return value[0] if isinstance(value, list) else value


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import LTX2DenoisingStage

    if not hasattr(LTX2DenoisingStage, "_prepare_denoising_loop"):
        logger.warning(
            "LTX2DenoisingStage._prepare_denoising_loop is missing; "
            "rollout denoising_env may lack text/audio cond kwargs."
        )
        _APPLIED = True
        return

    orig_prepare = LTX2DenoisingStage._prepare_denoising_loop

    def _prepare_denoising_loop(self, batch, server_args):
        ctx = orig_prepare(self, batch, server_args)
        if not (batch.rollout and batch.rollout_return_denoising_env):
            return ctx
        ctx.pos_cond_kwargs = dict(ctx.pos_cond_kwargs)
        if ctx.pos_cond_kwargs.get("encoder_hidden_states") is None:
            embeds = _first_batch_tensor(batch, "prompt_embeds")
            if embeds is not None:
                ctx.pos_cond_kwargs["encoder_hidden_states"] = embeds
        if ctx.pos_cond_kwargs.get("audio_encoder_hidden_states") is None:
            audio_embeds = _first_batch_tensor(batch, "audio_prompt_embeds")
            if audio_embeds is not None:
                ctx.pos_cond_kwargs["audio_encoder_hidden_states"] = audio_embeds
        attn_mask = getattr(batch, "prompt_attention_mask", None)
        if attn_mask is not None:
            if ctx.pos_cond_kwargs.get("encoder_attention_mask") is None:
                ctx.pos_cond_kwargs["encoder_attention_mask"] = attn_mask
            if ctx.pos_cond_kwargs.get("audio_encoder_attention_mask") is None:
                ctx.pos_cond_kwargs["audio_encoder_attention_mask"] = attn_mask
        return ctx

    LTX2DenoisingStage._prepare_denoising_loop = _prepare_denoising_loop
    _APPLIED = True
