"""Ensure LTX rollout denoising_env carries text context for miles train replay."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False


def _prompt_embeds_tensor(batch: Any) -> Any | None:
    pe = getattr(batch, "prompt_embeds", None)
    if pe is None:
        return None
    return pe[0] if isinstance(pe, list) else pe


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import (
        LTX2DenoisingStage,
    )

    if not hasattr(LTX2DenoisingStage, "_attach_ltx_rollout_cond_kwargs"):
        logger.warning(
            "LTX2DenoisingStage._attach_ltx_rollout_cond_kwargs is missing; "
            "rollout denoising_env may lack encoder_hidden_states. "
            "Upgrade sglang-diffusion or check the installed version."
        )
        _APPLIED = True
        return

    orig_attach = LTX2DenoisingStage._attach_ltx_rollout_cond_kwargs

    def _attach_ltx_rollout_cond_kwargs(self, ctx, batch):
        orig_attach(self, ctx, batch)
        if not (batch.rollout and batch.rollout_return_denoising_env):
            return
        if ctx.pos_cond_kwargs.get("encoder_hidden_states") is None:
            embeds = _prompt_embeds_tensor(batch)
            if embeds is not None:
                ctx.pos_cond_kwargs["encoder_hidden_states"] = embeds

    LTX2DenoisingStage._attach_ltx_rollout_cond_kwargs = _attach_ltx_rollout_cond_kwargs
    _APPLIED = True
