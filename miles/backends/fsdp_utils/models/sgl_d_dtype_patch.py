"""Force the sglang-diffusion DenoisingStage to honor ``pipeline_config.dit_precision``.

Why this exists: ``DenoisingStage._prepare_denoising_loop`` hardcodes
``target_dtype = torch.bfloat16`` and wraps the entire denoising loop in
``torch.autocast(dtype=target_dtype)``. For SD3 (which is fp16-trained) the
trainer-side recompute runs FSDP fp16 forward, so SGLang's bf16 autocast
forward and the trainer's fp16 forward produce systematically different
``noise_pred`` (~4-5e-2 abs mean) → systematic ``log_prob_old`` vs
``log_prob_new`` mismatch and inflated ``approx_kl`` / ``ratio_abs_minus_1``.

Fix: post-process the DenoisingContext returned by the original method to
override ``target_dtype`` (and ``autocast_enabled``) using
``pipeline_config.dit_precision``, and re-cast floating tensors that the
original prepare path already cast at bf16 so the rest of the loop sees the
right dtype. Targets only fp16/bf16/fp32 mappings; falls through unchanged
when the original (bf16) already matches the configured precision.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_PATCH_APPLIED_ATTR = "_sgl_d_dit_precision_patch_applied"


def _cast_floats_in_dict(d: dict[str, Any], dtype: torch.dtype) -> None:
    """Cast every floating-point tensor in ``d`` to ``dtype`` in place.

    Mirrors what ``_prepare_denoising_loop`` already did at bf16 for fields
    fed to the DiT forward (``encoder_hidden_states``, ``pooled_projections``,
    rotary embeddings, etc.). Bool/int masks and non-tensor values are left
    untouched.
    """
    if not d:
        return
    for k, v in list(d.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            d[k] = v.to(dtype)
        elif isinstance(v, list):
            d[k] = [
                x.to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
                for x in v
            ]


def apply_sgl_d_dit_precision_patch() -> None:
    """Monkey patch ``DenoisingStage._prepare_denoising_loop`` so it honors
    ``pipeline_config.dit_precision``. Idempotent.
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages import (
        denoising as _denoising_mod,
    )
    from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

    stage_cls = _denoising_mod.DenoisingStage
    if getattr(stage_cls, _PATCH_APPLIED_ATTR, False):
        return

    original = stage_cls._prepare_denoising_loop

    def patched_prepare(self, batch, server_args):  # type: ignore[no-redef]
        ctx = original(self, batch, server_args)
        try:
            precision = server_args.pipeline_config.dit_precision
        except AttributeError:
            return ctx
        target_dtype = PRECISION_TO_TYPE.get(precision)
        if target_dtype is None or target_dtype == ctx.target_dtype:
            return ctx

        old_dtype = ctx.target_dtype
        ctx.target_dtype = target_dtype
        ctx.autocast_enabled = (
            target_dtype != torch.float32
        ) and not server_args.disable_autocast

        _cast_floats_in_dict(ctx.image_kwargs, target_dtype)
        _cast_floats_in_dict(ctx.pos_cond_kwargs, target_dtype)
        _cast_floats_in_dict(ctx.neg_cond_kwargs, target_dtype)
        if isinstance(ctx.guidance, torch.Tensor) and ctx.guidance.is_floating_point():
            ctx.guidance = ctx.guidance.to(target_dtype)
        if isinstance(ctx.z, torch.Tensor) and ctx.z.is_floating_point():
            ctx.z = ctx.z.to(target_dtype)

        logger.info(
            "[sgl-d dtype patch] overrode DenoisingStage target_dtype %s -> %s "
            "(pipeline_config.dit_precision=%r)",
            old_dtype,
            target_dtype,
            precision,
        )
        return ctx

    patched_prepare.__qualname__ = original.__qualname__
    patched_prepare.__name__ = original.__name__

    stage_cls._prepare_denoising_loop = patched_prepare
    setattr(stage_cls, _PATCH_APPLIED_ATTR, True)
    logger.info(
        "[sgl-d dtype patch] DenoisingStage._prepare_denoising_loop patched to "
        "honor pipeline_config.dit_precision (was hardcoded torch.bfloat16)."
    )
