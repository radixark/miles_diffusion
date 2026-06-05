"""Force an identity LTX-2.3 stage1 guider for train/rollout alignment.

Miles GRPO train side computes ``forward_velocity`` as a **video-only** forward
with no CFG / STG / modality / rescale. The sglang LTX2.3 one-stage rollout,
however, applies a stage1 guider whose parameters default to ``video_cfg_scale=3``
etc. (see ``configs/sample/ltx_2.py``). Those parameters **cannot** be overridden
through ``POST /rollout/generate``: ``SamplingParams.from_user_sampling_params_args``
routes unknown kwargs through the base ``SamplingParams`` class, which rejects
LTX23-only fields. So the rollout-side ``rollout_model_outputs`` are post-guider
velocities that diverge from the train forward (~0.94 cosine, scale≈0.86).

This patch overrides ``LTX2DenoisingStage._get_ltx2_stage1_guider_params`` so the
guider becomes the identity transform:

    pred = cond
         + (cfg-1)*(cond-uncond_text)      # cfg=1   -> 0
         + stg*(cond-uncond_perturbed)     # stg=0   -> 0
         + (modality-1)*(cond-uncond_mod)  # mod=1   -> 0
    pred = rescale(cond, pred, 0.0)        # rescale=0 -> pred unchanged
         => pred == cond  (video-only x0)  => velocity == raw video velocity

Controlled by ``MILES_LTX_IDENTITY_GUIDER`` (default ``"1"``). Set to ``"0"`` to
keep the official guider (e.g. for generation-quality eval, not RL alignment).
"""

from __future__ import annotations

import os
from typing import Any

_APPLIED = False
_ORIG = None

_IDENTITY_GUIDER: dict[str, Any] = {
    "video_cfg_scale": 1.0,
    "video_stg_scale": 0.0,
    "video_rescale_scale": 0.0,
    "video_modality_scale": 1.0,
    "video_skip_step": 0,
    "video_stg_blocks": [],
    "audio_cfg_scale": 1.0,
    "audio_stg_scale": 0.0,
    "audio_rescale_scale": 0.0,
    "audio_modality_scale": 1.0,
    "audio_skip_step": 0,
    "audio_stg_blocks": [],
}


def _identity_enabled() -> bool:
    return os.environ.get("MILES_LTX_IDENTITY_GUIDER", "1") == "1"


def apply() -> None:
    global _APPLIED, _ORIG
    if _APPLIED:
        return

    from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import (
        LTX2DenoisingStage,
    )

    _ORIG = LTX2DenoisingStage._get_ltx2_stage1_guider_params

    def _patched_get_guider(self, batch, server_args, stage):
        result = _ORIG(self, batch, server_args, stage)
        # Only override when guider is active (stage1 returns a dict) and the
        # alignment flag is on. None (non-stage1 / official cfg path) is kept.
        if result is None or not _identity_enabled():
            return result
        merged = dict(result)
        merged.update(_IDENTITY_GUIDER)
        return merged

    LTX2DenoisingStage._get_ltx2_stage1_guider_params = _patched_get_guider
    _APPLIED = True
