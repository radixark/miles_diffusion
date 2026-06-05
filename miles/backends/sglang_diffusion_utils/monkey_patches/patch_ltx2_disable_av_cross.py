"""Disable LTX A2V/V2A cross-attention in sglang rollout (video-only parity).

Miles ltx_core train forward is video-only; the sglang LTX rollout otherwise
runs audio-video cross-attention. Enabled via ``MILES_LTX_DISABLE_AV_CROSS=1``
(set by miles rollout when ``--ltx-disable-av-cross-attn``), this injects the
disable flags into the DiT forward so the rollout video branch matches train.
"""

from __future__ import annotations

import os

_APPLIED = False


def _enabled() -> bool:
    return os.environ.get("MILES_LTX_DISABLE_AV_CROSS", "0") == "1"


def apply() -> None:
    global _APPLIED
    if _APPLIED or not _enabled():
        _APPLIED = True
        return

    from sglang.multimodal_gen.runtime.models.dits import ltx_2 as ltx2_mod

    model_cls = ltx2_mod.LTX2VideoTransformer3DModel
    orig_forward = model_cls.forward

    def forward(self, *args, **kwargs):
        kwargs["disable_a2v_cross_attn"] = True
        kwargs["disable_v2a_cross_attn"] = True
        return orig_forward(self, *args, **kwargs)

    model_cls.forward = forward
    _APPLIED = True
