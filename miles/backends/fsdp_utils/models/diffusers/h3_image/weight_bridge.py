"""H3 Image train->rollout weight transform: H3's native-layout transform for the backbone, and the per-step
output heads as the stacks sglang-d's weight update takes them by."""

import re

import torch

from miles.backends.fsdp_utils.adaptations.weight_bridge import register_param_transform
from miles.backends.fsdp_utils.models.diffusers.h3.weight_bridge import _to_fl2va

_PDD_HEAD_NAME = re.compile(r"^pdd_proj_out\.(\d+)\.(weight|bias)$")
# "weight" / "bias" -> the heads that arrived so far, by denoising step
_PENDING_PDD_HEADS: dict[str, dict[int, torch.Tensor]] = {}


def rollout_takes_pdd_fused_heads() -> bool:
    """Whether this sglang-d's weight update takes a per-step head stack (sgl-project/sglang#51654)."""
    try:
        from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (  # noqa: F401
            MINIMAX_H3_PDD_FUSED_HEADS_WEIGHT_NAME_PREFIX,
        )
    except ImportError:
        return False
    return True


def _is_pdd_head(name: str, param) -> bool:
    return _PDD_HEAD_NAME.match(name) is not None


def _to_fused_pdd_heads(name: str, full: torch.Tensor, model) -> list[tuple[str, torch.Tensor]]:
    """Stack the per-step output heads into the one tensor the rollout DiT serves them from.

    The trainer holds pdd_proj_out.{step}.{weight,bias}; sglang-d serves video_out.{weight,bias}[step], the layout of
    the release's pdd_fused_heads.safetensors, and its weight update takes a whole stack. Each head arrives in its own
    call, so hold them until the heads of every step are in.
    """
    step, suffix = _PDD_HEAD_NAME.match(name).groups()
    heads_by_step = _PENDING_PDD_HEADS.setdefault(suffix, {})
    heads_by_step[int(step)] = full
    num_steps = len(model.pdd_proj_out)
    if len(heads_by_step) < num_steps:
        return []
    del _PENDING_PDD_HEADS[suffix]
    # imported here: every family's bridge loads with the adaptations package, also on an sglang-d without it
    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MINIMAX_H3_PDD_FUSED_HEADS_WEIGHT_NAME_PREFIX

    stack = torch.stack([heads_by_step[step] for step in range(num_steps)])
    return [(f"{MINIMAX_H3_PDD_FUSED_HEADS_WEIGHT_NAME_PREFIX}video_out.{suffix}", stack)]


register_param_transform("h3_image", _is_pdd_head, _to_fused_pdd_heads)
# every other weight is H3's: the rollout loads the same native partition without the audio rows
register_param_transform("h3_image", lambda name, param: True, _to_fl2va)
