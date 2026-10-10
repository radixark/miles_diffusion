"""H3 train->rollout weight transform: the Diffusers weights the trainer holds -> the FL2VA native layout
GRPO rollouts load."""

import re

import torch
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _ARCH_DEFAULTS, _diffusers_h3_checkpoint

from miles.backends.fsdp_utils.adaptations.weight_bridge import register_param_transform

_QKV_NAME = re.compile(r"^(.+\.attn)\.to_[qkv]\.weight$")
_PENDING_QKV: dict[str, dict[str, torch.Tensor]] = {}


def _to_fl2va(name: str, full: torch.Tensor, model) -> list[tuple[str, torch.Tensor]]:
    """Rewrite one Diffusers H3 weight into the FL2VA native checkpoint layout the rollout DiT loads.

    Each weight goes through sglang-d's own Diffusers converter (native names, SwiGLU [value, gate] ->
    [gate, value]). FL2VA fuses an attention's to_q / to_k / to_v into one qkv_proj, and each weight arrives
    in its own call, so stash them and emit the fused tensor once the attention is complete.
    """
    qkv = _QKV_NAME.match(name)
    if qkv is None:
        return list(_diffusers_h3_checkpoint([(name, full)]))
    pieces = _PENDING_QKV.setdefault(qkv[1], {})
    pieces[name] = full
    if len(pieces) < 3:
        return []
    del _PENDING_QKV[qkv[1]]
    ((fused_name, fused),) = _diffusers_h3_checkpoint(pieces.items())
    # FL2VA stores qkv rows per head as [q_h, k_h, v_h]; the native qkv loader regroups them on load
    q, k, v = (t.unflatten(0, (-1, _ARCH_DEFAULTS.attention_head_dim)) for t in fused.chunk(3))
    return [(fused_name, torch.stack([q, k, v], dim=1).flatten(0, 2))]


# every H3 weight: rollouts load the FL2VA native partition, not the Diffusers layout the trainer holds
register_param_transform("h3", lambda name, param: True, _to_fl2va)
