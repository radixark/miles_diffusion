"""H3 train->rollout weight transform: the Diffusers weights the trainer holds -> the FL2VA native layout
GRPO rollouts load."""

import re
from collections import defaultdict
from collections.abc import Iterable, Iterator

import torch
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

from miles.backends.fsdp_utils.adaptations.weight_bridge import register_param_transform

_ARCH_DEFAULTS = MiniMaxH3DiTArchConfig()

_QKV_NAME = re.compile(r"^(.+\.attn)\.to_[qkv]\.weight$")
_PENDING_QKV: dict[str, dict[str, torch.Tensor]] = {}


def _diffusers_h3_checkpoint(
    iterator: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Map Diffusers H3 names/layout to the fused native checkpoint layout.

    Clone of sglang-d's ``_diffusers_h3_checkpoint`` (``dits/minimax_h3.py``); keep the two identical.
    Importing that module loads the whole DiT, whose attention backends need a GPU at import time,
    and the rollout manager has none.
    """
    mapping = get_param_names_mapping(_ARCH_DEFAULTS.param_names_mapping)
    pending: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)

    for source_name, tensor in iterator:
        target_name, merge_index, merge_count = mapping(source_name)

        # Diffusers SwiGLU stores [value, gate]; the native fused MLP consumes
        # [gate, value]. Packed GPTQ tensors carry output channels on dim 1.
        if ".ff.net.0.proj." in source_name:
            output_dim = 1 if source_name.endswith((".qweight", ".qzeros", ".scales")) else 0
            value, gate = tensor.chunk(2, dim=output_dim)
            tensor = torch.cat((gate, value), dim=output_dim)

        if merge_index is None:
            yield target_name, tensor
            continue

        assert merge_count is not None
        pending[target_name][merge_index] = tensor
        if len(pending[target_name]) != merge_count:
            continue

        merge_dim = 1 if target_name.endswith((".qweight", ".qzeros", ".scales")) else 0
        yield (
            target_name,
            torch.cat([pending[target_name][index] for index in range(merge_count)], dim=merge_dim),
        )
        del pending[target_name]

    if pending:
        incomplete = ", ".join(sorted(pending))
        raise ValueError(f"Incomplete Diffusers H3 fused parameters: {incomplete}")


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
