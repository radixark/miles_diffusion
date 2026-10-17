"""FLUX.3 image train->rollout weight transform: checkpoint QK-norm names -> the engine's module names.

The engine applies these renames only when it loads from disk; a tensor update matches names verbatim
and skips (with a warning) any name its module lacks, so unrenamed QK-norm weights would never sync.
"""

import torch

from miles.backends.fsdp_utils.adaptations.weight_bridge import register_param_transform

# Mirrors the engine's Flux3ImageDiT._param_renames. Not imported: the engine DiT module pulls GPU kernels,
# and this bridge is imported by every process that loads the FSDP backend, including GPU-less ones.
_RENAMES = {
    ".query_norm.scale": ".query_norm.weight",
    ".key_norm.scale": ".key_norm.weight",
}


def _is_renamed(name: str, param) -> bool:
    return name.endswith(tuple(_RENAMES))


def _to_engine_name(name: str, full: torch.Tensor, model) -> list[tuple[str, torch.Tensor]]:
    for src, dst in _RENAMES.items():
        if name.endswith(src):
            return [(name.removesuffix(src) + dst, full)]
    raise AssertionError(f"{name} matched no FLUX.3 rename")


register_param_transform("flux3_image", _is_renamed, _to_engine_name)
