"""WeightBridge: the registered train->rollout parameter-name/shape contract for the FSDP backend.

HF/FSDP training and the SGLang rollout loader don't always agree on param names/shapes (e.g. the H3 rollout
loads the native FL2VA layout while training holds the Diffusers one). Each disagreeing model registers a
ParamTransform (a ``matches`` selector + an ``expand`` that rewrites the materialized tensor) instead of
editing the sync loop. ``expand`` never touches DTensor/device state, so transforms are CPU-unit-testable.
"""

from collections.abc import Callable, Iterable
from typing import NamedTuple

import torch


class ParamTransform(NamedTuple):
    matches: Callable[[str, object], bool]
    expand: Callable[[str, torch.Tensor, torch.nn.Module], Iterable[tuple[str, torch.Tensor]]]


# diffusion model family -> registered transforms, tried in registration order
_REGISTRY: dict[str, list[ParamTransform]] = {}


def register_param_transform(diffusion_model_family: str, matches: Callable, expand: Callable) -> None:
    _REGISTRY.setdefault(diffusion_model_family, []).append(ParamTransform(matches, expand))


def get_param_transform(name: str, param, diffusion_model_family: str | None):
    """Return the ``expand`` fn for the transform matching this param, or None (passthrough)."""
    for transform in _REGISTRY.get(diffusion_model_family, ()):
        if transform.matches(name, param):
            return transform.expand
    return None
