"""Version-dispatched FSDP param-dtype-map patch."""

import importlib
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy


@dataclass(frozen=True)
class ParamDtypeMixedPrecisionPolicy(MixedPrecisionPolicy):
    param_dtype_map: Mapping[str, torch.dtype] | None = None


# Keyed by torch minor version; each implementation verifies the FSDP2 sources it replaces before patching.
_IMPLEMENTATION_MODULES = {
    "2.11": "miles.backends.fsdp_utils.monkey_patches._fsdp_param_dtype_patch_2_11",
    "2.14": "miles.backends.fsdp_utils.monkey_patches._fsdp_param_dtype_patch_2_14",
}


def apply_param_dtype_map_patch() -> None:
    torch_minor_version = ".".join(torch.__version__.split(".")[:2])
    if torch_minor_version not in _IMPLEMENTATION_MODULES:
        raise RuntimeError(
            f"The Miles FSDP param-dtype patch supports torch {', '.join(_IMPLEMENTATION_MODULES)}, "
            f"got {torch.__version__}"
        )
    importlib.import_module(_IMPLEMENTATION_MODULES[torch_minor_version]).apply_param_dtype_map_patch()


__all__ = [
    "ParamDtypeMixedPrecisionPolicy",
    "apply_param_dtype_map_patch",
]
