"""Per-model adaptation layer for the FSDP backend.

The ``weight_bridge`` registry plus the model families that register their transforms into it; a family
plugs in through ``models/diffusers/<family>/weight_bridge.py`` and the import below.
"""

from .weight_bridge import ParamTransform, get_param_transform, register_param_transform

# MUST be last: importing a family's weight_bridge registers its ParamTransform into the registry above.
from ..models.diffusers.h3 import weight_bridge as _h3_weight_bridge  # noqa: F401,E402  # isort: skip

__all__ = [
    "ParamTransform",
    "get_param_transform",
    "register_param_transform",
]
