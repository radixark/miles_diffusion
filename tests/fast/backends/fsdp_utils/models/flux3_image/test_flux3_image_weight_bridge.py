"""
FLUX.3 image full-weight sync: QK-norm weights must reach the engine under its module names.

The trainer holds the checkpoint names (``*.norm.query_norm.scale``); the engine module stores
``*.norm.query_norm.weight`` and a tensor update skips names it does not have, so without the
rename the rollout keeps the initial QK-norm forever while every other weight syncs.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import torch

from miles.backends.fsdp_utils.adaptations import get_param_transform
from miles.backends.fsdp_utils.models.flux3_image.weight_bridge import _to_engine_name


def test_qk_norm_scale_is_renamed_to_weight():
    scale = torch.arange(4, dtype=torch.float32)
    for norm in ("query_norm", "key_norm"):
        [(name, tensor)] = _to_engine_name(f"single_blocks.3.norm.{norm}.scale", scale, None)
        assert name == f"single_blocks.3.norm.{norm}.weight"
        assert tensor is scale


def test_bridge_registers_only_the_renamed_params():
    # registration rides on adaptations' last import; a family missing there would sync untransformed
    assert get_param_transform("single_blocks.0.norm.key_norm.scale", torch.zeros(1), "flux3_image") is _to_engine_name
    for name in ("single_blocks.0.q_proj.weight", "final_layer.image.linear.weight"):
        assert get_param_transform(name, torch.zeros(1), "flux3_image") is None
