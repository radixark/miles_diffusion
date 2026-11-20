"""
H3 Image full-weight sync: the per-step output heads the trainer holds -> the stacks the rollout serves.

    pdd_proj_out.0.weight --+   held until the heads of every step arrive
    pdd_proj_out.1.weight --+   (each arrives in its own call)
    pdd_proj_out.2.weight --+--> final_layer.pdd_fused_heads.video_out.weight   [step, out, in]

sglang-d keeps the heads outside its parameters and takes a stack by that name only: a head sent under its
trainer name is skipped with a warning, and the rollout keeps denoising with the heads it started with.

Every other weight goes through H3's transform, registered for family "h3_image" beside the heads'.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils.adaptations import get_param_transform
from miles.backends.fsdp_utils.models.diffusers.h3.weight_bridge import _to_fl2va
from miles.backends.fsdp_utils.models.diffusers.h3_image.weight_bridge import _to_fused_pdd_heads

NUM_STEPS = 3
MODEL = SimpleNamespace(pdd_proj_out=[None] * NUM_STEPS)


def _head(step: int, suffix: str) -> torch.Tensor:
    # every head holds its own value, so a head in another step's place shows up in the comparison
    return torch.full((4, 5) if suffix == "weight" else (4,), float(step))


@pytest.mark.parametrize("suffix", ["weight", "bias"])
def test_heads_are_held_until_every_step_arrived_then_stacked_by_step(suffix):
    assert _to_fused_pdd_heads(f"pdd_proj_out.2.{suffix}", _head(2, suffix), MODEL) == []
    assert _to_fused_pdd_heads(f"pdd_proj_out.0.{suffix}", _head(0, suffix), MODEL) == []
    [(name, stack)] = _to_fused_pdd_heads(f"pdd_proj_out.1.{suffix}", _head(1, suffix), MODEL)

    assert name == f"final_layer.pdd_fused_heads.video_out.{suffix}"
    torch.testing.assert_close(stack, torch.stack([_head(step, suffix) for step in range(NUM_STEPS)]))


def test_the_next_sync_starts_from_no_held_heads():
    for _sync in range(2):
        units = [_to_fused_pdd_heads(f"pdd_proj_out.{step}.bias", _head(step, "bias"), MODEL) for step in range(3)]
        assert [len(unit) for unit in units] == [0, 0, 1]


def test_bridge_registers_h3_image_on_import():
    # registration rides on adaptations' imports; a family missing there syncs untransformed, heads included
    assert get_param_transform("pdd_proj_out.0.weight", torch.zeros(1), "h3_image") is _to_fused_pdd_heads
    assert get_param_transform("transformer_blocks.0.attn.to_q.weight", torch.zeros(1), "h3_image") is _to_fl2va
    # the heads' transform is H3 Image's alone: H3 has one output head, which its own transform renames
    assert get_param_transform("pdd_proj_out.0.weight", torch.zeros(1), "h3") is _to_fl2va
