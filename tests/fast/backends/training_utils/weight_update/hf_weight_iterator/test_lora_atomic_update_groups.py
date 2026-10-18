"""
LoRA IPC buckets for a family whose rollout fuses q / k / v (H3: to_q, to_k, to_v -> qkv_proj).

    units (one LoRA module each: lora_A + lora_B), buffer smaller than any unit

    without groups:  [to_q] [to_k] [to_v] [to_out.0]       sglang-d replaces qkv_proj's LoRA per
                                                            request -> only to_v's section survives
    with groups:     [to_q to_k to_v] [to_out.0]           the fused layer gets all three sections

A config that targets only part of a fused projection cannot be composed by sglang-d, so it fails loudly.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups
from miles.backends.training_utils.weight_update.hf_weight_iterator.bucketing import (
    assemble_atomic_update_groups,
    pack_units_by_size,
)


def _adapter_unit(module: str) -> list[tuple[str, torch.Tensor]]:
    return [(f"{module}.lora_A", torch.zeros(4, 8)), (f"{module}.lora_B", torch.zeros(8, 4))]


def _buckets(modules: list[str]) -> list[list[str]]:
    units = assemble_atomic_update_groups(map(_adapter_unit, modules), get_hf_atomic_update_groups("h3"))
    return [[name for name, _ in bucket] for bucket in pack_units_by_size(units, max_bytes=1)]


def test_fused_qkv_adapters_share_one_bucket():
    attention = "transformer_blocks.0.attn"
    buckets = _buckets([f"{attention}.to_q", f"{attention}.to_k", f"{attention}.to_v", f"{attention}.to_out.0"])

    assert [len(bucket) for bucket in buckets] == [6, 2]
    assert {name.rsplit(".", 1)[0] for name in buckets[0]} == {
        f"{attention}.to_q",
        f"{attention}.to_k",
        f"{attention}.to_v",
    }


def test_partially_targeted_fused_projection_fails():
    with pytest.raises(AssertionError, match="Incomplete atomic update groups"):
        _buckets(["transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k"])
