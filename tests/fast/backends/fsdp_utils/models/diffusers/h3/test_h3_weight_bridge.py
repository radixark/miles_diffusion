"""
H3 full-weight sync: the Diffusers weights the trainer holds -> the FL2VA layout the rollout loads.

    transformer_blocks.0.attn.to_q.weight --+   held until all three arrive
    transformer_blocks.0.attn.to_v.weight --+   (each arrives in its own call)
    transformer_blocks.0.attn.to_k.weight --+--> blocks.0.attn.qkv_proj.weight
                                                 rows per head: [q_0 k_0 v_0 | q_1 k_1 v_1 | ...]

Pushing [q; k; v] instead of per-head rows has the same shape, so the native qkv loader
accepts it and silently scrambles attention.

The transform is registered for family "h3" by the last import in fsdp_utils/adaptations,
so it applies whenever the weight iterator uses the bridge.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import torch

from miles.backends.fsdp_utils.adaptations import get_param_transform
from miles.backends.fsdp_utils.models.diffusers.h3.weight_bridge import _to_fl2va

HEAD_DIM = 128


def _distinct_rows(first_value: int, num_heads: int) -> torch.Tensor:
    # every row holds its own value, so a misplaced row shows up in the comparison
    return torch.arange(first_value, first_value + num_heads * HEAD_DIM, dtype=torch.float32).unsqueeze(1)


def test_qkv_is_held_until_complete_then_fused_per_head():
    q, k, v = _distinct_rows(0, 2), _distinct_rows(1000, 2), _distinct_rows(2000, 2)

    assert _to_fl2va("transformer_blocks.0.attn.to_q.weight", q, None) == []
    assert _to_fl2va("transformer_blocks.0.attn.to_v.weight", v, None) == []
    [(name, qkv)] = _to_fl2va("transformer_blocks.0.attn.to_k.weight", k, None)

    assert name == "blocks.0.attn.qkv_proj.weight"
    head_0, head_1 = slice(0, HEAD_DIM), slice(HEAD_DIM, 2 * HEAD_DIM)
    per_head_rows = torch.cat([q[head_0], k[head_0], v[head_0], q[head_1], k[head_1], v[head_1]])
    torch.testing.assert_close(qkv, per_head_rows)


def test_bridge_registers_h3_on_import():
    # registration rides on adaptations' last import; a family missing there would sync untransformed
    assert get_param_transform("transformer_blocks.0.attn.to_q.weight", torch.zeros(1), "h3") is _to_fl2va
