"""DCP LoRA key mapping -> loader names.

    model_state.model.base_model.model.<m>.lora_A.default.weight
        -> transformer.<m>.lora_A.weight        (nested dicts flattened first)

Pins: renaming, non-LoRA keys skipped, foreign layouts and LoRA-free states rejected.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.backends.fsdp_utils.lora_export import convert_lora_state


def test_maps_peft_dcp_names_to_loader_names():
    flat = {
        "model_state": {
            "model": {
                "base_model": {
                    "model": {
                        "blocks.0.attn.to_q": {
                            "lora_A": {"default": {"weight": torch.zeros(4, 8)}},
                            "lora_B": {"default": {"weight": torch.zeros(8, 4)}},
                            "base_layer": {"weight": torch.zeros(8, 8)},
                        }
                    }
                }
            }
        }
    }
    out = convert_lora_state(flat)
    assert set(out) == {
        "transformer.blocks.0.attn.to_q.lora_A.weight",
        "transformer.blocks.0.attn.to_q.lora_B.weight",
    }


def test_rejects_foreign_layout_and_empty_state():
    with pytest.raises(ValueError, match="unexpected LoRA key layout"):
        convert_lora_state({"oops": {"lora_A": {"weight": torch.zeros(1)}}})
    with pytest.raises(ValueError, match="no LoRA tensors"):
        convert_lora_state({"model_state": {"model": {"w": torch.zeros(1)}}})
