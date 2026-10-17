"""DCP LoRA key mapping -> loader names.

    model_state.model.base_model.model.<m>.lora_A.default.weight
        -> <component>.<m>.lora_A.weight        (nested dicts flattened first)

Pins: renaming, non-LoRA keys skipped, foreign layouts and LoRA-free states rejected, and the
component resolved from meta.json (H3 Ref2VA trains transformer_ref).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import json

import pytest
import torch

from miles.backends.fsdp_utils.lora_export import checkpoint_component, convert_lora_state


def _peft_state():
    return {
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


def test_maps_peft_dcp_names_to_loader_names():
    assert set(convert_lora_state(_peft_state())) == {
        "transformer.blocks.0.attn.to_q.lora_A.weight",
        "transformer.blocks.0.attn.to_q.lora_B.weight",
    }
    assert set(convert_lora_state(_peft_state(), "transformer_ref")) == {
        "transformer_ref.blocks.0.attn.to_q.lora_A.weight",
        "transformer_ref.blocks.0.attn.to_q.lora_B.weight",
    }


def test_rejects_foreign_layout_and_empty_state():
    with pytest.raises(ValueError, match="unexpected LoRA key layout"):
        convert_lora_state({"oops": {"lora_A": {"weight": torch.zeros(1)}}})
    with pytest.raises(ValueError, match="no LoRA tensors"):
        convert_lora_state({"model_state": {"model": {"w": torch.zeros(1)}}})


def _write_meta(ckpt_dir, modules):
    (ckpt_dir / "meta.json").write_text(json.dumps({"iteration": 1, "update_weight_target_modules": modules}))


def test_component_comes_from_meta_json(tmp_path):
    assert checkpoint_component(tmp_path) == "transformer"
    assert checkpoint_component(tmp_path, "transformer_ref") == "transformer_ref"

    _write_meta(tmp_path, ["transformer_ref"])
    assert checkpoint_component(tmp_path) == "transformer_ref"
    assert checkpoint_component(tmp_path, "transformer_ref") == "transformer_ref"
    with pytest.raises(ValueError, match="does not match"):
        checkpoint_component(tmp_path, "transformer")

    _write_meta(tmp_path, ["transformer", "transformer_2"])
    with pytest.raises(ValueError, match="one trained module"):
        checkpoint_component(tmp_path)
