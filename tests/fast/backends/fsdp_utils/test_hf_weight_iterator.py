"""
FSDP weight iterator names: every key the trainer pushes must read as its Diffusers checkpoint name.

    training state_dict key                                   pushed to sglang-d
    -------------------------------------------------------   ------------------------------
    blocks.0.norm1.weight                                 ->  blocks.0.norm1.weight
    base_model.model.blocks.0.attn.to_q.base_layer.weight ->  blocks.0.attn.to_q.weight
    base_model.model.blocks.0.attn.to_q.lora_A.default.weight -> blocks.0.attn.to_q.lora_A
                     \\______________/          \\________/
                      PEFT LoraModel        adapter + .weight

sglang-d maps checkpoint names onto its own modules (param_names_mapping), so anything
left of PEFT in a pushed name is a weight the engine silently skips.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import pytest

from miles.backends.fsdp_utils.hf_weight_iterator import to_hf_name


@pytest.mark.parametrize(
    "train_name,hf_name",
    [
        ("transformer_blocks.0.norm1.weight", "transformer_blocks.0.norm1.weight"),
        ("base_model.model.transformer_blocks.0.attn.to_q.base_layer.weight", "transformer_blocks.0.attn.to_q.weight"),
        ("base_model.model.transformer_blocks.0.norm1.weight", "transformer_blocks.0.norm1.weight"),
        (
            "base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight",
            "transformer_blocks.0.attn.to_q.lora_A",
        ),
        (
            "base_model.model.transformer_blocks.1.attn.add_k_proj.lora_B.weight",
            "transformer_blocks.1.attn.add_k_proj.lora_B",
        ),
        # H3 keeps Diffusers names here; sglang-d's lora_merge path maps and fuses them
        (
            "base_model.model.transformer_blocks.0.ff.net.0.proj.lora_A.default.weight",
            "transformer_blocks.0.ff.net.0.proj.lora_A",
        ),
    ],
    ids=["plain", "peft-base-layer", "peft-untargeted", "lora-with-adapter", "lora-without-adapter", "h3-ffn-lora"],
)
def test_peft_wrappers_are_stripped(train_name, hf_name):
    assert to_hf_name(train_name) == hf_name
