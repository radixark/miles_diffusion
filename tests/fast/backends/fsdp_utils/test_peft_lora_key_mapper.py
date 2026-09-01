from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.backends.fsdp_utils.diffusion_update_weight_utils import PeftLoRAKeyMapper

_QWEN_A = "base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight"
_QWEN_B = "base_model.model.transformer_blocks.0.attn.to_q.lora_B.default.weight"
_SD3_A = "base_model.model.transformer_blocks.1.attn.add_k_proj.lora_A.weight"
_MLP_B = "base_model.model.transformer_blocks.2.img_mlp.net.0.proj.lora_B.default.weight"


class TestPeftLoRAKeyMapper:
    @pytest.mark.parametrize(
        "key,expected",
        [
            (_QWEN_A, "transformer_blocks.0.attn.to_q.lora_A"),
            (_QWEN_B, "transformer_blocks.0.attn.to_q.lora_B"),
            (_SD3_A, "transformer_blocks.1.attn.add_k_proj.lora_A"),
            (_MLP_B, "transformer_blocks.2.img_mlp.net.0.proj.lora_B"),
            ("transformer_blocks.0.attn.to_v.lora_A.default.weight", "transformer_blocks.0.attn.to_v.lora_A"),
        ],
    )
    def test_to_sgld_name_maps_peft_keys(self, key, expected):
        assert PeftLoRAKeyMapper.to_sgld_name(key) == expected

    @pytest.mark.parametrize(
        "key",
        [
            "transformer_blocks.0.attn.to_q.weight",
            "base_model.model.norm.weight",
            "lora_A.default.weight",
        ],
    )
    def test_to_sgld_name_returns_none_for_non_lora_keys(self, key):
        assert PeftLoRAKeyMapper.to_sgld_name(key) is None

    def test_is_lora_key(self):
        assert PeftLoRAKeyMapper.is_lora_key(_QWEN_A)
        assert PeftLoRAKeyMapper.is_lora_key(_QWEN_B)
        assert not PeftLoRAKeyMapper.is_lora_key("transformer_blocks.0.attn.to_q.weight")

    def test_collect_sgld_names_and_layer_prefixes(self):
        state_dict = {
            _QWEN_A: torch.zeros(4, 8),
            _QWEN_B: torch.zeros(8, 4),
            "base_model.model.norm.weight": torch.zeros(8),
        }
        sgld_names = PeftLoRAKeyMapper.collect_sgld_names(state_dict)
        assert sgld_names == {
            "transformer_blocks.0.attn.to_q.lora_A",
            "transformer_blocks.0.attn.to_q.lora_B",
        }
        assert PeftLoRAKeyMapper.collect_layer_prefixes(state_dict) == {"transformer_blocks.0.attn.to_q"}

    def test_summarize_mapping_reports_unmapped_lora_keys(self):
        state_dict = {
            _QWEN_A: torch.zeros(4, 8),
            "base_model.model.weird.lora_A.default.weight.extra": torch.zeros(4, 8),
        }
        num_tensors, num_layers, sample_layers, unmapped = PeftLoRAKeyMapper.summarize_mapping(state_dict)
        assert num_tensors == 1
        assert num_layers == 1
        assert sample_layers == ["transformer_blocks.0.attn.to_q"]
        assert unmapped == ["base_model.model.weird.lora_A.default.weight.extra"]

    def test_h3_peft_keys_stay_diffusers_shaped(self):
        """H3 IPC no longer pre-fuses names; sglang-d maps transformer_blocks → blocks."""
        state_dict = {
            "base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.zeros(4, 8),
            "base_model.model.transformer_blocks.0.attn.to_q.lora_B.default.weight": torch.zeros(8, 4),
            "base_model.model.transformer_blocks.0.ff.net.0.proj.lora_A.default.weight": torch.zeros(4, 8),
            "base_model.model.transformer_blocks.0.ff.net.0.proj.lora_B.default.weight": torch.zeros(8, 4),
            "base_model.model.token_refiner.refiner_blocks.1.attn.to_k.lora_A.default.weight": torch.zeros(4, 8),
            "base_model.model.token_refiner.refiner_blocks.1.attn.to_k.lora_B.default.weight": torch.zeros(8, 4),
        }
        assert PeftLoRAKeyMapper.collect_sgld_names(state_dict) == {
            "transformer_blocks.0.attn.to_q.lora_A",
            "transformer_blocks.0.attn.to_q.lora_B",
            "transformer_blocks.0.ff.net.0.proj.lora_A",
            "transformer_blocks.0.ff.net.0.proj.lora_B",
            "token_refiner.refiner_blocks.1.attn.to_k.lora_A",
            "token_refiner.refiner_blocks.1.attn.to_k.lora_B",
        }
