"""Krea-2's FSDP plan keeps exactly the norm weights diffusers keeps in fp32.

    diffusers Krea2Transformer2DModel._keep_in_fp32_modules  -->  params diffusers loads in fp32
    our FSDP_PARALLEL_PLAN.param_dtype_patterns              -->  params FSDP all-gathers in fp32
    the two sets must be equal, bare and under a PEFT LoRA wrapper (base_model.model.* prefix)
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import fnmatch

import pytest
import torch
from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel
from peft import LoraConfig, get_peft_model

from miles.backends.fsdp_utils.mixed_precision import compile_param_dtype_maps
from miles.backends.fsdp_utils.models.diffusers import load_fsdp_parallel_plan


def _tiny_krea2():
    return Krea2Transformer2DModel(
        in_channels=8,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        timestep_embed_dim=16,
        text_hidden_dim=32,
        num_text_layers=2,
        text_num_attention_heads=2,
        text_num_key_value_heads=2,
        text_intermediate_size=64,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(4, 6, 6),
    )


def _diffusers_fp32_names(model) -> set[str]:
    keep = set(Krea2Transformer2DModel._keep_in_fp32_modules)
    return {name for name, _ in model.named_parameters() if keep.intersection(name.split("."))}


def _plan_fp32_names(model) -> set[str]:
    patterns = load_fsdp_parallel_plan("krea2").param_dtype_patterns
    return {
        name for name, _ in model.named_parameters() if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
    }


@pytest.mark.parametrize("with_lora", [False, True])
def test_plan_matches_the_diffusers_keep_in_fp32_set(with_lora):
    """A diffusers rename or a new norm module would drift the two sets apart silently."""
    model = _tiny_krea2()
    if with_lora:
        model = get_peft_model(
            model, LoraConfig(r=4, lora_alpha=8, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
        )
    expected = _diffusers_fp32_names(model)
    assert expected
    assert _plan_fp32_names(model) == expected


def test_plan_compiles_against_the_fsdp_wraps():
    """An unmatched pattern or a split-dtype local FQN inside one block wrap fails at compile time."""
    model = _tiny_krea2()
    no_split = set(Krea2Transformer2DModel._no_split_modules)
    wraps = [module for module in model.modules() if module.__class__.__name__ in no_split]
    plan = load_fsdp_parallel_plan("krea2")
    maps = compile_param_dtype_maps(model, wraps, plan.param_dtype_patterns, torch.bfloat16)
    assert maps.override_count == len(_diffusers_fp32_names(model))
