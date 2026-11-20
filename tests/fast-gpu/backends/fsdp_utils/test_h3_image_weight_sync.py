"""H3 Image weight sync through the trainer's updater: what the rollout engine is sent.

    state dict of the trainer's DiT (plain, or wrapped by PEFT with its LoRA merged in)
        -> weight iterator -> family transform -> buckets
    pdd_proj_out.{0,1,2}.{weight,bias}  ->  final_layer.pdd_fused_heads.video_out.{weight,bias}   [step, ...]

The heads arrive one by one, in buckets of their own here, and leave as two stacks; nothing leaves under a
Diffusers name, which the engine would load without its layout change or skip with a warning.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-3-gpu-h200", labels=[])

import sys
from argparse import Namespace

import diffusers
import pytest
import torch
from peft import LoraConfig, get_peft_model

from miles.backends.fsdp_utils.diffusion_update_weight_utils import (
    DiffusionUpdateWeightFromTensor,
    DiffusionUpdateWeightFromTensorLoRA,
)


class _CaptureFullWeightUpdater(DiffusionUpdateWeightFromTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buckets: list[list[tuple[str, torch.Tensor]]] = []

    def update_bucket_weights(self, named_tensors, target_module):
        self.buckets.append([(name, tensor.clone()) for name, tensor in named_tensors])


class _CaptureMergedLoRAUpdater(DiffusionUpdateWeightFromTensorLoRA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buckets: list[list[tuple[str, torch.Tensor]]] = []

    def update_bucket_weights(self, named_tensors, target_module):
        self.buckets.append([(name, tensor.clone()) for name, tensor in named_tensors])


def _tiny_h3_image():
    torch.manual_seed(7)
    # one attention head of 128: the family transform regroups qkv rows by the release's head width
    return diffusers.MiniMaxH3ImageTransformer2DModel(
        num_attention_heads=1,
        attention_head_dim=128,
        hidden_size=128,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=24,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=24,
        time_embed_dim=16,
        rope_freq_dim=2,
        num_pdd_heads=3,
    ).cuda()


@pytest.mark.parametrize("lora_merged_in", [False, True], ids=["full-weight", "merged-lora"])
def test_a_weight_sync_sends_the_heads_as_two_stacks_and_no_diffusers_name(lora_merged_in):
    model = _tiny_h3_image()
    head_stacks = {
        f"final_layer.pdd_fused_heads.video_out.{suffix}": torch.stack(
            [getattr(head, suffix).detach().clone() for head in model.pdd_proj_out]
        )
        for suffix in ("weight", "bias")
    }
    updater_cls = _CaptureFullWeightUpdater
    if lora_merged_in:
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["to_q", "to_k", "to_v"]))
        updater_cls = _CaptureMergedLoRAUpdater
    # one byte: every tensor closes its bucket, so the held heads outlive the buckets they arrived in
    args = Namespace(update_weight_buffer_size=1, train_pipeline_config_path=None, diffusion_model_family="h3_image")

    updater = updater_cls(args, {"transformer": model})
    updater.update_weights()

    sent = [(name, tensor) for bucket in updater.buckets for name, tensor in bucket]
    names = [name for name, _ in sent]
    assert len(names) == len(set(names))
    assert not [name for name in names if name.startswith(("transformer_blocks.", "pdd_proj_out."))]
    sent_head_stacks = {name: tensor for name, tensor in sent if "pdd" in name}
    assert sent_head_stacks.keys() == head_stacks.keys()
    for name, stack in head_stacks.items():
        assert torch.equal(sent_head_stacks[name], stack), name


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
