from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    suite="stage-b-3-gpu-h200",
    labels=[],
)

from argparse import Namespace

import pytest
import torch
from peft import LoraConfig, get_peft_model

from miles.backends.fsdp_utils.diffusion_update_weight_utils import (
    DiffusionUpdateWeightFromTensorLoRA,
    DiffusionUpdateWeightFromTensorLoRAIPC,
    PeftLoRAKeyMapper,
)


class _TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 16, bias=False)
        self.norm = torch.nn.LayerNorm(16)


class _CaptureUpdater(DiffusionUpdateWeightFromTensorLoRA):
    """Capture flushed buckets instead of pushing to rollout engines."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buckets: list[list[tuple[str, torch.Tensor]]] = []

    def wait_and_update_bucket_weights(self, bucket, target_module, weight_update_mode=None):
        self.buckets.append([(name, tensor.clone()) for name, tensor in bucket])


class _CaptureLoRAIPCUpdater(DiffusionUpdateWeightFromTensorLoRAIPC):
    """Capture LoRA IPC buckets without rollout engines."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buckets: list[list[tuple[str, torch.Tensor]]] = []

    def wait_and_update_bucket_weights(self, bucket, target_module, weight_update_mode=None):
        self.buckets.append([(name, tensor.clone()) for name, tensor in bucket])


def _make_peft_model():
    torch.manual_seed(0)
    peft_model = get_peft_model(_TinyBlock().cuda(), LoraConfig(r=4, lora_alpha=8, target_modules=["proj"]))
    # peft zero-inits lora_B; randomize it so the merge delta is nonzero.
    for module in peft_model.modules():
        if hasattr(module, "lora_B"):
            for adapter in module.lora_B:
                torch.nn.init.normal_(module.lora_B[adapter].weight)
    return peft_model


def _run_update(peft_model, buffer_size):
    updater = _CaptureUpdater(Namespace(update_weight_buffer_size=buffer_size), {"transformer": peft_model})
    updater.update_weights()
    return updater.buckets


def _run_lora_ipc_update(peft_model, buffer_size):
    updater = _CaptureLoRAIPCUpdater(Namespace(update_weight_buffer_size=buffer_size), {"transformer": peft_model})
    updater.update_weights()
    return updater.buckets


def _assert_buckets_have_complete_ab_pairs(buckets):
    for bucket in buckets:
        by_layer: dict[str, set[str]] = {}
        for name, _ in bucket:
            prefix = PeftLoRAKeyMapper.layer_prefix(name)
            ab = "A" if ".lora_A" in name else "B"
            by_layer.setdefault(prefix, set()).add(ab)
        for prefix, abs_ in by_layer.items():
            assert abs_ == {"A", "B"}, f"incomplete LoRA pair for {prefix} in bucket: {abs_}"


def test_lora_merge_and_name_mapping():
    peft_model = _make_peft_model()
    synced = {name: tensor for bucket in _run_update(peft_model, 1 << 30) for name, tensor in bucket}

    # PEFT wrappers stripped, lora_* params excluded — names match sglang-d's DiT.
    assert set(synced) == {"proj.weight", "norm.weight", "norm.bias"}

    lora_layer = peft_model.base_model.model.proj
    A, B = lora_layer.lora_A["default"].weight, lora_layer.lora_B["default"].weight
    expected = lora_layer.base_layer.weight + lora_layer.scaling["default"] * (B @ A)
    torch.testing.assert_close(synced["proj.weight"], expected)
    torch.testing.assert_close(synced["norm.weight"], peft_model.base_model.model.norm.weight)


def test_bucket_flush_respects_buffer_size():
    buckets = _run_update(_make_peft_model(), buffer_size=1)
    assert all(len(bucket) == 1 for bucket in buckets)
    assert sum(len(bucket) for bucket in buckets) == 3


def test_lora_ipc_bucket_keeps_ab_together():
    peft_model = _make_peft_model()
    lora_layer = peft_model.base_model.model.proj
    pair_size = (
        lora_layer.lora_A["default"].weight.numel() * lora_layer.lora_A["default"].weight.element_size()
        + lora_layer.lora_B["default"].weight.numel() * lora_layer.lora_B["default"].weight.element_size()
    )
    buckets = _run_lora_ipc_update(peft_model, buffer_size=pair_size)
    _assert_buckets_have_complete_ab_pairs(buckets)
    synced = {name: tensor for bucket in buckets for name, tensor in bucket}
    assert set(synced) == {"proj.lora_A", "proj.lora_B"}
    A, B = lora_layer.lora_A["default"].weight, lora_layer.lora_B["default"].weight
    torch.testing.assert_close(synced["proj.lora_A"], A)
    torch.testing.assert_close(synced["proj.lora_B"], B)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
