from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=180,
    suite="stage-b-3-gpu-h200",
    labels=[],
)

import json
import os
from argparse import Namespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = 2

_TINY_WAN_CONFIG = {
    "_class_name": "WanTransformer3DModel",
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "num_layers": 2,
    "in_channels": 4,
    "out_channels": 4,
    "text_dim": 16,
    "freq_dim": 16,
    "ffn_dim": 32,
    "patch_size": [1, 2, 2],
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-6,
}


def _make_checkpoint(root: str) -> None:
    """Tiny diffusers-layout Wan checkpoint with a SHARDED transformer, so the
    stream loader's multi-file path is exercised."""
    from diffusers import WanTransformer3DModel

    torch.manual_seed(7)
    model = WanTransformer3DModel.from_config(_TINY_WAN_CONFIG)
    model.save_pretrained(os.path.join(root, "transformer"), max_shard_size="20KB")
    index = os.path.join(root, "transformer", "diffusion_pytorch_model.safetensors.index.json")
    assert os.path.exists(index), "tiny checkpoint must be sharded for this test"
    with open(os.path.join(root, "model_index.json"), "w") as f:
        json.dump(
            {
                "_class_name": "WanPipeline",
                "scheduler": ["diffusers", "UniPCMultistepScheduler"],
                "transformer": ["diffusers", "WanTransformer3DModel"],
            },
            f,
        )
    os.makedirs(os.path.join(root, "scheduler"), exist_ok=True)
    with open(os.path.join(root, "scheduler", "scheduler_config.json"), "w") as f:
        json.dump({"_class_name": "UniPCMultistepScheduler", "num_train_timesteps": 1000}, f)


def _fsdp_args(**overrides) -> Namespace:
    base = dict(
        hf_checkpoint=None,
        update_weight_target_modules=["transformer"],
        diffusion_forward_dtype="bf16",
        fsdp_reduce_dtype="fp32",
        gradient_checkpointing=False,
        lora_target_modules=["to_q", "to_k", "to_v"],
        diffusion_init_lora_weight="gaussian",
        lora_rank=4,
        lora_alpha=4,
    )
    base.update(overrides)
    return Namespace(**base)


def _stream_load(ckpt: str, use_lora: bool):
    from miles.backends.fsdp_utils.actor import (
        apply_fsdp2,
        apply_lora,
        materialize_sharded_model,
        peft_checkpoint_key_map,
        reset_lora_adapters,
    )
    from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend

    args = _fsdp_args(hf_checkpoint=ckpt)
    backend = DiffusersModelBackend(None)
    raw_models, _ = backend.build_models_and_scheduler(args, master_dtype=torch.float32)
    model = raw_models["transformer"]
    key_map = None
    if use_lora:
        model = apply_lora(model, args, None, on_meta=True)
        key_map = peft_checkpoint_key_map(model)
    model = apply_fsdp2(model, args=args)
    materialize_sharded_model(model, torch.cuda.current_device())
    backend.stream_load_weights(model, "transformer", args, master_dtype=torch.float32, key_map=key_map)
    if use_lora:
        reset_lora_adapters(model, args.diffusion_init_lora_weight)
    return model


def _worker(rank: int, ckpt: str, use_lora: bool):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29531"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=WORLD_SIZE)
    try:
        from diffusers import WanTransformer3DModel

        model = _stream_load(ckpt, use_lora)

        # reference: plain full from_pretrained — what the legacy path
        # materializes per component before sharding
        ref = WanTransformer3DModel.from_pretrained(ckpt, subfolder="transformer", torch_dtype=torch.float32)
        ref_sd = ref.state_dict()

        checked = 0
        for name, param in model.state_dict().items():
            if "lora_" in name:
                continue
            ref_key = name.replace(".base_layer", "").removeprefix("base_model.model.")
            full = param.full_tensor() if hasattr(param, "full_tensor") else param
            assert torch.equal(full.cpu(), ref_sd[ref_key]), f"mismatch: {name}"
            checked += 1
        assert checked == len(ref_sd), f"covered {checked}/{len(ref_sd)} reference tensors"

        # non-persistent rope tables are not in any state_dict: they must
        # survive to_empty via the snapshot in materialize_sharded_model
        buffers = dict(model.named_buffers())
        ref_buffers = dict(ref.named_buffers())
        for name, ref_buffer in ref_buffers.items():
            live = next(b for n, b in buffers.items() if n.endswith(name))
            assert torch.equal(live.cpu(), ref_buffer), f"buffer mismatch: {name}"

        if use_lora:
            from peft.tuners.lora import LoraLayer

            lora_layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
            assert lora_layers, "no LoRA layers injected"
            for layer in lora_layers:
                a = layer.lora_A["default"].weight
                b = layer.lora_B["default"].weight
                a = a.full_tensor() if hasattr(a, "full_tensor") else a
                b = b.full_tensor() if hasattr(b, "full_tensor") else b
                assert torch.isfinite(a).all() and a.abs().sum() > 0, "lora_A not initialized"
                assert (b == 0).all(), "lora_B must start at zero"
    finally:
        dist.barrier()
        dist.destroy_process_group()


@pytest.mark.parametrize("use_lora", [False, True], ids=["base", "lora"])
def test_stream_load_matches_from_pretrained(tmp_path, use_lora):
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"needs {WORLD_SIZE} GPUs")
    ckpt = str(tmp_path / "ckpt")
    os.makedirs(ckpt)
    _make_checkpoint(ckpt)
    mp.spawn(_worker, args=(ckpt, use_lora), nprocs=WORLD_SIZE, join=True)
