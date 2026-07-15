from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import json
from argparse import Namespace

import pytest
import torch

from miles.backends.fsdp_utils.actor import (
    apply_lora,
    materialize_sharded_model,
    peft_checkpoint_key_map,
    reset_lora_adapters,
)
from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend, ModelBackend


def _tiny_wan_checkpoint(tmp_path):
    """A minimal diffusers-layout checkpoint dir: model_index.json + a 1-block
    WanTransformer3DModel config + scheduler config. No weight files — enough
    for build_models_and_scheduler, which must not read any."""
    config = {
        "_class_name": "WanTransformer3DModel",
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "num_layers": 1,
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
    model_index = {
        "_class_name": "WanPipeline",
        "scheduler": ["diffusers", "UniPCMultistepScheduler"],
        "transformer": ["diffusers", "WanTransformer3DModel"],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
    }
    scheduler_config = {"_class_name": "UniPCMultistepScheduler", "num_train_timesteps": 1000}
    (tmp_path / "transformer").mkdir()
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "model_index.json").write_text(json.dumps(model_index))
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(config))
    (tmp_path / "scheduler" / "scheduler_config.json").write_text(json.dumps(scheduler_config))
    return tmp_path


class TestBuildModelsAndScheduler:
    # The core contract: params on meta (no weight I/O), buffers real (computed
    # by __init__) — Wan's non-persistent rope tables are the regression case.
    def test_params_meta_buffers_real(self, tmp_path):
        ckpt = _tiny_wan_checkpoint(tmp_path)
        args = Namespace(hf_checkpoint=str(ckpt), update_weight_target_modules=["transformer"])
        raw_models, scheduler = DiffusersModelBackend(None).build_models_and_scheduler(
            args, master_dtype=torch.float32
        )
        model = raw_models["transformer"]
        assert all(p.is_meta for p in model.parameters())
        buffers = dict(model.named_buffers())
        assert "rope.freqs_cos" in buffers and not buffers["rope.freqs_cos"].is_meta
        assert type(scheduler).__name__ == "UniPCMultistepScheduler"

    def test_missing_component_raises(self, tmp_path):
        ckpt = _tiny_wan_checkpoint(tmp_path)
        args = Namespace(hf_checkpoint=str(ckpt), update_weight_target_modules=["vae"])
        with pytest.raises(ValueError, match="has no component"):
            DiffusersModelBackend(None).build_models_and_scheduler(args, master_dtype=torch.float32)

    # transformers-library components (text_encoder) can't meta-init through
    # diffusers class resolution — must point at the legacy escape hatch.
    def test_non_diffusers_component_raises(self, tmp_path):
        ckpt = _tiny_wan_checkpoint(tmp_path)
        args = Namespace(hf_checkpoint=str(ckpt), update_weight_target_modules=["text_encoder"])
        with pytest.raises(ValueError, match="legacy"):
            DiffusersModelBackend(None).build_models_and_scheduler(args, master_dtype=torch.float32)

    def test_custom_backend_without_meta_support_points_to_legacy(self):
        class _LegacyOnly(ModelBackend):
            def load_models_and_scheduler(self, args, *, master_dtype):
                raise NotImplementedError

        with pytest.raises(NotImplementedError, match="fsdp-load-mode legacy"):
            _LegacyOnly(None).build_models_and_scheduler(None, master_dtype=torch.float32)


class TestMaterializeShardedModel:
    # to_empty wipes buffers; non-persistent ones aren't in any checkpoint, so
    # materialize must carry them over (Wan rope). Persistent ones may be
    # garbage afterwards — the stream load refills them.
    def test_non_persistent_buffers_survive(self):
        class _M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.register_buffer("freqs", torch.arange(4, dtype=torch.float32), persistent=False)

        from accelerate import init_empty_weights

        with init_empty_weights(include_buffers=False):
            m = _M()
        assert m.linear.weight.is_meta and not m.freqs.is_meta
        materialize_sharded_model(m, torch.device("cpu"))
        assert not m.linear.weight.is_meta
        torch.testing.assert_close(m.freqs, torch.arange(4, dtype=torch.float32))


class _TinyDit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(8, 8, bias=False)
        self.norm = torch.nn.LayerNorm(8)


def _lora_args(init_weight="gaussian"):
    return Namespace(
        lora_target_modules=["to_q"],
        diffusion_init_lora_weight=init_weight,
        lora_rank=4,
        lora_alpha=4,
    )


class TestLoraStreamPath:
    # Key map must invert the FQN wrapping so raw checkpoint keys land on the
    # peft-wrapped names — same string contract as diffusion_update_weight_utils.
    def test_key_map_inverts_peft_wrapping(self, monkeypatch):
        import torch.distributed as dist

        monkeypatch.setattr(dist, "get_rank", lambda: 1)
        with torch.device("meta"):
            model = _TinyDit()
        model = apply_lora(model, _lora_args(), None, on_meta=True)
        key_map = peft_checkpoint_key_map(model)
        assert key_map("to_q.weight") == "base_model.model.to_q.base_layer.weight"
        assert key_map("norm.weight") == "base_model.model.norm.weight"
        assert key_map("unknown.weight") == "unknown.weight"  # passthrough

    def test_adapters_created_on_meta_then_reset(self, monkeypatch):
        import torch.distributed as dist

        monkeypatch.setattr(dist, "get_rank", lambda: 1)
        with torch.device("meta"):
            model = _TinyDit()
        model = apply_lora(model, _lora_args(), None, on_meta=True)
        lora_layer = model.base_model.model.to_q
        assert lora_layer.lora_A["default"].weight.is_meta
        model.to_empty(device="cpu")
        reset_lora_adapters(model, "gaussian")
        assert torch.isfinite(lora_layer.lora_A["default"].weight).all()
        assert (lora_layer.lora_B["default"].weight == 0).all()

    # pissa/olora-style inits read the base weights at wrap time; on meta they
    # would silently produce garbage — must be an explicit error.
    def test_data_aware_init_rejected_on_meta(self):
        with torch.device("meta"):
            model = _TinyDit()
        with pytest.raises(ValueError, match="legacy"):
            apply_lora(model, _lora_args(init_weight="pissa"), None, on_meta=True)
