"""CPU tests of the new branch's isolation, learning, and checkpoint contract."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import copy
import json

import pytest
import torch
from diffusers import WanTransformer3DModel

from miles.backends.fsdp_utils.models.wan_controlnet.modeling import WanPoseControlTransformer


@pytest.fixture
def case():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    base = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=12,
        freq_dim=8,
        ffn_dim=32,
        num_layers=3,
        rope_max_seq_len=32,
    ).eval()
    model = WanPoseControlTransformer(base, num_control_blocks=2)
    inputs = {
        "hidden_states": torch.randn(1, 4, 2, 4, 4),
        "timestep": torch.tensor([350.0]),
        "encoder_hidden_states": torch.randn(1, 5, 12),
    }
    pose = torch.randn_like(inputs["hidden_states"])
    yield base, model, inputs, pose
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("per_token_timesteps", [False, True])
def test_zero_init_is_exactly_base(case, per_token_timesteps):
    base, model, inputs, pose = case
    if per_token_timesteps:
        inputs["timestep"] = torch.full((1, 8), 350.0)
        inputs["timestep"][:, :4] = 0
    with torch.no_grad():
        expected = base(**inputs).sample
        actual = model(**inputs, control_latents=pose).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert all(parameter.requires_grad for parameter in model.controlnet.parameters())


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_branch_learns_and_base_stays_frozen(case, gradient_checkpointing):
    base, model, inputs, pose = case
    model.train()
    assert not base.training
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
    original = {name: tensor.clone() for name, tensor in base.state_dict().items()}
    optimizer = torch.optim.AdamW(model.controlnet.parameters(), lr=0.01)
    target = torch.randn_like(inputs["hidden_states"])
    loss = (model(**inputs, control_latents=pose).sample - target).square().mean()
    loss.backward()
    assert all(parameter.grad is None for parameter in base.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.controlnet.output_projections.parameters()
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # First step opens the zero gate; next step must reach copied blocks and pose encoder.
    loss = (model(**inputs, control_latents=pose).sample - target).square().mean()
    loss.backward()
    assert model.controlnet.patch_embedding.weight.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.controlnet.blocks.parameters()
    )
    with torch.no_grad():
        output = model(**inputs, control_latents=pose).sample
        changed_pose = model(**inputs, control_latents=-pose).sample
        disabled = model(**inputs, control_latents=pose, control_scale=0).sample
        vanilla = base(**inputs).sample
    assert not torch.equal(output, changed_pose)
    torch.testing.assert_close(disabled, vanilla, rtol=0, atol=0)
    for name, tensor in base.state_dict().items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


def test_control_checkpoint_roundtrip_excludes_base(case, tmp_path):
    from safetensors.torch import load_file

    base, model, inputs, pose = case
    with torch.no_grad():
        for projection in model.controlnet.output_projections:
            projection.weight.normal_(std=0.02)
    model.save_controlnet(tmp_path, state_dict=model.state_dict())
    state = load_file(str(tmp_path / "controlnet.safetensors"))
    assert state and all(not key.startswith("backbone.") for key in state)
    metadata = json.loads((tmp_path / "controlnet_config.json").read_text())
    assert metadata["block_indices"] == [0, 1]
    restored = WanPoseControlTransformer.from_controlnet(copy.deepcopy(base), tmp_path)
    with torch.no_grad():
        expected = model(**inputs, control_latents=pose).sample
        actual = restored(**inputs, control_latents=pose).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rejects_unaligned_control(case):
    _, model, inputs, pose = case
    with pytest.raises(ValueError, match="align exactly"):
        model(**inputs, control_latents=pose[:, :, :1])


def test_loading_honors_meta_and_freezes_base(tmp_path, monkeypatch):
    from accelerate import init_empty_weights
    from miles.backends.fsdp_utils.models.wan_controlnet.loading import load_component

    base = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=12,
        freq_dim=8,
        ffn_dim=32,
        num_layers=3,
        rope_max_seq_len=32,
    )
    base.save_pretrained(tmp_path / "transformer")
    # Keep this loader materialization test tiny; production architecture checks
    # are exercised independently below without loading 5B weights.
    monkeypatch.setattr(
        "miles.backends.fsdp_utils.models.wan_controlnet.loading.validate_base_checkpoint",
        lambda path: WanTransformer3DModel.load_config(path, subfolder="transformer"),
    )
    materialized = load_component(
        "transformer",
        checkpoint_path=str(tmp_path),
        master_dtype=torch.float32,
        materialize_weights=True,
        num_control_blocks=2,
    )
    with init_empty_weights(include_buffers=False):
        empty = load_component(
            "transformer",
            checkpoint_path=str(tmp_path),
            master_dtype=torch.float32,
            materialize_weights=False,
            num_control_blocks=2,
        )
    assert all(parameter.is_meta for parameter in empty.parameters())
    assert empty.state_dict().keys() == materialized.state_dict().keys()
    assert all(not parameter.requires_grad for parameter in empty.backbone.parameters())
    for name, value in base.state_dict().items():
        torch.testing.assert_close(materialized.backbone.state_dict()[name], value, rtol=0, atol=0)


def test_inference_condition_restores_after_exception(case):
    _, model, inputs, pose = case
    with pytest.raises(RuntimeError, match="intentional"):
        with model.control_condition(pose, control_scale=0.5):
            assert model._inference_control[0] is pose
            with torch.no_grad():
                model(**inputs)
            raise RuntimeError("intentional")
    assert model._inference_control is None


def test_export_reads_control_tensors_and_reloads(case, tmp_path):
    import torch.distributed.checkpoint as dcp
    from miles.backends.fsdp_utils.models.wan_controlnet.export import export_controlnet

    base, model, inputs, pose = case
    base_path = tmp_path / "base"
    base.save_pretrained(base_path / "transformer")
    # Actual training loader records this path on the base config.
    base.register_to_config(_name_or_path=str(base_path))
    with torch.no_grad():
        model.controlnet.output_projections[0].weight.normal_(std=0.01)
    checkpoint_dir = tmp_path / "iter_0000001"
    dcp.save({"model_state": {"model": model.state_dict()}}, checkpoint_id=checkpoint_dir / "model")
    export_controlnet(checkpoint_dir, tmp_path / "export", base_checkpoint=str(base_path), num_control_blocks=2)
    restored = WanPoseControlTransformer.from_controlnet(copy.deepcopy(base), tmp_path / "export")
    with torch.no_grad():
        expected = model(**inputs, control_latents=pose).sample
        actual = restored(**inputs, control_latents=pose).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    metadata = json.loads((tmp_path / "export" / "export_info.json").read_text())
    assert metadata["parameter_count"] == sum(parameter.numel() for parameter in model.controlnet.parameters())


def test_bf16_autocast_keeps_zero_init_and_trainable_gradients(case):
    base, model, inputs, pose = case
    inputs["hidden_states"] = inputs["hidden_states"].to(torch.bfloat16)
    inputs["encoder_hidden_states"] = inputs["encoder_hidden_states"].to(torch.bfloat16)
    pose = pose.to(torch.bfloat16)
    model.enable_gradient_checkpointing()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = base(**inputs).sample
        actual = model(**inputs, control_latents=pose).sample
        loss = actual.float().square().mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    loss.backward()
    assert all(parameter.grad is None for parameter in base.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.controlnet.parameters()
        if parameter.grad is not None
    )
    assert model.controlnet.output_projections[0].weight.grad.abs().sum() > 0


def test_fsdp_precision_preserves_wan_fp32_parameters(case):
    from miles.backends.fsdp_utils.mixed_precision import compile_param_dtype_maps
    from miles.backends.fsdp_utils.models.wan_controlnet.parallel_plan import FSDP_PARALLEL_PLAN

    _, model, _, _ = case
    blocks = [module for module in model.modules() if type(module).__name__ == "WanTransformerBlock"]
    policy = compile_param_dtype_maps(model, blocks, FSDP_PARALLEL_PLAN.param_dtype_patterns, torch.bfloat16)
    assert all(mapping["scale_shift_table"] == torch.float32 for mapping in policy.wrap_maps)
    assert all(mapping["norm2.weight"] == torch.float32 for mapping in policy.wrap_maps)
    assert policy.root_map["backbone.scale_shift_table"] == torch.float32
    assert policy.root_map["backbone.condition_embedder.time_embedder.linear_1.weight"] == torch.float32


def test_checkpoint_base_path_is_provenance_not_architecture(case, tmp_path):
    base, model, _, _ = case
    model.save_controlnet(tmp_path)
    reloaded_base = copy.deepcopy(base)
    reloaded_base.register_to_config(_name_or_path="/moved/snapshot/transformer")
    restored = WanPoseControlTransformer.from_controlnet(reloaded_base, tmp_path)
    assert restored.controlnet.block_indices == model.controlnet.block_indices
    config_path = tmp_path / "controlnet_config.json"
    metadata = json.loads(config_path.read_text())
    metadata["backbone_config"]["text_dim"] += 1
    config_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="architecture"):
        restored.load_controlnet(tmp_path)


@pytest.mark.parametrize("variant", ["dual_expert", "wrong_channels", "missing_ti2v_flag", "supported"])
def test_loader_rejects_unsupported_base_before_weights(tmp_path, variant):
    from miles.backends.fsdp_utils.models.wan_controlnet.loading import validate_base_checkpoint

    pipeline = {"_class_name": "WanPipeline", "transformer_2": [None, None], "expand_timesteps": True}
    config = dict(
        in_channels=48,
        out_channels=48,
        num_layers=30,
        num_attention_heads=24,
        attention_head_dim=128,
        text_dim=4096,
        ffn_dim=14336,
        image_dim=None,
        added_kv_proj_dim=None,
        patch_size=[1, 2, 2],
    )
    if variant == "dual_expert":
        pipeline["transformer_2"] = ["diffusers", "WanTransformer3DModel"]
    elif variant == "wrong_channels":
        config["in_channels"] = 16
    elif variant == "missing_ti2v_flag":
        pipeline["expand_timesteps"] = False
    (tmp_path / "model_index.json").write_text(json.dumps(pipeline))
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(config))
    if variant == "supported":
        assert validate_base_checkpoint(str(tmp_path))["in_channels"] == 48
    else:
        with pytest.raises(ValueError, match="dual-expert|unsupported Wan|WanPipeline configuration"):
            validate_base_checkpoint(str(tmp_path))


def test_checkpoint_checks_same_shape_semantic_configuration(case, tmp_path):
    base, model, _, _ = case
    model.save_controlnet(tmp_path)
    incompatible = copy.deepcopy(base)
    incompatible.register_to_config(eps=0.1)
    with pytest.raises(ValueError, match="full backbone configuration"):
        WanPoseControlTransformer.from_controlnet(incompatible, tmp_path)


def test_legacy_checkpoint_load_explicitly_warns(case, tmp_path):
    _, model, _, _ = case
    model.save_controlnet(tmp_path)
    config_path = tmp_path / "controlnet_config.json"
    config = json.loads(config_path.read_text())
    del config["backbone_config_full"]
    config_path.write_text(json.dumps(config))
    with pytest.warns(RuntimeWarning, match="lacks full backbone configuration"):
        model.load_controlnet(tmp_path)
