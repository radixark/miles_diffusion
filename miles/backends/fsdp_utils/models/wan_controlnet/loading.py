"""MilesModelBackend component loader for the new Wan pose-control branch."""

from diffusers import DiffusionPipeline, WanTransformer3DModel

from .modeling import WanPoseControlTransformer


def validate_base_checkpoint(checkpoint_path):
    """Reject unsupported experts/architectures before any large weight load."""
    pipeline = DiffusionPipeline.load_config(checkpoint_path)
    second = pipeline.get("transformer_2")
    has_second = second is not None and second != [None, None] and second != (None, None)
    if has_second or pipeline.get("boundary_ratio") is not None:
        raise ValueError("wan_controlnet does not support dual-expert Wan checkpoints")
    if pipeline.get("_class_name") != "WanPipeline" or pipeline.get("expand_timesteps") is not True:
        raise ValueError("wan_controlnet requires the Wan2.2-TI2V-5B WanPipeline configuration")
    config = WanTransformer3DModel.load_config(checkpoint_path, subfolder="transformer")
    expected = {
        "in_channels": 48,
        "out_channels": 48,
        "num_layers": 30,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "text_dim": 4096,
        "ffn_dim": 14336,
        "image_dim": None,
        "added_kv_proj_dim": None,
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if list(config.get("patch_size", ())) != [1, 2, 2]:
        mismatch["patch_size"] = (config.get("patch_size"), [1, 2, 2])
    if mismatch:
        raise ValueError(f"unsupported Wan base architecture; expected Wan2.2-TI2V-5B: {mismatch}")
    return config


def load_component(
    component, *, checkpoint_path, master_dtype, materialize_weights, num_control_blocks=4, controlnet_checkpoint=None
):
    if component != "transformer":
        raise ValueError("Wan2.2-TI2V-5B pose control has one transformer, not two experts")
    config = validate_base_checkpoint(checkpoint_path)
    if materialize_weights:
        backbone = WanTransformer3DModel.from_pretrained(
            checkpoint_path,
            subfolder="transformer",
            torch_dtype=master_dtype,
            low_cpu_mem_usage=True,
        )
    else:
        # The caller's init_empty_weights context preserves all parameters on meta.
        backbone = WanTransformer3DModel.from_config(config).to(dtype=master_dtype)
    model = WanPoseControlTransformer(
        backbone,
        num_control_blocks=num_control_blocks,
    )
    checkpoint = controlnet_checkpoint
    if checkpoint and materialize_weights:
        model.load_controlnet(checkpoint)
    return model
