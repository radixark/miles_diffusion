"""The portable OpenPI bundle shared by model loading and weight export."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

META_FILE = "miles_pi05.json"


def read_bundle(path):
    root = Path(path)
    metadata = json.loads((root / META_FILE).read_text())
    if metadata.get("version") != 1 or metadata.get("config_name") != "pi05_libero":
        raise ValueError("Expected a version 1 pi05_libero bundle; see docs/models/pi05.md")
    if not metadata["model"].get("pi05"):
        raise ValueError("The bundle must describe pi05=True")
    stats = root / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    if not stats.is_file():
        raise FileNotFoundError(f"Missing training normalization statistics: {stats}")
    return metadata


def serving_config(model):
    # SGLang's native loader reads LeRobot-style config keys even for OpenPI weights.
    # Return all padded dimensions: downstream clients apply OpenPI output transforms.
    return {
        "model_type": "pi05",
        "paligemma_variant": model["paligemma_variant"],
        "action_expert_variant": model["action_expert_variant"],
        "chunk_size": model["action_horizon"],
        "n_action_steps": model["action_horizon"],
        "max_action_dim": model["action_dim"],
        "max_state_dim": model["action_dim"],
        "tokenizer_max_length": model["max_token_len"],
        "image_resolution": [224, 224],
        "input_features": {
            f"observation.images.{key}": {"type": "VISUAL", "shape": [3, 224, 224]}
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "output_features": {"action": {"type": "ACTION", "shape": [model["action_dim"]]}},
    }


def export_checkpoint(ckpt_dir, base_bundle, out):
    """Consolidate the actor's full-model DCP checkpoint without importing OpenPI or starting NCCL."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    from miles.backends.fsdp_utils.lora_export import flatten_state

    base_bundle, out = Path(base_bundle), Path(out)
    metadata = read_bundle(base_bundle)
    meta_path = Path(ckpt_dir) / "meta.json"
    metadata["training_checkpoint"] = {
        "path": str(Path(ckpt_dir).resolve()),
        **(json.loads(meta_path.read_text()) if meta_path.exists() else {}),
    }
    if out.exists():
        raise FileExistsError(f"Refusing to replace {out}")
    model_dir = Path(ckpt_dir) / "model"
    if not (model_dir / ".metadata").is_file():
        raise FileNotFoundError(f"Not a miles DCP model directory: {model_dir}")
    with tempfile.TemporaryDirectory() as tmp:
        consolidated = Path(tmp) / "model.pt"
        dcp_to_torch_save(str(model_dir), str(consolidated))
        tree = torch.load(consolidated, map_location="cpu", weights_only=True)
    prefix = "model_state.model."
    state = {}
    for name, tensor in flatten_state(tree):
        if not name.startswith(prefix) or "lora_" in name:
            raise ValueError(f"Expected a full OpenPI model checkpoint, got {name}")
        state[name.removeprefix(prefix)] = tensor.contiguous()
    # Check every base tensor and shape, including safetensors' tied-weight convention.
    # DCP may contain extra aliases for tied weights; export the exact base key set.
    with safe_open(str(base_bundle / "model.safetensors"), framework="pt") as base:
        expected = set(base.keys())
        tensor_metadata = base.metadata() or {}
        aliases = {key: value for key, value in tensor_metadata.items() if value in expected}
        missing = expected - state.keys()
        if missing:
            raise ValueError(f"DCP checkpoint is missing base keys: {sorted(missing)[:5]}")
        for name in expected:
            if tuple(state[name].shape) != tuple(base.get_slice(name).get_shape()):
                raise ValueError(f"Checkpoint shape mismatch: {name}")
        unexpected = state.keys() - expected - aliases.keys()
        if unexpected:
            raise ValueError(f"DCP checkpoint has unexpected keys: {sorted(unexpected)[:5]}")
        for alias, canonical in aliases.items():
            if alias in state and not torch.equal(state[alias], state[canonical]):
                raise ValueError(f"Tied checkpoint weights diverged: {alias} and {canonical}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".pi05-export-") as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        shutil.copytree(base_bundle / "assets", stage / "assets")
        (stage / META_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
        (stage / "config.json").write_text(json.dumps(serving_config(metadata["model"]), indent=2) + "\n")
        # Clone aliases independently: save_file rejects shared tensor storage.
        save_file(
            {key: state[key].clone() for key in sorted(expected)},
            str(stage / "model.safetensors"),
            metadata=tensor_metadata,
        )
        stage.rename(out)
    return out
