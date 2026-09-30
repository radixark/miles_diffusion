"""Export only ControlNet tensors from a Miles distributed checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .modeling import WanPoseControlTransformer


def export_controlnet(ckpt_dir, out, *, base_checkpoint, num_control_blocks=4):
    """Read selected DCP tensors without materializing frozen base weights.

    Architecture validation uses a meta-device backbone. The only resident
    parameter-sized allocations are the saved ControlNet tensors themselves.
    """
    import torch.distributed.checkpoint as dcp
    from accelerate import init_empty_weights
    from diffusers import WanTransformer3DModel
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    ckpt_dir, out = Path(ckpt_dir), Path(out)
    model_dir = ckpt_dir if (ckpt_dir / ".metadata").is_file() else ckpt_dir / "model"
    if not (model_dir / ".metadata").is_file():
        raise FileNotFoundError(f"no distributed checkpoint metadata under {model_dir}")
    reader = dcp.FileSystemReader(str(model_dir))
    metadata = reader.read_metadata()
    prefix = "model_state.model.controlnet."
    selected = {
        name: torch.empty(spec.size, dtype=spec.properties.dtype, device="cpu")
        for name, spec in metadata.state_dict_metadata.items()
        if name.startswith(prefix)
    }
    if not selected:
        raise ValueError("checkpoint has no single-transformer Wan ControlNet tensors")
    dcp.load(
        selected,
        storage_reader=reader,
        no_dist=True,
        planner=DefaultLoadPlanner(flatten_state_dict=False, allow_partial_load=True),
    )
    config = WanTransformer3DModel.load_config(base_checkpoint, subfolder="transformer")
    with init_empty_weights(include_buffers=False):
        backbone = WanTransformer3DModel.from_config(config)
        backbone.register_to_config(_name_or_path=str(base_checkpoint))
        model = WanPoseControlTransformer(backbone, num_control_blocks=num_control_blocks)
    branch_state = {name.removeprefix(prefix): value for name, value in selected.items()}
    model.save_controlnet(out, state_dict=branch_state)
    export_info = {
        "source_checkpoint": str(model_dir.resolve()),
        "base_checkpoint": str(base_checkpoint),
        "num_control_blocks": num_control_blocks,
        "tensor_count": len(branch_state),
        "parameter_count": sum(value.numel() for value in branch_state.values()),
    }
    (out / "export_info.json").write_text(json.dumps(export_info, indent=2) + "\n")
    return out
