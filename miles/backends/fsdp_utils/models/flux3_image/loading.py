"""FLUX.3 image checkpoint loading: the DiT of a package (config.json + model.safetensors), local or on the Hub."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .modeling import Flux3ImageTransformer

TRAIN_COMPONENT = "transformer"


def package_dir(checkpoint_path: str, *, with_weights: bool) -> Path:
    """A local package directory, or the package files of a Hub repo id (fetched into the HF cache)."""
    if os.path.isdir(checkpoint_path):
        return Path(checkpoint_path)
    from huggingface_hub import snapshot_download

    patterns = ["config.json", "model.safetensors"] if with_weights else ["config.json"]
    return Path(snapshot_download(checkpoint_path, allow_patterns=patterns))


def build_transformer(checkpoint_path: str) -> Flux3ImageTransformer:
    dit = json.loads((package_dir(checkpoint_path, with_weights=False) / "config.json").read_text())["dit"]
    return Flux3ImageTransformer(
        hidden_size=dit["hidden"],
        num_heads=dit["heads"],
        depth=dit["depth"],
        depth_single_blocks=dit["depth_single"],
        mlp_ratio=dit["mlp_ratio"],
        axes_dim=tuple(dit["axes_dim"]),
        theta=dit["theta"],
        image_channels=dit["in_channels"]["image"],
        context_in_dim=dit["in_channels"]["text"],
    )


def load_component(
    component: str,
    *,
    checkpoint_path: str,
    master_dtype: torch.dtype,
    materialize_weights: bool,
) -> Flux3ImageTransformer:
    if component != TRAIN_COMPONENT:
        raise ValueError(f"FLUX.3 image trains only {TRAIN_COMPONENT!r}, got {component!r}")
    # Every parameter comes from the checkpoint, so skip the random init even on rank 0.
    with torch.device("meta"):
        model = build_transformer(checkpoint_path)
    if materialize_weights:
        from safetensors.torch import load_file

        checkpoint = load_file(str(package_dir(checkpoint_path, with_weights=True) / "model.safetensors"))
        # The checkpoint also carries weights this text-to-image module tree does not build.
        own = model.state_dict()
        model.load_state_dict({name: checkpoint[name] for name in own}, strict=True, assign=True)
    return model.to(master_dtype)
