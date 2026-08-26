"""Convert a PEFT LoRA DCP checkpoint into one loader-ready safetensors file.

Renames ``model_state.model.base_model.model.<m>.lora_{A,B}.default.weight`` (the
PEFT DCP layout every miles LoRA run saves) to ``transformer.<m>.lora_{A,B}.weight``,
plus an ``adapter_config.json`` sidecar so loaders pick up lora_alpha. Multi-DiT
families are not handled yet and fail the layout check loudly.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

DCP_PREFIX = "model_state.model.base_model.model."


def flatten_state(tree, prefix=""):
    if isinstance(tree, torch.Tensor):
        yield prefix, tree
        return
    for key, value in tree.items():
        yield from flatten_state(value, f"{prefix}.{key}" if prefix else key)


def convert_lora_state(flat) -> dict:
    """PEFT DCP names -> diffusers/sglang loader names."""
    out = {}
    for name, tensor in flatten_state(flat):
        if "lora_" not in name:
            continue
        if not name.startswith(DCP_PREFIX):
            raise ValueError(f"unexpected LoRA key layout: {name}")
        out["transformer." + name.removeprefix(DCP_PREFIX).replace(".default.", ".")] = tensor.contiguous()
    if not out:
        raise ValueError("checkpoint holds no LoRA tensors")
    return out


def export_lora_safetensors(ckpt_dir: Path, out: Path, lora_rank: int, lora_alpha: int) -> Path:
    """Gather the DCP model dir under ``ckpt_dir`` and write ``out`` plus its sidecar."""
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    model_dir = Path(ckpt_dir) / "model"
    if not model_dir.is_dir():
        raise ValueError(f"{model_dir} is not a checkpoint model directory")

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        dcp_to_torch_save(str(model_dir), tmp.name)
        flat = torch.load(tmp.name, map_location="cpu", weights_only=False)

    out_state = convert_lora_state(flat)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_state, str(out), metadata={"lora_rank": str(lora_rank), "lora_alpha": str(lora_alpha)})
    sidecar = out.parent / "adapter_config.json"
    sidecar.write_text(json.dumps({"lora_alpha": lora_alpha, "r": lora_rank}, indent=1) + "\n")
    logger.info("exported %d LoRA tensors -> %s (+ %s)", len(out_state), out, sidecar.name)
    return out
