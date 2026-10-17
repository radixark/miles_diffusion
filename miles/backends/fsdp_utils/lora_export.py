"""Convert a PEFT LoRA DCP checkpoint into one loader-ready safetensors file.

Renames ``model_state.model.base_model.model.<m>.lora_{A,B}.default.weight`` (the
PEFT DCP layout every miles LoRA run saves) to ``<component>.<m>.lora_{A,B}.weight``,
plus an ``adapter_config.json`` sidecar so loaders pick up lora_alpha. ``<component>``
is the pipeline module the run trained (``transformer``, or ``transformer_ref`` for
MiniMax H3 Ref2VA), recorded in the checkpoint's ``meta.json``. Multi-DiT families
are not handled yet and fail loudly.
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


def convert_lora_state(flat, component: str = "transformer") -> dict:
    """PEFT DCP names -> diffusers/sglang loader names under ``component``."""
    out = {}
    for name, tensor in flatten_state(flat):
        if "lora_" not in name:
            continue
        if not name.startswith(DCP_PREFIX):
            raise ValueError(f"unexpected LoRA key layout: {name}")
        out[f"{component}." + name.removeprefix(DCP_PREFIX).replace(".default.", ".")] = tensor.contiguous()
    if not out:
        raise ValueError("checkpoint holds no LoRA tensors")
    return out


def checkpoint_component(ckpt_dir: Path, component: str | None = None) -> str:
    """The pipeline module whose LoRA ``ckpt_dir`` holds.

    A single-module run saves the bare PEFT model, so the DCP keys do not name the module;
    ``meta.json`` does. Checkpoints written before it recorded the module trained ``transformer``.
    """
    meta_path = Path(ckpt_dir) / "meta.json"
    saved = json.loads(meta_path.read_text()).get("update_weight_target_modules") if meta_path.exists() else None
    if saved is None:
        return component or "transformer"
    if len(saved) != 1:
        raise ValueError(f"LoRA export supports one trained module, the checkpoint has {saved}")
    if component is not None and component != saved[0]:
        raise ValueError(f"--component {component} does not match the checkpoint's trained module {saved[0]}")
    return saved[0]


def export_lora_safetensors(
    ckpt_dir: Path, out: Path, lora_rank: int, lora_alpha: int, component: str | None = None
) -> Path:
    """Gather the DCP model dir under ``ckpt_dir`` and write ``out`` plus its sidecar."""
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    model_dir = Path(ckpt_dir) / "model"
    if not model_dir.is_dir():
        raise ValueError(f"{model_dir} is not a checkpoint model directory")
    component = checkpoint_component(ckpt_dir, component)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        dcp_to_torch_save(str(model_dir), tmp.name)
        flat = torch.load(tmp.name, map_location="cpu", weights_only=False)

    out_state = convert_lora_state(flat, component)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_state, str(out), metadata={"lora_rank": str(lora_rank), "lora_alpha": str(lora_alpha)})
    sidecar = out.parent / "adapter_config.json"
    sidecar.write_text(json.dumps({"lora_alpha": lora_alpha, "r": lora_rank}, indent=1) + "\n")
    logger.info("exported %d %s LoRA tensors -> %s (+ %s)", len(out_state), component, out, sidecar.name)
    return out
