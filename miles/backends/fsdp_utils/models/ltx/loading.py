"""LTX-2 checkpoint resolution, materialization, and component loading."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

TRAIN_COMPONENT = "transformer"


def _is_hf_model_id(ref: str | None) -> bool:
    if not ref:
        return False
    text = str(ref)
    if text.endswith(".safetensors") or os.path.exists(text):
        return False
    return "/" in text or "ltx" in text.lower()


def _diffusion_cache_root() -> Path:
    from sglang.multimodal_gen import envs

    return Path(envs.SGLANG_DIFFUSION_CACHE_ROOT)


def _find_cached_materialized_dir(hf_model_id: str) -> Path | None:
    materialized = _diffusion_cache_root() / "materialized_models"
    if not materialized.is_dir():
        return None

    prefix = hf_model_id.replace("/", "__") + "-"
    candidates = sorted(
        (d for d in materialized.iterdir() if d.is_dir() and d.name.startswith(prefix)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for directory in candidates:
        checkpoint = directory / TRAIN_COMPONENT / "model.safetensors"
        if checkpoint.is_file():
            return directory
    return None


def _transformer_checkpoint_in_dir(materialized_dir: Path) -> Path:
    checkpoint = materialized_dir / TRAIN_COMPONENT / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Materialized LTX model at {materialized_dir} is missing " f"{TRAIN_COMPONENT}/model.safetensors"
        )
    return checkpoint


def _materialized_config_path(checkpoint: Path) -> Path | None:
    """Return sibling ``config.json`` for sglang overlay materialized DiT weights."""
    config_json = checkpoint.parent / "config.json"
    return config_json if config_json.is_file() else None


def _is_materialized_diffusers_checkpoint(checkpoint: Path) -> bool:
    return _materialized_config_path(checkpoint) is not None


def _read_materialized_transformer_config(checkpoint: Path) -> dict:
    config_json = _materialized_config_path(checkpoint)
    if config_json is None:
        raise FileNotFoundError(f"Materialized LTX checkpoint {checkpoint} is missing sibling config.json")
    transformer_cfg = json.loads(config_json.read_text())
    return {TRAIN_COMPONENT: transformer_cfg}


def load_transformer_for_train(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    dtype: Any = None,
    materialize_weights: bool = True,
):
    """Load LTX DiT for FSDP train from materialized diffusers or comfy safetensors."""
    from ltx_core.loader.helpers import create_meta_model, load_state_dict
    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LTX checkpoint not found: {checkpoint}")

    torch_device = torch.device(device) if isinstance(device, str) else device
    if dtype is None:
        dtype = torch.bfloat16

    if _is_materialized_diffusers_checkpoint(checkpoint):
        config = _read_materialized_transformer_config(checkpoint)
        meta_model = create_meta_model(LTXModelConfigurator, config, ())
        if not materialize_weights:
            return meta_model.to(dtype=dtype)
        loader = SafetensorsModelStateDictLoader()
        sd = load_state_dict(
            str(checkpoint),
            loader,
            DummyRegistry(),
            torch.device("cpu"),
            None,
        )
        state_dict = sd.sd
        if dtype is not None:
            state_dict = {key: value.to(dtype=dtype) for key, value in state_dict.items()}
        meta_model.load_state_dict(state_dict, strict=False, assign=True)
        logger.info(
            "LTX train: loaded materialized diffusers transformer from %s",
            checkpoint,
        )
        return meta_model.to(torch_device)

    builder = SingleGPUModelBuilder(
        model_path=str(checkpoint),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    if not materialize_weights:
        return builder.meta_model(builder.model_config(), builder.module_ops).to(dtype=dtype)
    return builder.build(device=torch_device, dtype=dtype)


def ensure_materialized_model(hf_model_id: str) -> Path:
    """Materialize the overlay model via sglang (same pipeline as rollout)."""
    cached = _find_cached_materialized_dir(hf_model_id)
    if cached is not None:
        return cached

    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model

    logger.info(
        "LTX: materializing overlay model for %s (first run may download HF weights)",
        hf_model_id,
    )
    materialized = maybe_download_model(
        hf_model_id,
        download=True,
        force_diffusers_model=True,
    )
    materialized_dir = Path(materialized)
    _transformer_checkpoint_in_dir(materialized_dir)
    return materialized_dir


def resolve_materialized_model_dir(
    hf_model_id: str,
    *,
    materialize: bool = True,
) -> Path | None:
    cached = _find_cached_materialized_dir(hf_model_id)
    if cached is not None:
        return cached
    if not materialize:
        return None
    return ensure_materialized_model(hf_model_id)


def resolve_transformer_checkpoint(
    hf_checkpoint: str | None,
    *,
    materialize: bool = True,
) -> str:
    """Resolve the single-file DiT checkpoint used by FSDP train."""
    if hf_checkpoint:
        path = Path(str(hf_checkpoint)).expanduser()
        if path.is_file() and path.suffix == ".safetensors":
            return str(path)

        if _is_hf_model_id(str(hf_checkpoint)):
            materialized_dir = resolve_materialized_model_dir(
                str(hf_checkpoint),
                materialize=materialize,
            )
            if materialized_dir is not None:
                checkpoint = _transformer_checkpoint_in_dir(materialized_dir)
                logger.info(
                    "LTX train: using materialized transformer %s (from %s)",
                    checkpoint,
                    materialized_dir,
                )
                return str(checkpoint)

    raise FileNotFoundError(
        "Could not resolve LTX transformer checkpoint. Pass --hf-checkpoint "
        "Lightricks/LTX-2.3 (recommended) or a .safetensors override."
    )


def load_component(
    component: str,
    args,
    *,
    master_dtype: torch.dtype,
    materialize_weights: bool,
):
    if component != TRAIN_COMPONENT:
        raise ValueError(f"LTX trains the single DiT ({TRAIN_COMPONENT!r}); got {component!r}")
    checkpoint = resolve_transformer_checkpoint(str(args.hf_checkpoint))
    return load_transformer_for_train(
        checkpoint,
        device="cpu",
        dtype=master_dtype,
        materialize_weights=materialize_weights,
    )
