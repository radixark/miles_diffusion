"""LTX-2 sglang-d rollout engine config.

Rollout engine uses ``model_path=Lightricks/LTX-2.3`` + ``model_id=LTX-2.3`` so
sglang's overlay wrapper materializes a full diffusers tree (``model_index.json``,
VAE, text encoder, connectors). Train FSDP still loads ``--diffusion-model`` as a
single official safetensors file; ``transformer_weights_path`` pins rollout DiT
to that same file for weight parity.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# sglang registry + overlay wrapper (see model_overlay.py).
LTX_DEFAULT_HF_MODEL = "Lightricks/LTX-2.3"
LTX_DEFAULT_MODEL_ID = "LTX-2.3"


def is_ltx_model(args) -> bool:
    model_type = (getattr(args, "diffusion_model_type", "auto") or "auto").lower()
    if model_type == "ltx":
        return True
    if model_type != "auto":
        return False
    diff_model = (getattr(args, "diffusion_model", None) or "").lower()
    return "ltx" in diff_model or diff_model.endswith(".safetensors")


def resolve_ltx_model_id(args) -> str:
    """Short registry id for ``ServerArgs.model_id`` (matches ``Lightricks/LTX-2.3``)."""
    if getattr(args, "sglang_model_id", None):
        return str(args.sglang_model_id)
    env_id = os.environ.get("MILES_LTX_MODEL_ID")
    if env_id:
        return env_id
    return LTX_DEFAULT_MODEL_ID


def resolve_sglang_model_path(args) -> str:
    """HF hub id for sglang pipeline skeleton (overlay materializes components)."""
    if getattr(args, "sglang_model_path", None):
        return str(args.sglang_model_path)
    env_path = os.environ.get("MILES_LTX_ROLLOUT_MODEL_PATH")
    if env_path:
        return env_path
    return LTX_DEFAULT_HF_MODEL


def resolve_ltx_transformer_weights_path(
    diffusion_model: str | None,
    *,
    explicit_path: str | None = None,
) -> str | None:
    """Return official single-file safetensors for sglang ``transformer_weights_path``.

    Overlay materialized ``transformer/model.safetensors`` is a different checkpoint
    variant than dev 22B; miles train loads ``--diffusion-model`` via ltx_core. Point
    rollout DiT init + weight sync at the same single-file ckpt as train.
    """
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.is_file():
            return str(path)
        return None

    env_path = os.environ.get("MILES_LTX_TRANSFORMER_WEIGHTS_PATH")
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_file():
            return str(path)

    if diffusion_model and str(diffusion_model).endswith(".safetensors"):
        path = Path(diffusion_model).expanduser()
        if path.is_file():
            return str(path)
    return None


def server_kwargs_extras(args) -> dict:
    """Extra ``ServerArgs`` kwargs; call only when ``is_ltx_model(args)``."""
    extras: dict = {
        "model_id": resolve_ltx_model_id(args),
    }

    explicit = getattr(args, "sglang_transformer_weights_path", None)
    weights_path = resolve_ltx_transformer_weights_path(
        getattr(args, "diffusion_model", None),
        explicit_path=explicit,
    )
    if weights_path and not explicit:
        extras["transformer_weights_path"] = weights_path
        logger.info(
            "LTX rollout: model_path=%s model_id=%s transformer_weights_path=%s",
            resolve_sglang_model_path(args),
            extras["model_id"],
            weights_path,
        )
    elif explicit:
        extras["transformer_weights_path"] = explicit
        logger.info(
            "LTX rollout: model_path=%s model_id=%s transformer_weights_path=%s",
            resolve_sglang_model_path(args),
            extras["model_id"],
            explicit,
        )
    else:
        logger.warning(
            "LTX rollout: no transformer_weights_path resolved from --diffusion-model; "
            "rollout DiT will use overlay default (may diverge from train ckpt)."
        )

    gemma_path = getattr(args, "ltx_gemma_path", None)
    if gemma_path:
        extras["component_paths"] = {"text_encoder": gemma_path}

    return extras
