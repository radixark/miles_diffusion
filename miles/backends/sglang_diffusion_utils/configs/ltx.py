"""LTX-2 sglang-d rollout engine config.

Mirrors ``fsdp_utils/configs/ltx.py`` on the train side: model detection,
weight-path resolution, and extra ``ServerArgs`` fields for LTX2Pipeline.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def is_ltx_model(args) -> bool:
    model_type = (getattr(args, "diffusion_model_type", "auto") or "auto").lower()
    if model_type == "ltx":
        return True
    if model_type != "auto":
        return False
    diff_model = (getattr(args, "diffusion_model", None) or "").lower()
    return "ltx" in diff_model or diff_model.endswith(".safetensors")


def resolve_ltx_transformer_weights_path(
    diffusion_model: str | None,
    *,
    explicit_path: str | None = None,
) -> str | None:
    """Return official safetensors path for sglang ``transformer_weights_path``.

    When miles train loads a single-file LTX checkpoint, sglang should load the
    same safetensors via ``transformer_weights_path`` instead of the HF
    materialized ``model.safetensors`` overlay (which can differ at ~1e-4 bf16).
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


def resolve_sglang_model_path(args) -> str:
    model_path = args.diffusion_model
    if is_ltx_model(args) and model_path.endswith(".safetensors"):
        return os.path.dirname(model_path)
    return model_path


def server_kwargs_extras(args) -> dict:
    """Extra ``ServerArgs`` kwargs; call only when ``is_ltx_model(args)``."""
    extras: dict = {"pipeline_class_name": "LTX2Pipeline"}
    if getattr(args, "sglang_pipeline_class_name", None):
        extras["pipeline_class_name"] = args.sglang_pipeline_class_name

    explicit = getattr(args, "sglang_transformer_weights_path", None)
    weights_path = resolve_ltx_transformer_weights_path(
        getattr(args, "diffusion_model", None),
        explicit_path=explicit,
    )
    if weights_path and not explicit:
        extras["transformer_weights_path"] = weights_path
        logger.info("LTX rollout: transformer_weights_path=%s", weights_path)
    elif explicit:
        extras["transformer_weights_path"] = explicit

    gemma_path = getattr(args, "ltx_gemma_path", None)
    if gemma_path:
        extras["component_paths"] = {"text_encoder": gemma_path}

    return extras
