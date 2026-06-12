"""LTX-2 model family: checkpoint resolution, rollout engine, and sampling hooks."""

from __future__ import annotations

import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LTX_DEFAULT_HF_MODEL = "Lightricks/LTX-2.3"
LTX_DEFAULT_MODEL_ID = "LTX-2.3"


def is_ltx_model(args) -> bool:
    model_type = (getattr(args, "diffusion_model_type", "auto") or "auto").lower()
    if model_type == "ltx":
        return True
    if model_type != "auto":
        return False
    return _looks_like_ltx_ref(getattr(args, "diffusion_model", None))


def _looks_like_ltx_ref(diffusion_model: str | None) -> bool:
    if not diffusion_model:
        return False
    ref = str(diffusion_model).lower()
    return "ltx" in ref or ref.endswith(".safetensors")


def _is_hf_model_id(ref: str | None) -> bool:
    if not ref:
        return False
    text = str(ref)
    if text.endswith(".safetensors") or os.path.exists(text):
        return False
    return "/" in text or "ltx" in text.lower()


def resolve_hf_model_id(args) -> str:
    """HF hub id used for sglang ``model_path`` / overlay materialization."""
    diffusion_model = getattr(args, "diffusion_model", None)
    if _is_hf_model_id(diffusion_model):
        return str(diffusion_model)
    if getattr(args, "sglang_model_path", None):
        return str(args.sglang_model_path)
    env_path = os.environ.get("MILES_LTX_ROLLOUT_MODEL_PATH")
    if env_path:
        return env_path
    return LTX_DEFAULT_HF_MODEL


def resolve_model_id(args) -> str:
    """Short registry id for sglang ``ServerArgs.model_id``."""
    if getattr(args, "sglang_model_id", None):
        return str(args.sglang_model_id)
    env_id = os.environ.get("MILES_LTX_MODEL_ID")
    if env_id:
        return env_id
    return LTX_DEFAULT_MODEL_ID


def _diffusion_cache_root() -> Path:
    return Path(
        os.environ.get("SGLANG_DIFFUSION_CACHE_ROOT", "/data/wenhao/sgl_diffusion_cache")
    )


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
        checkpoint = directory / "transformer" / "model.safetensors"
        if checkpoint.is_file():
            return directory
    return None


def _transformer_checkpoint_in_dir(materialized_dir: Path) -> Path:
    checkpoint = materialized_dir / "transformer" / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Materialized LTX model at {materialized_dir} is missing "
            f"transformer/model.safetensors"
        )
    return checkpoint


def _materialized_config_path(checkpoint: Path) -> Path | None:
    """Return sibling ``config.json`` for sglang overlay materialized DiT weights."""
    config_json = checkpoint.parent / "config.json"
    return config_json if config_json.is_file() else None


def _is_materialized_diffusers_checkpoint(checkpoint: Path) -> bool:
    return _materialized_config_path(checkpoint) is not None


def _read_materialized_transformer_config(checkpoint: Path) -> dict:
    import json

    config_json = _materialized_config_path(checkpoint)
    if config_json is None:
        raise FileNotFoundError(
            f"Materialized LTX checkpoint {checkpoint} is missing sibling config.json"
        )
    transformer_cfg = json.loads(config_json.read_text())
    return {"transformer": transformer_cfg}


def load_ltx_transformer_for_train(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    dtype: Any = None,
):
    """Load LTX DiT for FSDP train from materialized diffusers or comfy safetensors.

    Materialized overlay weights (``transformer/model.safetensors`` + ``config.json``)
    use the same key layout as ltx_core / sglang and do not embed config in safetensors
    metadata. Comfy-style single-file checkpoints keep using safetensors metadata.
    """
    import torch
    from ltx_core.loader.helpers import create_meta_model, load_state_dict
    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer.model_configurator import (
        LTXModelConfigurator,
        LTXV_MODEL_COMFY_RENAMING_MAP,
    )

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LTX checkpoint not found: {checkpoint}")

    torch_device = torch.device(device) if isinstance(device, str) else device
    if dtype is None:
        dtype = torch.bfloat16

    if _is_materialized_diffusers_checkpoint(checkpoint):
        config = _read_materialized_transformer_config(checkpoint)
        meta_model = create_meta_model(LTXModelConfigurator, config, ())
        loader = SafetensorsModelStateDictLoader()
        sd = load_state_dict(
            str(checkpoint), loader, DummyRegistry(), torch.device("cpu"), None,
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

    return SingleGPUModelBuilder(
        model_path=str(checkpoint),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    ).build(device=torch_device, dtype=dtype)


def ensure_materialized_model(hf_model_id: str) -> Path:
    """Materialize the overlay model via sglang (same pipeline as rollout).

    Downloads HF source weights + overlay metadata on first use, then caches
    under ``SGLANG_DIFFUSION_CACHE_ROOT/materialized_models/``.
    """
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
    diffusion_model: str | None,
    *,
    explicit_path: str | None = None,
    materialize: bool = True,
) -> str:
    """Resolve the single-file DiT checkpoint used by FSDP train.

    Resolution order:
    1. Explicit ``--sglang-transformer-weights-path`` / env override
    2. ``--diffusion-model`` pointing at a ``.safetensors`` file
    3. Overlay materialized ``transformer/model.safetensors`` for a HF model id
       (materializes via sglang on cache miss when ``materialize=True``)
    """
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"LTX transformer checkpoint not found: {path}")

    env_path = os.environ.get("MILES_LTX_TRANSFORMER_WEIGHTS_PATH")
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"LTX transformer checkpoint not found: {path}")

    if diffusion_model:
        path = Path(str(diffusion_model)).expanduser()
        if path.is_file() and path.suffix == ".safetensors":
            return str(path)

        if _is_hf_model_id(str(diffusion_model)):
            materialized_dir = resolve_materialized_model_dir(
                str(diffusion_model), materialize=materialize,
            )
            if materialized_dir is not None:
                checkpoint = _transformer_checkpoint_in_dir(materialized_dir)
                logger.info(
                    "LTX train: using materialized transformer %s (from %s)",
                    checkpoint,
                    materialized_dir,
                )
                return str(checkpoint)

    materialized_dir = resolve_materialized_model_dir(
        LTX_DEFAULT_HF_MODEL, materialize=materialize,
    )
    if materialized_dir is not None:
        checkpoint = _transformer_checkpoint_in_dir(materialized_dir)
        logger.info("LTX train: using default materialized transformer %s", checkpoint)
        return str(checkpoint)

    raise FileNotFoundError(
        "Could not resolve LTX transformer checkpoint. Pass --diffusion-model "
        f"Lightricks/LTX-2.3 (recommended) or a .safetensors override."
    )


def server_kwargs_extras(args) -> dict:
    """Extra ``ServerArgs`` kwargs; call only when ``is_ltx_model(args)``."""
    hf_model_id = resolve_hf_model_id(args)
    extras: dict = {"model_id": resolve_model_id(args)}

    # Only override rollout DiT when user explicitly passes a safetensors path.
    # For ``--diffusion-model Lightricks/LTX-2.3`` both sides use overlay defaults.
    explicit = getattr(args, "sglang_transformer_weights_path", None)
    weights_path = None
    if explicit:
        weights_path = resolve_transformer_checkpoint(
            getattr(args, "diffusion_model", None), explicit_path=explicit,
        )
    elif getattr(args, "diffusion_model", None) and str(args.diffusion_model).endswith(
        ".safetensors"
    ):
        weights_path = resolve_transformer_checkpoint(args.diffusion_model)

    if weights_path:
        extras["transformer_weights_path"] = weights_path
        logger.info(
            "LTX rollout: model_path=%s model_id=%s transformer_weights_path=%s",
            hf_model_id,
            extras["model_id"],
            weights_path,
        )
    else:
        logger.info(
            "LTX rollout: model_path=%s model_id=%s (overlay default DiT weights)",
            hf_model_id,
            extras["model_id"],
        )

    gemma_path = getattr(args, "ltx_gemma_path", None)
    if gemma_path:
        extras["component_paths"] = {"text_encoder": gemma_path}

    return extras


def patch_rollout_sampling_params(
    sampling_params: dict[str, Any],
    args: Namespace,
    *,
    evaluation: bool,
) -> None:
    """Apply LTX-specific rollout sampling fields in-place."""
    if getattr(args, "ltx_frames", None) is not None:
        sampling_params["num_frames"] = int(args.ltx_frames)
    if getattr(args, "ltx_fps", None) is not None:
        sampling_params["fps"] = int(args.ltx_fps)
    sampling_params["guidance_scale"] = 1.0
    sampling_params["negative_prompt"] = None

    if evaluation:
        return

    from miles.utils.sde_log_prob import normalize_dynamics_type

    dynamics = normalize_dynamics_type(getattr(args, "ltx_dynamics_type", "cps"))
    if dynamics == "dance_sde":
        raise NotImplementedError(
            "dance_sde rollout is not implemented in sglang-d flow_sde_sampling yet."
        )
    sampling_params["rollout_sde_type"] = dynamics
    if dynamics in ("cps", "ode"):
        sampling_params["rollout_log_prob_no_const"] = True
    elif dynamics == "flow_sde":
        ltx_sigma_min = getattr(args, "ltx_sigma_min", None)
        if ltx_sigma_min is not None:
            sampling_params["rollout_sigma_min"] = float(ltx_sigma_min)


def patch_rollout_engine_env_vars(env_vars: dict[str, str], args) -> None:
    """Add LTX-specific env vars for Ray rollout engine workers."""
    if not is_ltx_model(args):
        return

    from miles.backends.sglang_diffusion_utils.monkey_patches import LTX_ROLLOUT_PATCHES_ENV

    if getattr(args, "ltx_disable_av_cross_attn", False):
        env_vars["MILES_LTX_DISABLE_AV_CROSS"] = "1"
    for name in (LTX_ROLLOUT_PATCHES_ENV, "MILES_LTX_DISABLE_AV_CROSS"):
        if os.environ.get(name):
            env_vars[name] = os.environ[name]


def register_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--ltx-frames",
        type=int,
        default=25,
        help="LTX video frame count (e.g. 57 for verl-omni default).",
    )
    parser.add_argument(
        "--ltx-fps",
        type=float,
        default=24.0,
        help="LTX video fps for rollout VAE rescale.",
    )
    parser.add_argument(
        "--ltx-num-sde-steps",
        type=int,
        default=3,
        help="Number of denoising steps with SDE noise + log_prob during LTX rollout.",
    )
    parser.add_argument(
        "--ltx-sde-step-candidates",
        type=str,
        default=None,
        help="Comma-separated SDE step indices for LTX rollout (e.g. 0,1,...,9).",
    )
    parser.add_argument(
        "--ltx-dynamics-type",
        type=str,
        default="CPS",
        choices=["Flow-SDE", "CPS", "ODE", "Dance-SDE"],
        help="Stochastic dynamics for LTX SDE step during training.",
    )
    parser.add_argument(
        "--ltx-sigma-min",
        type=float,
        default=None,
        help="Override σ_min for LTX SDE step.",
    )
    parser.add_argument(
        "--ltx-disable-av-cross-attn",
        action="store_true",
        default=False,
        help="Disable LTX A2V/V2A cross-attn in sglang rollout (align with ltx_core video-only train).",
    )
    parser.add_argument(
        "--ltx-gemma-path",
        type=str,
        default=None,
        help=(
            "Deprecated: optional text_encoder override. "
            "When unset, sglang overlay materializes text_encoder from --diffusion-model."
        ),
    )
    parser.add_argument(
        "--pickscore-num-frames",
        type=int,
        default=3,
        help="Number of evenly spaced frames to score per video (LTX PickScore reward).",
    )


def validate_args(args: Namespace) -> None:
    ltx_gs = float(getattr(args, "diffusion_guidance_scale", 1.0))
    if ltx_gs != 1.0:
        logger.warning(
            "LTX rollout/train alignment expects --diffusion-guidance-scale 1.0 "
            "(no CFG); using %s may break log_prob parity.",
            ltx_gs,
        )
    if getattr(args, "fsdp_master_dtype", "fp32") == "fp32":
        logger.warning(
            "diffusion_model_type=ltx with fsdp_master_dtype=fp32 is unlikely to fit "
            "on small GPU counts; consider --fsdp-master-dtype bf16."
        )
