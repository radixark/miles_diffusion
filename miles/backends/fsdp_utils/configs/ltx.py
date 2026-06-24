"""LTX-2: model-family helpers + FSDP train pipeline config."""

from __future__ import annotations

import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import torch

from miles.backends.fsdp_utils.train_step_backend import LTXTrainStepBackend
from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config

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
    return Path(os.environ.get("SGLANG_DIFFUSION_CACHE_ROOT", "/data/wenhao/sgl_diffusion_cache"))


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
            f"Materialized LTX model at {materialized_dir} is missing " f"transformer/model.safetensors"
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
        raise FileNotFoundError(f"Materialized LTX checkpoint {checkpoint} is missing sibling config.json")
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
                str(diffusion_model),
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

    materialized_dir = resolve_materialized_model_dir(
        LTX_DEFAULT_HF_MODEL,
        materialize=materialize,
    )
    if materialized_dir is not None:
        checkpoint = _transformer_checkpoint_in_dir(materialized_dir)
        logger.info("LTX train: using default materialized transformer %s", checkpoint)
        return str(checkpoint)

    raise FileNotFoundError(
        "Could not resolve LTX transformer checkpoint. Pass --diffusion-model "
        "Lightricks/LTX-2.3 (recommended) or a .safetensors override."
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
            getattr(args, "diffusion_model", None),
            explicit_path=explicit,
        )
    elif getattr(args, "diffusion_model", None) and str(args.diffusion_model).endswith(".safetensors"):
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


def _normalize_ltx_dynamics_type(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    allowed = ("flow_sde", "cps", "ode", "dance_sde")
    if key not in allowed:
        raise ValueError(f"Unknown ltx dynamics_type {name!r}; expected one of {allowed}")
    return key


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

    dynamics = _normalize_ltx_dynamics_type(getattr(args, "ltx_dynamics_type", "cps"))
    if dynamics == "dance_sde":
        raise NotImplementedError("dance_sde rollout is not implemented in sglang-d flow_sde_sampling yet.")
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


# --- FSDP train pipeline config ---


@register_train_pipeline_config("ltx")
class LTXTrainPipelineConfig(TrainPipelineConfig):
    """Training-side adapter for LTX-2.3 video DiT."""

    train_step_backend_cls = LTXTrainStepBackend
    needs_timestep_scaling = False
    # Rollout stores σ×1000 in dit_trajectory.timesteps; CPS uses scheduler σ∈[0,1].
    sde_timestep_divisor = 1000.0

    lora_target_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "net.0.proj",
        "net.2",
    ]

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None:
            return {}
        kwargs: dict = {}
        if cond.encoder_hidden_states:
            ctx = torch.cat(cond.encoder_hidden_states).to(device)
            if ctx.ndim == 2:
                ctx = ctx.unsqueeze(0)
            kwargs["context"] = ctx
        if cond.audio_encoder_hidden_states:
            audio_ctx = torch.cat(cond.audio_encoder_hidden_states).to(device)
            if audio_ctx.ndim == 2:
                audio_ctx = audio_ctx.unsqueeze(0)
            kwargs["audio_context"] = audio_ctx
        if cond.encoder_attention_mask is not None:
            mask = cond.encoder_attention_mask.to(device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            kwargs["context_mask"] = mask
        if cond.audio_encoder_attention_mask is not None:
            audio_mask = cond.audio_encoder_attention_mask.to(device)
            if audio_mask.ndim == 1:
                audio_mask = audio_mask.unsqueeze(0)
            kwargs["audio_context_mask"] = audio_mask
        return kwargs

    def build_train_cond_kwargs(
        self,
        cond: CondKwargs | None,
        *,
        latents: torch.Tensor,
        args,
        device: torch.device,
    ) -> dict:
        """Merge rollout text embeds with locally rebuilt T2V geometry."""
        from miles.backends.fsdp_utils.ltx_geometry import build_ltx_t2v_geometry

        kwargs = self.prepare_cond_kwargs(cond, device)
        if "context" not in kwargs:
            raise ValueError("LTX train requires denoising_env.pos_cond_kwargs.encoder_hidden_states")

        ref = latents[0] if latents.ndim >= 2 else latents
        if ref.ndim == 2:
            batch_size, num_tokens, latent_dim = 1, ref.shape[0], ref.shape[1]
        else:
            batch_size, num_tokens, latent_dim = ref.shape[0], ref.shape[1], ref.shape[2]

        geom = build_ltx_t2v_geometry(
            batch_size=batch_size,
            num_tokens=num_tokens,
            latent_dim=latent_dim,
            height=int(getattr(args, "diffusion_height", 512)),
            width=int(getattr(args, "diffusion_width", 512)),
            num_frames=int(getattr(args, "ltx_frames", 25)),
            fps=float(getattr(args, "ltx_fps", 24.0)),
            device=device,
            dtype=ref.dtype,
        )
        kwargs.update(geom)
        return kwargs

    def build_sde_extra(
        self,
        scheduler,
        grids: dict,
        sample_indices: torch.Tensor,
        tstep_indices: torch.Tensor,
        args,
    ) -> dict | None:
        window = grids.get("sde_step_indices_window")
        if window is None:
            return None
        idx = window[sample_indices][:, tstep_indices].reshape(-1).long()
        return {
            "sde_step_indices": idx,
            "sigmas": scheduler.sigmas,
            "dynamics_type": getattr(args, "ltx_dynamics_type", "cps"),
            "sigma_min_override": getattr(args, "ltx_sigma_min", None),
        }

    def expand_cond_for_timestep_batch(self, cond_kwargs: dict, batch_size: int) -> dict:
        out: dict = {}
        for k, v in cond_kwargs.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.expand(batch_size, *v.shape[1:]) if v.shape[0] == 1 else v
            else:
                out[k] = v
        return out

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
    ) -> dict:
        out: dict = {}
        for key in per_sample_cond_kwargs[0]:
            values = [kw[key] for kw in per_sample_cond_kwargs if key in kw]
            if not values:
                continue
            if isinstance(values[0], torch.Tensor):
                out[key] = torch.cat(values, dim=0).to(device)
            else:
                out[key] = values
        return out

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        if scale == 1.0:
            return noise_pred_pos
        return noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)

    def preprocess_model_before_fsdp(self, model: torch.nn.Module) -> None:
        return None

    @staticmethod
    def _modality_timesteps_for_adaln(per_token_t: torch.Tensor) -> torch.Tensor:
        """Collapse per-token sigma to batch-global AdaLN input when uniform.

        sglang rollout builds temb with shape ``[B, 1, D]`` (scheduler timestep
        is batch-scalar expanded only for masking). ltx_core defaults to
        ``[B, T, D]`` when ``Modality.timesteps`` has length T, which diverges
        in AdaLN even when every active token shares the same sigma.
        """
        if per_token_t.ndim != 2 or per_token_t.shape[1] == 1:
            return per_token_t
        ref = per_token_t[:, :1]
        if torch.allclose(per_token_t, ref.expand_as(per_token_t), rtol=0.0, atol=0.0):
            return ref
        return per_token_t

    def forward_velocity(
        self,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        cond: dict,
    ) -> torch.Tensor:
        from ltx_core.model.transformer.modality import Modality
        from ltx_core.utils import to_denoised

        device = latents_input.device
        dtype = latents_input.dtype
        B = latents_input.shape[0]

        # dit_trajectory.timesteps are σ×1000; ltx_core AdaLN expects σ∈[0,1] and
        # multiplies by timestep_scale_multiplier (1000) internally.
        sigma_scaled = timesteps_input.to(latents_input.dtype)
        sigma_unit = sigma_scaled / float(self.sde_timestep_divisor)
        denoise_mask = cond["denoise_mask"].to(device)
        denoise_mask_2d = denoise_mask.squeeze(-1) if denoise_mask.ndim == 3 else denoise_mask
        denoise_mask_float = denoise_mask_2d.float()

        per_token_t = (sigma_unit.view(B, 1) * denoise_mask_2d).to(dtype)
        adaln_timesteps = self._modality_timesteps_for_adaln(per_token_t)

        video_modality = Modality(
            enabled=True,
            latent=latents_input,
            sigma=sigma_unit.reshape(B),
            timesteps=adaln_timesteps,
            positions=cond["positions"].to(dtype),
            context=cond["context"].to(dtype),
            context_mask=None,
        )
        with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype):
            velocity, _ = model(video=video_modality, audio=None, perturbations=None)

        per_token_t_3d = per_token_t.unsqueeze(-1) if per_token_t.ndim == 2 else per_token_t
        x0_pred = to_denoised(latents_input, velocity, per_token_t_3d).float()

        clean_latent = cond["clean_latent"].to(device).float()
        denoise_mask_3d = denoise_mask_float.unsqueeze(-1) if denoise_mask_float.ndim == 2 else denoise_mask_float
        x0_pred = x0_pred * denoise_mask_3d + clean_latent * (1.0 - denoise_mask_3d)

        sigma_safe = torch.clamp(sigma_unit, min=1e-8).view(B, 1, 1)
        velocity_for_sde = (latents_input.float() - x0_pred) / sigma_safe
        return velocity_for_sde.to(dtype)

    def forward_velocity_cfg_joint(
        self,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        joint_cond: dict,
    ) -> torch.Tensor:
        raise NotImplementedError("LTX trains with guidance_scale=1.0; --fsdp-cfg-batching is not supported.")

    def sde_step(
        self,
        scheduler,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
        prev_sample: torch.Tensor,
        *,
        noise_level: float,
        extra: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from miles.utils.sde_log_prob import sde_step_with_logprob

        if extra is None or "sigmas" not in extra or "sde_step_indices" not in extra:
            raise ValueError("LTXTrainPipelineConfig.sde_step requires extra={'sigmas','sde_step_indices',...}")
        sigmas = extra["sigmas"].to(sample.device).float()
        step_indices = extra["sde_step_indices"].to(sample.device).long()
        sigma_view = timesteps.float()
        sigma_next = sigmas[torch.clamp(step_indices + 1, max=len(sigmas) - 1)]

        dynamics_type = _normalize_ltx_dynamics_type(extra.get("dynamics_type", "cps"))
        if dynamics_type != "cps":
            raise NotImplementedError(
                f"LTXTrainPipelineConfig.sde_step supports dynamics_type='cps' only " f"(got {dynamics_type!r})."
            )

        prev, log_prob, prev_mean, std_dev_t = sde_step_with_logprob(
            None,
            noise_pred.float(),
            sigma_view,
            sample.float(),
            prev_sample.float(),
            noise_level=noise_level,
            sde_type="cps",
            sigma=sigma_view,
            sigma_prev=sigma_next,
        )
        if std_dev_t.ndim > 1:
            std_dev_t = std_dev_t.mean(dim=tuple(range(1, std_dev_t.ndim)))
        return prev, log_prob, prev_mean, std_dev_t
