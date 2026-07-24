"""Training-side pipeline config for diffusion models.

Model-specific logic for the GRPO training forward (after rollout has built
train-pair dicts):
  - prepare_cond_kwargs / collate / expand for DenoisingEnv
  - CFG combine
  - FSDP preprocess hooks

Trajectory unpacking for train-pair construction lives in
``miles.utils.train_data_utils.RolloutTrainDataConverter``.
"""

from __future__ import annotations

import abc
import os

import torch
from miles.utils.types import CondKwargs


_REGISTRY: dict[str, type[TrainPipelineConfig]] = {}


def register_train_pipeline_config(family: str):
    """Decorator: register a TrainPipelineConfig subclass under a family key (``sd3``, ``wan``, ...)."""

    def wrapper(cls):
        _REGISTRY[family.lower()] = cls
        return cls

    return wrapper


def _populate_registry() -> None:
    # Import every config module here (lazily — they import this module back);
    # registration is an import side effect.
    import importlib
    import pkgutil

    package = importlib.import_module("miles.backends.fsdp_utils.configs")
    for module_info in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package.__name__}.{module_info.name}")


def resolve_diffusion_model_family(model_ref: str) -> str:
    """Map a model reference to a family key by the refs each family declares."""
    override = os.environ.get("MILES_DIFFUSION_MODEL_FAMILY")
    if override:
        return override.strip().lower()

    _populate_registry()
    ref = str(model_ref).lower()
    for family, config_cls in _REGISTRY.items():
        if any(pattern in ref for pattern in config_cls.hf_ckpt_name_patterns):
            return family
    raise ValueError(
        f"Cannot resolve diffusion model family for '{model_ref}' "
        f"(known families: {list(_REGISTRY)}). "
        "Set MILES_DIFFUSION_MODEL_FAMILY to override."
    )


def get_train_pipeline_config_cls(family: str) -> type[TrainPipelineConfig]:
    """The TrainPipelineConfig class registered for a resolved family key."""
    cls = _REGISTRY.get(family.lower())
    if cls is None:
        raise ValueError(
            f"No TrainPipelineConfig registered for family '{family}'. " f"Known families: {list(_REGISTRY.keys())}"
        )
    return cls


class TrainPipelineConfig(abc.ABC):
    """Base class. Subclass per model family."""

    lora_target_modules: list[str] = ["to_q", "to_k", "to_v", "to_out.0"]
    needs_timestep_scaling: bool = True
    optimizer_state_allowed_missing: list[str] = []
    # Case-insensitive substrings matched against the checkpoint name (--diffusion-model).
    hf_ckpt_name_patterns: tuple[str, ...] = ()
    supports_cfg_training: bool = True
    # Rollout parity patch group applied by the engine (see monkey_patches; None = none).
    rollout_patch_group: str | None = None
    # Default component paths (miles custom-function style); CLI args override.
    model_backend_path: str = "miles.backends.fsdp_utils.model_backend.DiffusersModelBackend"

    @classmethod  # noqa: B027 — optional hook, deliberately non-abstract
    def validate_args(cls, args) -> None:
        """Family-specific arg validation/defaults; runs once at arg validation."""

    sde_timestep_divisor = 1.0

    def configure(self, args) -> None:  # noqa: B027  optional no-op hook, not abstract
        """Bind the request constants a family needs at train time; default binds none."""

    def compute_noise_pred(
        self,
        *,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        pos_cond: dict | None,
        neg_cond: dict | None,
        joint_cond: dict | None,
        use_cfg: bool,
        cfg_batching: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
    ) -> torch.Tensor:
        """Default diffusers forward with CFG; families with a different forward override."""

        def _forward(cond: dict) -> torch.Tensor:
            return model(
                hidden_states=latents_input,
                timestep=timesteps_input,
                return_dict=False,
                **cond,
            )[0]

        if not use_cfg:
            return _forward(pos_cond)
        if cfg_batching:
            # forward pos+neg as one joint batch to align with sglang-d
            joint_out = model(
                hidden_states=torch.cat([latents_input, latents_input], dim=0),
                timestep=torch.cat([timesteps_input, timesteps_input], dim=0),
                return_dict=False,
                **joint_cond,
            )[0]
            noise_pred_pos, noise_pred_neg = joint_out.chunk(2, dim=0)
        else:
            noise_pred_pos = _forward(pos_cond)
            noise_pred_neg = _forward(neg_cond)
        return self.cfg_combine(
            noise_pred_pos,
            noise_pred_neg,
            guidance_scale,
            true_cfg_scale=true_cfg_scale,
        )

    @abc.abstractmethod
    def prepare_cond_kwargs(
        self,
        cond: CondKwargs | None,
        device: torch.device,
    ) -> dict:
        """Convert CondKwargs to model-specific forward() kwargs."""

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
        pad_to_len: int | None = None,
    ) -> dict:
        """Stack a list of per-sample cond_kwargs (output of prepare_cond_kwargs)
        into a single batched dict suitable for one DiT forward over M samples.

        Model-specific because variable-length text embeds need padding + mask.
        Default: naive concat along batch dim, only valid when shapes match.

        ``pad_to_len`` is part of the contract so the trainer can uniformly ask
        every config to pad text to a shared width (the legacy window-wide
        seq_len) for bitwise grouping parity. Configs that do variable-length
        padding (e.g. Qwen-Image) must honor it; configs that concat
        fixed-length embeds (SD3, Wan2.2, LTX) accept and ignore it.
        """
        raise NotImplementedError(
            "Must implement collate_cond_for_sample_batch to enable micro-batch-size > 1 in fsdp training"
        )

    def maybe_legacy_window_pad_len(self, conds: list) -> int | None:
        """LEGACY 2D parity: seq_len to pad text embeds to (whole-window max), or None for
        fixed-length-cond models. Qwen-Image overrides. TODO: remove with the legacy 2D path."""
        return None

    @abc.abstractmethod
    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        """Apply classifier-free guidance. Model-specific (e.g. rescale or not)."""

    def postprocess_model_after_materialize(self, model: torch.nn.Module) -> None:
        """Postprocess the model after FSDP wrap + weight materialization (default: no-op)."""
        return None
