"""OpenPI π₀.₅ full-parameter supervised fine-tuning."""

from __future__ import annotations

import torch

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config


@register_train_pipeline_config("pi05")
class Pi05TrainPipelineConfig(TrainPipelineConfig):
    tasks = ("vla",)
    hf_ckpt_name_patterns = ("pi05", "pi0.5")
    supports_cfg_training = False
    # Start with official full fp32 training; lower precision needs separate numerical validation.
    enable_autocast = False
    model_backend_path = "miles.backends.fsdp_utils.models.pi05.backend.Pi05ModelBackend"
    sft_requires_visual_grid = False
    sft_streams = ("action",)
    # The continuous-action forward never consumes LM logits or the final prefix output.
    optimizer_state_allowed_missing = ["lm_head", "paligemma.model.language_model.norm"]

    @classmethod
    def validate_args(cls, args):
        if args.loss_type != "sft_loss" or not args.train_only:
            raise ValueError("π₀.₅ currently requires --loss-type sft_loss --train-only")
        if args.use_lora or args.sequence_parallel_size != 1:
            raise ValueError("π₀.₅ currently supports full-parameter FSDP/DP only (no LoRA or SP)")
        if args.fsdp_master_dtype != "fp32" or args.diffusion_forward_dtype != "fp32":
            raise ValueError("The initial OpenPI recipe requires fp32 master and forward dtypes")
        if args.fsdp_attention_backend not in (None, "eager"):
            raise ValueError("π₀.₅ training uses OpenPI eager attention")
        if args.update_weight_target_modules != ["policy"]:
            raise ValueError("π₀.₅ requires --update-weight-target-module policy")
        hooks = {
            "rollout_function_path": "miles.rollout.pi05_rollout.generate_rollout",
            "custom_prepare_train_batch_path": "miles.backends.fsdp_utils.models.pi05.loss.prepare_batch",
            "custom_loss_function_path": "miles.backends.fsdp_utils.models.pi05.loss.loss_formula",
        }
        for name, expected in hooks.items():
            if getattr(args, name) != expected:
                raise ValueError(f"π₀.₅ requires --{name.replace('_', '-')} {expected}")
        if args.fsdp_supervised_streams != ["action"]:
            raise ValueError("π₀.₅ requires --fsdp-supervised-streams action")

    def configure(self, args):
        from miles.backends.fsdp_utils.models.pi05.checkpoint import read_bundle

        model = read_bundle(args.hf_checkpoint)["model"]
        self.action_shape = (model["action_horizon"], model["action_dim"])
        # Final prefix-layer outputs have no path to the action loss, although their
        # K/V projections DO. Do not ignore the entire layer's optimizer state.
        from openpi.models import gemma

        last = gemma.get_config(model["paligemma_variant"]).depth - 1
        prefix = f"paligemma_with_expert.paligemma.model.language_model.layers.{last}."
        self.optimizer_state_allowed_missing = [
            *type(self).optimizer_state_allowed_missing,
            *(prefix + suffix for suffix in ("self_attn.o_proj", "post_attention_layernorm", "mlp")),
        ]

    def compute_noise_pred(self, *, model, latents_input, pos_cond, use_cfg, **kwargs):
        if use_cfg:
            raise ValueError("π₀.₅ does not use classifier-free guidance")
        # This hook's generic name is historical. The custom loss consumes OpenPI's
        # unreduced MSE directly, retaining its noise/time sampling and augmentation.
        actions = latents_input["action"]
        devices = [actions.device] if actions.is_cuda else []
        # Keep OpenPI's exact Beta(1.5, 1)*0.999+0.001 sampling, with distinct,
        # replayable rank-local noise/augmentation streams, independent of prefetch.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(pos_cond["rng_seed"])
            return {"action_loss": model(pos_cond["observation"], actions)}

    def prepare_cond_kwargs(self, cond, device):
        raise NotImplementedError("π₀.₅ observations are prepared by the OpenPI SFT data path")

    def cfg_combine(self, *args, **kwargs):
        raise NotImplementedError("π₀.₅ does not use classifier-free guidance")
