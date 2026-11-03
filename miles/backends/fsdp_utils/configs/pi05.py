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
    model_backend_path = "miles.backends.fsdp_utils.model_backend.MilesModelBackend"
    model_package = "miles.backends.fsdp_utils.models.pi05"
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
        if args.fsdp_supervised_streams != ["action"]:
            raise ValueError("π₀.₅ requires --fsdp-supervised-streams action")

    def configure(self, args):
        from miles.backends.fsdp_utils.models.pi05.bundle import read_bundle

        model = read_bundle(args.hf_checkpoint)["model"]
        self.action_shape = (model["action_horizon"], model["action_dim"])
        # Final prefix-layer outputs have no path to the action loss, although their
        # K/V projections DO. Do not ignore the entire layer's optimizer state.
        from miles.backends.fsdp_utils.models.pi05.model import gemma_config

        last = gemma_config(model["paligemma_variant"]).num_hidden_layers - 1
        prefix = f"paligemma_with_expert.paligemma.model.language_model.layers.{last}."
        self.optimizer_state_allowed_missing = [
            *type(self).optimizer_state_allowed_missing,
            *(prefix + suffix for suffix in ("self_attn.o_proj", "post_attention_layernorm", "mlp")),
        ]

    def compute_noise_pred(self, *, model, latents_input, timesteps_input, pos_cond, use_cfg, **kwargs):
        if use_cfg:
            raise ValueError("π₀.₅ does not use classifier-free guidance")
        noisy_actions = latents_input["action"]
        devices = [noisy_actions.device] if noisy_actions.is_cuda else []
        # Augmentation has its own replayable rank-local stream; noise and time
        # were sampled by the prepare hook. Call __call__ so FSDP hooks execute.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(pos_cond["rng_seed"])
            return {"action": model(pos_cond["observation"], noisy_actions, timesteps_input["action"])}

    def prepare_cond_kwargs(self, cond, device):
        raise NotImplementedError("π₀.₅ observations are supplied as train pairs by an external data hook")

    def cfg_combine(self, *args, **kwargs):
        raise NotImplementedError("π₀.₅ does not use classifier-free guidance")
