"""Supervised pose control for the single-transformer Wan2.2 TI2V 5B base."""

import torch

from miles.backends.fsdp_utils.model_backend import MilesModelBackend
from .train_pipeline_config import register_train_pipeline_config
from .wan2_2 import Wan2_2TrainPipelineConfig


class WanControlNetBackend(MilesModelBackend):
    def load_component(self, component, *, checkpoint_path, master_dtype, materialize_weights):
        return self._pkg.loading.load_component(
            component,
            checkpoint_path=checkpoint_path,
            master_dtype=master_dtype,
            materialize_weights=materialize_weights,
            num_control_blocks=self.config.num_control_blocks,
            controlnet_checkpoint=self.config.controlnet_checkpoint,
        )


@register_train_pipeline_config("wan_controlnet")
class WanControlNetTrainPipelineConfig(Wan2_2TrainPipelineConfig):
    # Explicit opt-in: do not change the existing Wan family auto-detection.
    hf_ckpt_name_patterns = ()
    model_package = "miles.backends.fsdp_utils.models.wan_controlnet"
    model_backend_path = "miles.backends.fsdp_utils.configs.wan_controlnet.WanControlNetBackend"
    input_dtype_policy = {"latents": "default", "cond": "default", "timestep": "fp32"}
    num_control_blocks = 4
    controlnet_checkpoint = None

    @classmethod
    def validate_args(cls, args):
        if args.loss_type != "sft_loss" or not args.train_only:
            raise ValueError("wan_controlnet currently supports --loss-type sft_loss --train-only")
        if args.sequence_parallel_size != 1:
            raise ValueError("wan_controlnet currently requires --sequence-parallel-size 1")
        if args.use_lora:
            raise ValueError("wan_controlnet trains its control branch; --use-lora is unsupported")
        if args.wan_controlnet_num_blocks < 1:
            raise ValueError("--wan-controlnet-num-blocks must be positive")
        if args.update_weight_target_module != "transformer":
            raise ValueError("wan_controlnet requires --update-weight-target-module transformer")

    def configure(self, args):
        self.num_control_blocks = args.wan_controlnet_num_blocks
        self.controlnet_checkpoint = args.wan_controlnet_checkpoint

    def postprocess_model_after_materialize(self, model):
        # Use the same patch embedding arithmetic for training and Diffusers inference.
        pass

    def component_for_timestep(self, timestep, num_train_timesteps):
        return "transformer"

    def select_guidance_scale(self, timestep, num_train_timesteps, guidance_scale, guidance_scale_2):
        return guidance_scale

    def collate_cond_for_sample_batch(self, per_sample_cond_kwargs, device, pad_to_len=None):
        return {
            key: torch.cat([kw[key] for kw in per_sample_cond_kwargs], dim=0).to(device)
            for key in ("encoder_hidden_states", "control_latents")
        }
