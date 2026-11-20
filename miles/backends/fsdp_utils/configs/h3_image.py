"""MiniMax H3 Image: t2i Flow-GRPO on the release's PDD grid, LoRA only."""

from __future__ import annotations

from argparse import Namespace

import torch

from .h3 import H3TrainPipelineConfig, _without_padding_tail
from .train_pipeline_config import register_train_pipeline_config


@register_train_pipeline_config("h3_image")
class H3ImageTrainPipelineConfig(H3TrainPipelineConfig):
    """MiniMax H3 Image: the H3 DiT without audio rows, with one output head per denoising step."""

    tasks = ("t2i",)
    hf_ckpt_name_patterns = ("minimax-h3-image", "minimax_h3_image", "/h3-image", "/h3_image")
    # the release's sigma grid, one output head per step
    num_denoising_steps = 6

    @classmethod
    def validate_args(cls, args: Namespace) -> None:
        from miles.utils.arguments import rollout_merges_lora

        if args.loss_type != "policy_loss":
            raise ValueError(
                "H3 Image trains with --loss-type policy_loss only: its DiT picks an output head by the rollout's "
                "denoising step, which no other built-in loss sets, and custom_loss is untested"
            )
        has_trainer = not args.debug_rollout_only
        syncs_weights_to_rollout = has_trainer and not args.train_only
        if has_trainer and not args.use_lora:
            raise ValueError(
                "H3 Image runs its trainer with LoRA adapters only (--use-lora): full-weight training would update "
                "the DiT's per-step output heads, each from the ranks whose pairs came from its step, which is untested"
            )
        if syncs_weights_to_rollout and not rollout_merges_lora(args):
            raise ValueError(
                "H3 Image syncs its LoRA as adapters (--lora-ipc-weight-sync under --colocate, plain --use-lora "
                "otherwise): a merged sync sends the DiT's per-step output heads, which the rollout engine cannot "
                "receive"
            )
        if syncs_weights_to_rollout and args.diffusion_model_family != cls.model_family:
            raise ValueError(
                "H3 Image named by --train-pipeline-config-path cannot sync to a rollout engine: the fused-qkv LoRA "
                "group is looked up by --diffusion-model-family, which a config path leaves unset. Let "
                "--hf-checkpoint or --diffusion-model-family name the family, or run --train-only"
            )
        if args.diffusion_flow_shift is not None:
            raise ValueError("H3 Image samples on its release's sigma grid; drop --diffusion-flow-shift")
        if args.sequence_parallel_size != 1:
            raise ValueError("H3 Image is untested with trainer sequence parallelism; set --sequence-parallel-size 1")
        if args.diffusion_num_steps != cls.num_denoising_steps or args.diffusion_eval_num_steps not in (
            None,
            cls.num_denoising_steps,
        ):
            raise ValueError(
                f"H3 Image has one output head per step of its release's {cls.num_denoising_steps}-step grid; set "
                f"--diffusion-num-steps {cls.num_denoising_steps} and drop --diffusion-eval-num-steps"
            )
        if args.rollout_microgroup_size != 1 or args.diffusion_output_num_frames != 1:
            raise ValueError(
                "H3 Image rollout serves one image per request; keep --rollout-microgroup-size and "
                "--diffusion-output-num-frames at 1"
            )
        # miles_validate_args has folded the sample x timestep tiles into micro_batch_size
        if args.micro_batch_size != 1:
            raise ValueError(
                "H3 Image's packed forward takes one pair; keep --micro-batch-size, --micro-batch-size-sample and "
                "--micro-batch-size-tstep at 1"
            )
        if args.diffusion_quality not in ("exact", "lossless"):
            raise ValueError(
                f"H3 Image rollout serves --diffusion-quality exact or lossless, got {args.diffusion_quality!r}"
            )
        if args.diffusion_true_cfg_scale not in (None, 1.0):
            raise ValueError("H3 Image rollout is unguided; drop --diffusion-true-cfg-scale")

    @classmethod
    def apply_rollout_sampling_params(
        cls,
        args: Namespace,
        sampling_params: dict,
        extra_sampling_params: dict,
    ) -> None:
        # the engine generates a request below its default area on a larger canvas and resizes the image back
        extra_sampling_params["min_generation_area"] = int(args.diffusion_width) * int(args.diffusion_height)

    def compute_noise_pred(
        self,
        *,
        model: torch.nn.Module,
        latents_input: dict[str, torch.Tensor],
        timesteps_input: dict[str, torch.Tensor],
        denoising_step_indices: torch.Tensor | None,
        pos_cond: dict | None,
        neg_cond: dict | None,
        joint_cond: dict | None,
        use_cfg: bool,
        cfg_batching: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
    ) -> dict[str, torch.Tensor]:
        """Run the H3 Image forward on the target image rows, through the output head of the pair's rollout step."""
        del neg_cond, joint_cond, use_cfg, cfg_batching, guidance_scale, true_cfg_scale
        if denoising_step_indices is None:
            raise ValueError(
                "H3 Image picks its output head by each pair's denoising step, which these pairs do not carry: "
                "an expander that feeds Flow-GRPO must set denoising_step_index on every pair"
            )
        step = int(denoising_step_indices.item())
        num_heads = len(model.pdd_proj_out)
        # a negative step would index a head from the end
        if not 0 <= step < num_heads:
            raise ValueError(
                f"denoising_step_index {step} is not one of H3 Image's {num_heads} per-step output heads "
                f"(0..{num_heads - 1})"
            )
        cond = pos_cond
        visual = latents_input["visual"]
        device, dtype = visual.device, visual.dtype
        if visual.shape[0] != 1:
            raise NotImplementedError("H3 Image packed forward supports batch size 1 for now")

        layout = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in cond["h3_packed_layout"].items()}
        token_tags, position_ids = _without_padding_tail(layout, torch.as_tensor(cond["h3_token_tags"], device=device))
        if not bool(layout["update_mask"].all()):
            raise NotImplementedError("H3 Image trains t2i only; this layout carries reference image rows")

        out = model(
            hidden_states=visual,
            encoder_hidden_states=cond["encoder_hidden_states"].to(device=device, dtype=dtype),
            # H3 time runs from t=0 noise to t=1 clean; every row of a t2i sequence is at the target's time
            timestep=(1.0 - timesteps_input["visual"].float() / float(self.num_train_timesteps)).view(-1).to(dtype),
            timestep_indices=torch.zeros(token_tags.shape[0], device=device, dtype=torch.long),
            token_tags=token_tags.long(),
            position_ids=position_ids.to(device=device, dtype=torch.float32),
            image_indices=layout["img_pos"].view(-1).long(),
            text_indices=layout["text_pos"].view(-1).long(),
            pdd_step_index=step,
            return_dict=False,
        )
        # H3 predicts clean - noise; the shared flow objective predicts noise - clean.
        return {"visual": (-out[0]).to(dtype)}
