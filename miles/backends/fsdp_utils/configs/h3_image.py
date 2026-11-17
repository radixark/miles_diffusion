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
    hf_ckpt_name_patterns = ("minimax-h3-image", "minimax_h3_image", "/h3-image")

    @classmethod
    def validate_args(cls, args: Namespace) -> None:
        from miles.utils.arguments import rollout_merges_lora

        if args.loss_type != "policy_loss":
            raise ValueError(
                "H3 Image trains with --loss-type policy_loss only: its DiT picks an output head by the rollout's "
                "denoising step, which no other loss provides"
            )
        if not args.train_only and not rollout_merges_lora(args):
            raise ValueError(
                "H3 Image trains LoRA adapters and syncs them as adapters (--use-lora --lora-ipc-weight-sync): "
                "every other sync sends the DiT's per-step output heads, which the rollout engine cannot receive"
            )
        if args.diffusion_flow_shift is not None:
            raise ValueError("H3 Image samples on its release's sigma grid; drop --diffusion-flow-shift")
        if args.sequence_parallel_size != 1:
            raise ValueError("H3 Image is untested with trainer sequence parallelism; set --sequence-parallel-size 1")

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
                "H3 Image picks its output head by the rollout's denoising step, which this loss does not provide"
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
            pdd_step_index=int(denoising_step_indices.item()),
            return_dict=False,
        )
        # H3 predicts clean - noise; the shared flow objective predicts noise - clean.
        return {"visual": (-out[0]).to(dtype)}
