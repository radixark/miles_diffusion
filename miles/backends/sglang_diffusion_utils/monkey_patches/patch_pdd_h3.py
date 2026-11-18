"""PDD rollout for MiniMax H3 t2va. Keep the native DiT, denoise loop and rollout collector.

The native video and audio output projections become 32 heads each. The loop steps all
32 intervals of the shifted grids (one per H3 grid point of a 32-step request): the first
interval of each block runs the DiT and caches every head, the block's other three step on
the cache without a forward. The rollout trajectory returns each interval's start state of
both streams: video target rows, then audio rows zero-padded to the video width; its sigmas
are the video grid followed by the audio grid. The trainer's family hooks cut both back apart.

The block cache lives on the model, so an engine must run one request at a time (PDD validates
one GPU per engine and rollout microgroup size 1).
"""

import torch
from torch import nn

from miles.distillation.integration import BLOCK_SIZE as BLOCK
from miles.distillation.integration import NUM_INTERVALS as INTERVALS
from miles.distillation.pdd import ParallelLinear


class PDDHeads(nn.Module):
    """A native output projection as N heads, returned like ``ColumnParallelLinear``: ([rows, N*C], None)."""

    def __init__(self, source):
        super().__init__()
        self.parallel = ParallelLinear(source, INTERVALS)

    def forward(self, hidden):
        return self.parallel(hidden).permute(1, 0, 2).flatten(1), None


def heads_at(block, step_index):
    """Interval ``step_index``'s prediction from a block's cached [rows, N*C] head output."""
    return block.unflatten(-1, (INTERVALS, -1))[:, step_index].contiguous()


def pack_audio(trajectory, audio_steps, audio_sigmas):
    """Append the recorded audio rows (zero-padded to the video width) and the audio grid to the trajectory."""
    video = trajectory.latents
    if trajectory.latent_step_indices.tolist() != list(range(len(audio_steps))):
        raise ValueError("H3 PDD needs the full trajectory of both streams")
    if len(audio_sigmas) != len(audio_steps):
        raise ValueError("H3 PDD audio grid and audio states differ in length")
    audio = torch.stack(audio_steps).to(video.dtype).unsqueeze(0)
    padded = audio.new_zeros(*audio.shape[:-1], video.shape[-1])
    padded[..., : audio.shape[-1]] = audio
    trajectory.latents = torch.cat((video, padded), dim=2)
    audio_sigmas = torch.tensor([float(s) for s in audio_sigmas], dtype=trajectory.sigmas.dtype)
    trajectory.sigmas = torch.cat((trajectory.sigmas, audio_sigmas))
    return trajectory


def apply():
    from sglang.multimodal_gen.runtime.distributed import get_sp_world_size, get_tp_world_size
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        denoise_loop,
        minimax_h3_rollout,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        denoising,
        timestep_preparation,
    )

    stage_cls = denoising.MiniMaxH3DenoisingStage
    if getattr(stage_cls, "_miles_pdd", False):
        return
    original_load = MiniMaxH3DiTModel.post_load_weights
    original_time = timestep_preparation.MiniMaxH3TimestepPreparationStage.forward
    original_forward = stage_cls._forward_dit
    original_loop = denoise_loop.minimax_h3_denoise_loop
    original_build = minimax_h3_rollout.MiniMaxH3RolloutCollector.build_trajectory_data

    def post_load(self):
        original_load(self)
        if get_tp_world_size() != 1 or get_sp_world_size() != 1:
            raise ValueError("H3 PDD stacks the heads per GPU; serve with TP=SP=1")
        if self.final_layer._pdd_heads is not None:
            raise ValueError("H3 PDD trains live heads; do not load fused PDD heads")
        for name in ("video_out", "audio_out"):
            if not isinstance(getattr(self.final_layer, name), PDDHeads):
                setattr(self.final_layer, name, PDDHeads(getattr(self.final_layer, name)))

    def timesteps(self, batch, server_args):
        # A request for 8 steps runs the 32-interval grid; block boundaries are its every 4th point.
        if batch.num_inference_steps != INTERVALS // BLOCK:
            raise ValueError(f"H3 PDD requires {INTERVALS // BLOCK} inference steps")
        batch.num_inference_steps = INTERVALS
        return original_time(self, batch, server_args)

    def forward_dit(self, model, call_kwargs, step_index, **kwargs):
        if step_index % BLOCK == 0:
            video, audio = original_forward(self, model, call_kwargs, step_index, **kwargs)
            model._miles_pdd_block = (video, audio)
        video, audio = model._miles_pdd_block
        return heads_at(video, step_index), heads_at(audio, step_index)

    def loop(*, positive, initial_audio_rows, on_step=None, rollout_ctx=None, **kwargs):
        if rollout_ctx is None:
            return original_loop(positive=positive, initial_audio_rows=initial_audio_rows, on_step=on_step, **kwargs)
        audio_target = positive.audio_target_slice
        collector = rollout_ctx.collector
        collector.audio_steps = [initial_audio_rows[audio_target].detach().float().cpu().clone()]
        collector.audio_sigmas = list(kwargs["sigmas_audio"])

        def record(step, video_rows, audio_rows):
            # Called after both streams moved, so this is the audio state at interval step + 1.
            collector.audio_steps.append(audio_rows[audio_target].detach().float().cpu().clone())
            if on_step is not None:
                on_step(step, video_rows, audio_rows)

        return original_loop(
            positive=positive,
            initial_audio_rows=initial_audio_rows,
            on_step=record,
            rollout_ctx=rollout_ctx,
            **kwargs,
        )

    def build(self):
        data = original_build(self)
        if getattr(self, "audio_steps", None) is not None:
            pack_audio(data.dit_trajectory, self.audio_steps, self.audio_sigmas)
        return data

    MiniMaxH3DiTModel.post_load_weights = post_load
    timestep_preparation.MiniMaxH3TimestepPreparationStage.forward = timesteps
    stage_cls._forward_dit = forward_dit
    denoise_loop.minimax_h3_denoise_loop = loop
    minimax_h3_rollout.MiniMaxH3RolloutCollector.build_trajectory_data = build
    stage_cls._miles_pdd = True
