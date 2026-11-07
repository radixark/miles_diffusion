"""Experimental head-only PDD rollout. Keep the native DiT and rollout scheduler.

The projection emits a block-average velocity, so the existing ODE scheduler
performs the PDD update and retains its trajectory/conditioning capture hooks.
No fused-head cache or exported checkpoint is needed for online weight sync.
"""

import torch
from torch import nn

from miles.distillation.integration import engine_timesteps, make_grid
from miles.distillation.pdd import ParallelLinear


class PDDProjection(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.parallel = ParallelLinear(source, 32)
        self.grid = None
        self.start = None

    def forward(self, hidden):
        if self.grid is None or self.start not in range(0, 32, 4):
            raise ValueError("PDD projection requires a grid and block index")
        start = self.start
        predictions = self.parallel(hidden)
        # Reduce in FP32, even when the native forward runs under BF16 autocast.
        with torch.autocast(hidden.device.type, enabled=False):
            h = self.grid.diff()[start : start + 4].to(hidden.device)
            delta = torch.einsum("k,k...->...", h, predictions[start : start + 4].float())
            return delta / (self.grid[start + 4] - self.grid[start])


def apply():
    from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import QwenImage21PipelineConfig
    from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import QwenImage21Transformer2DModel
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
        QwenImage21DenoisingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import TimestepPreparationStage

    if getattr(QwenImage21DenoisingStage, "_miles_pdd", False):
        return
    original_load = QwenImage21Transformer2DModel.post_load_weights
    original_time = TimestepPreparationStage.forward
    original_prepare = QwenImage21DenoisingStage._prepare_denoising_loop
    original_step = QwenImage21DenoisingStage._run_denoising_step

    # Request and engine-size settings are validated by Miles' PDD arguments; the
    # checks here cover the patch's own constants and server-only options.
    def post_load(self):
        original_load(self)
        if not isinstance(self.proj_out, PDDProjection):
            self.proj_out = PDDProjection(self.proj_out)

    def timesteps(self, batch, server_args):
        if not isinstance(server_args.pipeline_config, QwenImage21PipelineConfig):
            return original_time(self, batch, server_args)
        if (batch.height, batch.width, batch.num_inference_steps) != (1024, 1024, 8):
            raise ValueError("PDD requires 1024x1024 and 8 steps")
        batch = original_time(self, batch, server_args)
        grid = make_grid(server_args.model_path, 32, 4096, batch.timesteps.device)
        batch.timesteps = engine_timesteps(grid[:-1:4]).to(grid.device)
        batch.scheduler.timesteps = batch.timesteps
        batch.scheduler.sigmas = grid[::4]
        batch.extra["pdd_grid"] = grid
        return batch

    def prepare(self, batch, server_args):
        if server_args.enable_torch_compile or server_args.enable_breakable_cuda_graph or server_args.cache_dit_config:
            raise ValueError("PDD graph/Cache-DiT modes are not validated")
        if server_args.dit_layerwise_offload or server_args.use_fsdp_inference:
            raise ValueError("PDD requires a resident native transformer")
        if len(batch.extra["qwen21_positive"]["layouts"]) != 1:
            raise ValueError("PDD requires batch size 1")
        if batch.extra["qwen21_positive"].get("condition_latents") is not None:
            raise ValueError("PDD does not support image conditioning")
        batch.latents = batch.latents.bfloat16().float()
        ctx = original_prepare(self, batch, server_args)
        if not isinstance(self.transformer.proj_out, PDDProjection):
            raise ValueError("PDD output heads were not installed")
        self.transformer.proj_out.grid = batch.extra["pdd_grid"]
        return ctx

    def step(self, ctx, step, batch, server_args):
        step.current_model.proj_out.start = step.step_index * 4
        with torch.autocast(ctx.latents.device.type, dtype=torch.bfloat16):
            original_step(self, ctx, step, batch, server_args)
        if not torch.isfinite(ctx.latents).all():
            raise FloatingPointError("Nonfinite PDD rollout")

    # Reuse the existing train/rollout prefix-cache policy, without replacing
    # the transformer forward, time embeddings, normalization, or RoPE.
    from .patch_qwen_image21 import apply as apply_qwen_image21

    apply_qwen_image21()
    # Native fast paths are bit-exact only against the engine's BF16-weight path;
    # with FP32 weights some reject the dtype (Q/K norm + RoPE). Use the eager path.
    from sglang.kernels.ops.diffusion.sites.bitexact_gate import BitExactFusionGate
    from sglang.multimodal_gen.runtime.models.dits import qwen_image21 as native

    for gate in vars(native).values():
        if isinstance(gate, BitExactFusionGate):
            gate.disable()
    QwenImage21Transformer2DModel.post_load_weights = post_load
    TimestepPreparationStage.forward = timesteps
    QwenImage21PipelineConfig.supports_dynamic_batching = lambda self: False
    QwenImage21DenoisingStage._prepare_denoising_loop = prepare
    QwenImage21DenoisingStage._run_denoising_step = step
    QwenImage21DenoisingStage._miles_pdd = True
