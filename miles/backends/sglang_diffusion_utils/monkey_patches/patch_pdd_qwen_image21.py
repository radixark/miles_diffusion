"""PDD rollout for Qwen-Image 2.1: keep the native DiT and rollout scheduler, replace only ``proj_out``.

The ODE scheduler steps all 32 intervals. The first interval of each block runs the
DiT, whose 32 heads are cached; the block's other three intervals step on the cached
heads without a forward. The native trajectory capture thus returns every interval's
start state (33 states, 8 forwards), which the trainer's teacher target starts from.
No fused-head cache or exported checkpoint is needed for online weight sync.
"""

import torch
from torch import nn

from miles.distillation.integration import (
    BLOCK_SIZE,
    IMAGE_TOKENS,
    NUM_INTERVALS,
    RESOLUTION,
    engine_timesteps,
    make_grid,
)
from miles.distillation.pdd import ParallelLinear


class PDDProjection(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.parallel = ParallelLinear(source, NUM_INTERVALS)
        self.interval = None
        self.block = None

    def forward(self, hidden):
        # One forward per block; FP32 head outputs, even under BF16 autocast.
        self.block = self.parallel(hidden).float()
        return self.block[0]


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
    original_predict = QwenImage21DenoisingStage._predict_noise

    # Request and engine-size settings are validated by Miles' PDD arguments; the
    # checks here cover the patch's own constants and server-only options.
    def post_load(self):
        original_load(self)
        if not isinstance(self.proj_out, PDDProjection):
            self.proj_out = PDDProjection(self.proj_out)

    def timesteps(self, batch, server_args):
        if not isinstance(server_args.pipeline_config, QwenImage21PipelineConfig):
            return original_time(self, batch, server_args)
        if (batch.height, batch.width, batch.num_inference_steps) != (
            RESOLUTION,
            RESOLUTION,
            NUM_INTERVALS // BLOCK_SIZE,
        ):
            raise ValueError(f"PDD requires {RESOLUTION}x{RESOLUTION} and {NUM_INTERVALS // BLOCK_SIZE} steps")
        batch = original_time(self, batch, server_args)
        grid = make_grid(server_args.model_path, NUM_INTERVALS, IMAGE_TOKENS, batch.timesteps.device)
        # Every interval is a scheduler step; only block starts reach the DiT, at these timesteps.
        batch.timesteps = engine_timesteps(grid[:-1]).to(grid.device)
        batch.scheduler.timesteps = batch.timesteps
        batch.scheduler.sigmas = grid
        batch.num_inference_steps = NUM_INTERVALS
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
        return ctx

    def predict(self, current_model, *args, **kwargs):
        head = current_model.proj_out
        if head.interval % BLOCK_SIZE == 0:
            original_predict(self, current_model, *args, **kwargs)
        return head.block[head.interval]

    def step(self, ctx, step, batch, server_args):
        step.current_model.proj_out.interval = step.step_index
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
    QwenImage21DenoisingStage._predict_noise = predict
    QwenImage21DenoisingStage._miles_pdd = True
