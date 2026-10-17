"""Run only the conditional sample when the FLUX.3 image CFG scale is <= 1.

With a scale of 1 the engine still encodes and denoises the whole CFG pair and mixes
``uncond + 1 * (cond - uncond)``: two forwards per step, and not bitwise the conditional
velocity. DiffusionNFT trains and samples the unguided velocity, so the patched denoise
loop runs only the conditional half of the pair, one forward per step; the text and
rollout stages are unchanged.
"""

from __future__ import annotations

import torch
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# sglang-d's logger: the scheduler process has no handler for a plain logging.getLogger yet.
logger = init_logger(__name__)

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import flux3_image as stage_mod

    orig_denoise_packed = stage_mod.denoise_packed

    def denoise_conditional(dit, packed, schedule, *, step_callback=None, mode="image", transition=None):
        """denoise_packed's loop on the conditional half alone: one forward per step, no CFG mix."""
        # The pair is packed branch-major, [negatives..., positives...]; the slice shares the
        # pair's timestep buffers.
        half = packed.num_samples // 2
        conditional = packed.slice_samples(half, packed.num_samples)
        device_type = conditional.data[mode].device.type
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device_type == "cuda"):
            for step in range(len(schedule) - 1):
                timestep = schedule[step].item()
                delta = (schedule[step + 1] - schedule[step]).item()
                conditional.timesteps[mode].fill_(timestep)
                with stage_mod._dit_step(step, step_callback):
                    velocity = dit(packed=conditional)[mode]
                sample = conditional.data[mode]
                if transition is None:
                    conditional.data[mode] = sample + delta * velocity
                else:
                    conditional.data[mode] = transition(step=step, sample=sample, velocity=velocity)
        return conditional.data[mode]

    logged = False

    def denoise_packed(dit, packed, schedule, guidance, *, sequential_cfg=False, **kwargs):
        nonlocal logged
        if guidance > 1.0:
            return orig_denoise_packed(dit, packed, schedule, guidance, sequential_cfg=sequential_cfg, **kwargs)
        # CFG parallel runs one sample of the pair per rank; with CFG off there is nothing to split.
        if isinstance(dit, stage_mod._CFGParallelModel):
            raise ValueError("FLUX.3 image CFG off does not support --enable-cfg-parallel")
        if not logged:
            logger.info("FLUX.3 image CFG off: one conditional forward per step")
            logged = True
        return denoise_conditional(dit, packed, schedule, **kwargs)

    stage_mod.denoise_packed = denoise_packed
    _APPLIED = True
    logger.info("Applied FLUX.3 image CFG-off rollout patch")
