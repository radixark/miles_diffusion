"""Prepare OpenPI action flow matching for miles' shared SFT velocity loss."""

from __future__ import annotations

import torch

# Keep the original dotted hook path compatible while using the shared formula.
from miles.backends.fsdp_utils.loss_hub.sft import sft_loss_formula as loss_formula
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext, PreparedBatch
from miles.utils.hash_utils import stable_hash

__all__ = ["prepare_batch", "loss_formula"]


def stack_tree(items, device):
    if isinstance(items[0], dict):
        if any(item.keys() != items[0].keys() for item in items):
            raise ValueError("All observations must have the same keys and camera masks")
        return {key: stack_tree([item[key] for item in items], device) for key in items[0]}
    return torch.stack([torch.as_tensor(item) for item in items]).to(device)


def prepare_batch(ctx: DiffusionLossContext, batch: list[dict], *, pad_to_len=None) -> PreparedBatch:
    if not batch:
        raise ValueError("π₀.₅ microbatch must not be empty")
    actions = torch.stack([pair["actions"] for pair in batch]).to(device=ctx.device, dtype=torch.float32)
    expected = (len(batch), *ctx.train_pipeline_config.action_shape)
    if tuple(actions.shape) != expected:
        raise ValueError(f"Expected action shape {expected}, got {tuple(actions.shape)}")
    observation = stack_tree([pair["observation"] for pair in batch], ctx.device)
    model = ctx.models["policy"]
    seed = stable_hash("pi05", ctx.args.seed, ctx.rollout_id, ctx.microbatch_id, ctx.dp_rank)
    devices = [actions.device] if actions.is_cuda else []
    # Reuse OpenPI's Gaussian noise and Beta(1.5, 1)*0.999+0.001 time sampling.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        noise = model.sample_noise(actions.shape, actions.device)
        time = model.sample_time(len(batch), actions.device)
    sigma = time[:, None, None]
    return PreparedBatch(
        latents={"action": sigma * noise + (1 - sigma) * actions},
        timesteps={"action": time},
        timesteps_for_model={"action": time},
        model=model,
        component_name="policy",
        guidance_scale=1.0,
        use_cfg=False,
        cfg_batching=False,
        true_cfg_scale=None,
        pos_cond={
            "observation": observation,
            "rng_seed": stable_hash(seed, "augmentation"),
        },
        neg_cond=None,
        joint_cond=None,
        advantage=torch.ones(len(batch), device=ctx.device),
        extras={"target": {"action": noise - actions}, "sigmas": {"action": time}},
    )
