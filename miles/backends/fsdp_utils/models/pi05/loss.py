"""Bridge official unreduced flow-matching loss to miles' per-sample sum contract."""

from __future__ import annotations

import torch

from miles.backends.fsdp_utils.loss_hub.types import PreparedBatch
from miles.utils.hash_utils import stable_hash


def stack_tree(items, device):
    if isinstance(items[0], dict):
        if any(item.keys() != items[0].keys() for item in items):
            raise ValueError("All observations must have the same keys and camera masks")
        return {key: stack_tree([item[key] for item in items], device) for key in items[0]}
    return torch.stack([torch.as_tensor(item) for item in items]).to(device)


def prepare_batch(ctx, batch, *, pad_to_len=None):
    from openpi.models.model import Observation

    if not batch:
        raise ValueError("π₀.₅ microbatch must not be empty")
    actions = torch.stack([pair["actions"] for pair in batch]).to(device=ctx.device, dtype=torch.float32)
    expected = (len(batch), *ctx.train_pipeline_config.action_shape)
    if tuple(actions.shape) != expected:
        raise ValueError(f"Expected action shape {expected}, got {tuple(actions.shape)}")
    observation = Observation.from_dict(stack_tree([pair["observation"] for pair in batch], ctx.device))
    return PreparedBatch(
        latents={"action": actions},
        timesteps={},
        timesteps_for_model={},
        model=ctx.models["policy"],
        component_name="policy",
        guidance_scale=1.0,
        use_cfg=False,
        cfg_batching=False,
        true_cfg_scale=None,
        pos_cond={
            "observation": observation,
            "rng_seed": stable_hash("pi05", ctx.args.seed, ctx.rollout_id, ctx.microbatch_id, ctx.dp_rank),
        },
        neg_cond=None,
        joint_cond=None,
        advantage=torch.ones(len(batch), device=ctx.device),
    )


def loss_formula(ctx, batch, prepared, *, new_pred, old_pred, ref_pred, metrics, **kwargs):
    losses = new_pred["action_loss"]
    if losses.shape != prepared.latents["action"].shape:
        raise ValueError("OpenPI must return unreduced [batch, horizon, padded_action_dim] MSE")
    # Include padded dimensions exactly as the official training loop does. The
    # actor divides by num_local_pairs and accumulates microbatches before stepping.
    loss_sum = losses.float().flatten(1).mean(1).sum()
    metrics.emit_mean("loss", total=loss_sum.detach(), count=len(batch))
    return loss_sum
