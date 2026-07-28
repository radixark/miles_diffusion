"""DiT forward input preparation hooks (actor pipeline step before forward).

Custom algorithms swap ``--custom-prepare-train-batch-path``; loss formulas live
in ``losses.py`` / algorithm plugins (e.g. ``nft.py``).
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

import torch

from miles.backends.fsdp_utils.loss_hub.context import DiffusionLossContext, PreparedBatch
from miles.backends.fsdp_utils.loss_hub.nft import corrupt, sample_noise
from miles.utils.misc import load_function

PrepareFn = Callable[..., PreparedBatch]


def _cast_cond_to_dtype(cond: dict, dtype: torch.dtype) -> dict:
    out = {}
    for k, v in cond.items():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            out[k] = v.to(dtype=dtype)
        else:
            out[k] = v
    return out


def _stack_pair_field(batch: list[dict], key: str, device: torch.device) -> torch.Tensor:
    return torch.stack([pair[key] for pair in batch]).to(device=device, dtype=torch.float32)


def prepare_flow_grpo_batch(
    ctx: DiffusionLossContext,
    batch: list[dict],
    *,
    pad_to_len: int | None = None,
) -> PreparedBatch:
    """Stack SDE-pair fields and build CFG conditioning (guidance from args)."""
    args = ctx.args
    device = ctx.device
    config = ctx.train_pipeline_config
    num_train_timesteps = int(ctx.scheduler.config.num_train_timesteps)
    bsz = len(batch)

    latents = _stack_pair_field(batch, "latent", device)
    next_latents = _stack_pair_field(batch, "next_latent", device)
    timesteps = _stack_pair_field(batch, "timestep", device)
    next_timesteps = _stack_pair_field(batch, "next_timestep", device)
    log_prob_old = _stack_pair_field(batch, "log_prob_old", device)
    advantage = torch.tensor(
        [float(pair["advantage"]) for pair in batch],
        device=device,
        dtype=torch.float32,
    )
    advantage = torch.clamp(advantage, -args.diffusion_adv_clip_max, args.diffusion_adv_clip_max)

    guidance_scale = args.diffusion_guidance_scale
    true_cfg_scale = args.diffusion_true_cfg_scale
    cfg_scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
    use_cfg = cfg_scale > 0

    if len(ctx.models) == 1:
        component_name, model = next(iter(ctx.models.items()))
    else:
        components = {config.component_for_timestep(t, num_train_timesteps) for t in timesteps.tolist()}
        if len(components) > 1:
            raise ValueError(
                f"Micro-batch mixes denoising phases {sorted(components)}; set "
                "--micro-batch-size 1 so each forward is phase-pure (one DiT, one CFG scale)."
            )
        component_name = components.pop()
        model = ctx.models[component_name]
        guidance_scale = config.select_guidance_scale(
            float(timesteps[0]),
            num_train_timesteps,
            guidance_scale,
            args.diffusion_guidance_scale_2,
        )

    if config.needs_timestep_scaling:
        timesteps_for_model = timesteps / float(num_train_timesteps)
    else:
        timesteps_for_model = timesteps

    pos_list = [config.prepare_cond_kwargs(batch[i]["denoising_env"].pos_cond_kwargs, device) for i in range(bsz)]
    neg_list = (
        [config.prepare_cond_kwargs(batch[i]["denoising_env"].neg_cond_kwargs, device) for i in range(bsz)]
        if use_cfg
        else None
    )
    cfg_batching = use_cfg and bool(args.fsdp_cfg_batching)
    joint_cond = pos_cond = neg_cond = None
    if cfg_batching:
        joint_cond = _cast_cond_to_dtype(
            config.collate_cond_for_sample_batch(pos_list + neg_list, device, pad_to_len=pad_to_len),
            ctx.forward_dtype,
        )
    else:
        pos_cond = _cast_cond_to_dtype(
            config.collate_cond_for_sample_batch(pos_list, device, pad_to_len=pad_to_len),
            ctx.forward_dtype,
        )
        if use_cfg and neg_list is not None:
            neg_cond = _cast_cond_to_dtype(
                config.collate_cond_for_sample_batch(neg_list, device, pad_to_len=pad_to_len),
                ctx.forward_dtype,
            )

    return PreparedBatch(
        latents=latents,
        timesteps=timesteps,
        timesteps_for_model=timesteps_for_model,
        model=model,
        component_name=component_name,
        guidance_scale=guidance_scale,
        use_cfg=use_cfg,
        cfg_batching=cfg_batching,
        true_cfg_scale=true_cfg_scale if use_cfg else None,
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        joint_cond=joint_cond,
        advantage=advantage,
        extras={
            "next_latents": next_latents,
            "next_timesteps": next_timesteps,
            "log_prob_old": log_prob_old,
        },
    )


def prepare_nft_batch(
    ctx: DiffusionLossContext,
    batch: list[dict],
    *,
    pad_to_len: int | None = None,
) -> PreparedBatch:
    """Corrupt clean x0 at each pair's sigma; CFG-free cond."""
    if len(ctx.models) != 1:
        raise ValueError("DiffusionNFT currently supports a single DiT component (SD3)")
    device = ctx.device
    config = ctx.train_pipeline_config
    bsz = len(batch)
    x0 = torch.stack([pair["x0"] for pair in batch]).to(device=device, dtype=torch.float32)
    t = torch.tensor([float(pair["timestep"]) for pair in batch], device=device, dtype=torch.float32)
    advantage = torch.tensor([float(pair["advantage"]) for pair in batch], device=device, dtype=torch.float32)

    component_name, model = next(iter(ctx.models.items()))
    pos_list = [config.prepare_cond_kwargs(batch[i]["denoising_env"].pos_cond_kwargs, device) for i in range(bsz)]
    pos_cond = _cast_cond_to_dtype(
        config.collate_cond_for_sample_batch(pos_list, device, pad_to_len=pad_to_len),
        ctx.forward_dtype,
    )

    num_train_timesteps = int(getattr(ctx.scheduler.config, "num_train_timesteps", 1000))
    if config.needs_timestep_scaling:
        timesteps_for_model = t.to(dtype=torch.float32)
    else:
        timesteps_for_model = t * float(num_train_timesteps)

    xt = corrupt(x0, t, sample_noise(x0))
    return PreparedBatch(
        latents=xt,
        timesteps=t,
        timesteps_for_model=timesteps_for_model,
        model=model,
        component_name=component_name,
        guidance_scale=0.0,
        use_cfg=False,
        cfg_batching=False,
        true_cfg_scale=None,
        pos_cond=pos_cond,
        neg_cond=None,
        joint_cond=None,
        advantage=advantage,
        extras={"x0": x0},
    )


def resolve_prepare_fn(args: Namespace) -> PrepareFn:
    path = getattr(args, "custom_prepare_train_batch_path", None)
    if path:
        fn = load_function(path)
        if fn is None:
            raise ValueError(f"Failed to load custom prepare from {path!r}")
        return fn
    return prepare_flow_grpo_batch
