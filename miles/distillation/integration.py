"""PDD protocol shared by the Miles actor, loader and the SGLang rollout patches.

SGLang rolls out each 8-block trajectory with fixed student weights; the actor then
takes one optimizer update per block, in block order. ``set_pdd_defaults`` wires the
generic Miles hooks; nothing in the trainer branches on ``--loss-type pdd``. What
differs per model family (latent streams, sigma grids, time inputs, rollout patch)
lives in ``PDD_FAMILIES``.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from .pdd import ParallelLinear, validate_grid


def make_grid(model_path, intervals, tokens, device="cpu"):
    return _cached_grid(str(model_path), intervals, tokens).to(device, copy=True)


@functools.lru_cache(maxsize=8)
def _cached_grid(model_path, intervals, tokens):
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler", local_files_only=True
    )
    cfg = scheduler.config
    mu = calculate_shift(tokens, cfg.base_image_seq_len, cfg.max_image_seq_len, cfg.base_shift, cfg.max_shift)
    scheduler.set_timesteps(intervals, sigmas=np.linspace(1.0, 1.0 / intervals, intervals), mu=mu)
    grid = scheduler.sigmas.float()
    validate_grid(grid)
    return grid


def engine_timesteps(sigmas):
    """SGLang timesteps whose BF16 time-embedding input equals the trainer's.

    Under BF16 autocast the native DiT embeds ``bf16(bf16(t) / 1000)``, while the
    Diffusers trainer embeds ``bf16(sigma)``; ``t = 1000 * sigma`` can round to a
    different value. Pick the representable ``t`` nearest ``1000 * sigma`` that
    lands on the trainer's value, or the nearest reachable value when none does;
    the online trainer then feeds ``embedded_sigmas`` of that choice.
    """
    result = []
    for sigma in sigmas.detach().float().cpu():
        target = float(sigma.bfloat16())
        center = float(sigma) * 1000
        candidates = torch.arange(center - 16, center + 16, 1 / 16).bfloat16().unique().float()
        miss = ((candidates.bfloat16() / 1000).float() - target).abs()
        # Lexicographic: embedding mismatch first, then distance from 1000 * sigma.
        best = torch.argmin(miss * 1e6 + (candidates - center).abs())
        result.append(candidates[best])
    return torch.stack(result)


def embedded_sigmas(timesteps):
    """The sigma the native SGLang DiT embeds for ``timesteps`` under BF16 autocast."""
    return (timesteps.bfloat16() / 1000).float()


class TokenParallelProjection(torch.nn.Module):
    """An output layer as N heads, emitted as [B, S, N*C] so token selection and masks stay row-wise."""

    def __init__(self, source, num_intervals):
        super().__init__()
        self.parallel = ParallelLinear(source, num_intervals)

    def forward(self, hidden):
        return self.parallel(hidden).permute(1, 2, 0, 3).flatten(2)


@dataclass(frozen=True)
class PDDFamily:
    """What PDD needs from one model family, looked up by ``--diffusion-model-family``.

    ``heads`` maps each latent stream to the Diffusers transformer's output layer that becomes N heads;
    ``grids(args)`` gives each stream's (N+1)-point sigma grid; ``model_time`` / ``teacher_time`` turn a grid
    sigma into the student's and the teacher's ``timesteps_for_model``; ``split_states`` cuts the rollout's
    per-interval states into streams; ``validate`` checks the family's rollout settings.
    """

    task: str
    heads: dict[str, str]
    patch_group: str
    engine_precision: str
    # None keeps the family's FSDP plan (e.g. FP32 heads over a BF16 all-gather).
    fsdp_param_dtype: str | None
    grids: Callable[[Any], dict[str, torch.Tensor]]
    model_time: Callable[[Any, str, torch.Tensor], torch.Tensor]
    teacher_time: Callable[[Any, str, torch.Tensor], torch.Tensor]
    split_states: Callable[[torch.Tensor, Any], dict[str, torch.Tensor]]
    validate: Callable[[Any], None]
    cond_extras: dict[str, Any] = field(default_factory=dict)


def _qwen_validate(args):
    if args.diffusion_height != 1024 or args.diffusion_width != 1024:
        raise ValueError("PDD requires 1024 x 1024 for Qwen-Image 2.1")


PDD_FAMILIES = {
    "qwen_image21": PDDFamily(
        task="t2i",
        heads={"visual": "proj_out"},
        patch_group="pdd_qwen_image21",
        engine_precision="fp32",
        # Gathered student weights stay FP32 (head biases, norms); autocast handles matmuls.
        fsdp_param_dtype="fp32",
        grids=lambda args: {
            "visual": make_grid(args.hf_checkpoint, args.pdd_num_intervals, (args.diffusion_height // 16) ** 2)
        },
        # The student sees the time value SGLang embedded; the grid dynamics stay unchanged.
        model_time=lambda args, stream, sigma: embedded_sigmas(engine_timesteps(sigma)),
        teacher_time=lambda args, stream, sigma: sigma,
        split_states=lambda states, env: {"visual": states},
        validate=_qwen_validate,
    ),
}


def pdd_family(args):
    try:
        return PDD_FAMILIES[args.diffusion_model_family]
    except KeyError:
        raise ValueError(f"PDD supports --diffusion-model-family {sorted(PDD_FAMILIES)}") from None


def install_student(model, args):
    """``--custom-train-model-transform-path`` hook: each stream's output layer becomes N heads, trained in
    full beside the backbone's LoRA adapters."""
    for attr in pdd_family(args).heads.values():
        layer = getattr(model, attr)
        if not isinstance(layer, torch.nn.Linear):
            raise ValueError(f"PDD needs an unmodified {attr} output layer")
        setattr(model, attr, TokenParallelProjection(layer, args.pdd_num_intervals))


def expand_trajectories(pairs, *, intervals, block_size):
    """Called AFTER DP splitting; each local window keeps the same ordered prompts."""
    if len({p["sample_index"] for p in pairs}) != len(pairs):
        raise ValueError("PDD needs a distinct lane ID per sampled prompt occurrence")
    return [dict(pair, pdd_start=start) for start in range(0, intervals, block_size) for pair in pairs]


def expand_train_pairs(ctx, train_pairs):
    """PDD's train-input hook: one block-ordered update per trajectory block."""
    return expand_trajectories(train_pairs, intervals=ctx.args.pdd_num_intervals, block_size=ctx.args.pdd_block_size)


_HOOK_DEFAULTS = {
    "custom_convert_samples_to_train_data_path": "miles.rollout.pdd_rollout.convert_samples_to_train_data",
    "custom_rm_path": "miles.rollout.pdd_rollout.no_reward",
    "custom_train_model_transform_path": "miles.distillation.integration.install_student",
}
_DIFFUSION_ROLLOUT = "miles.rollout.sglang_diffusion_rollout.generate_rollout"


def set_pdd_defaults(args):
    """Fill PDD's rollout/loader hooks and schedule (the loss hub supplies its train hooks);
    ``validate_pdd_args`` rejects explicit conflicting values."""
    for key, value in _HOOK_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.rollout_function_path == "miles.rollout.sglang_rollout.generate_rollout":
        args.rollout_function_path = _DIFFUSION_ROLLOUT
    family = PDD_FAMILIES.get(args.diffusion_model_family)
    if family is not None:
        if args.rollout_patch_group is None:
            args.rollout_patch_group = family.patch_group
        if args.fsdp_param_dtype is None:
            args.fsdp_param_dtype = family.fsdp_param_dtype
    if args.teacher_load is None:
        args.teacher_load = args.hf_checkpoint
    args.rollout_return_full_trajectory = True
    # One optimizer update per block; each update sees every trajectory lane once.
    if args.num_steps_per_rollout is None:
        args.num_steps_per_rollout = args.pdd_num_intervals // args.pdd_block_size
    if args.global_batch_size is None:
        args.global_batch_size = args.rollout_batch_size


def validate_pdd_args(args):
    """Narrow, validated integration; unsupported modes fail before loading GPUs."""
    family = pdd_family(args)
    if args.diffusion_task != family.task:
        raise ValueError(f"PDD supports {args.diffusion_model_family} {family.task} only")
    family.validate(args)
    if args.train_only or args.train_backend != "fsdp":
        raise ValueError("PDD trains on FSDP from SGLang rollouts; drop --train-only")
    if args.update_weight_target_modules != ["transformer"]:
        raise ValueError("PDD trains only the transformer component")
    if (args.pdd_num_intervals, args.pdd_block_size) != (32, 4):
        raise ValueError("PDD uses 32 intervals and block size 4 (8 student forwards)")
    if args.micro_batch_size != 1 or args.micro_batch_size_sample is not None:
        raise ValueError("PDD requires --micro-batch-size 1 and no legacy sample/timestep tiling")
    if args.sequence_parallel_size != 1:
        raise ValueError("PDD sequence parallelism is not yet validated; use FSDP with SP=1")
    # The student is LoRA on the backbone plus its new heads, trained in full; the teacher is the frozen base.
    if not args.use_lora:
        raise ValueError("PDD trains a LoRA student (plus full heads); pass --use-lora")
    # The engine applies the adapter unmerged like the trainer: merged BF16 weights would round most of the
    # adapter's update away. The heads go as plain weights after the adapter.
    if args.colocate and not args.lora_ipc_weight_sync:
        raise ValueError("PDD syncs the LoRA adapter itself; pass --lora-ipc-weight-sync")
    if getattr(args, "sglang_lora_merge_mode", None) != "dynamic":
        raise ValueError("PDD needs the engine's unmerged LoRA forward; pass --sglang-lora-merge-mode dynamic")
    if args.use_ema or args.ref_load or args.ref_mode != "none":
        raise ValueError("PDD uses a frozen teacher; no EMA or reference model")
    if args.teacher_lora_adapter_paths or args.lora_adapter_paths:
        raise ValueError("PDD starts the student from the base model; no initial or teacher LoRA adapters")
    if args.diffusion_recompute_old_log_prob or args.diffusion_kl_beta:
        raise ValueError("PDD does not use RL log probabilities or KL")
    if args.diffusion_guidance_scale != 1 or args.diffusion_true_cfg_scale not in (None, 1):
        raise ValueError("PDD requires CFG=1")
    if not args.prompt_data or not args.rollout_global_dataset:
        raise ValueError("PDD requires --prompt-data and the global dataset")
    if args.n_samples_per_prompt != 1:
        raise ValueError("PDD requires --n-samples-per-prompt 1")
    if args.fsdp_master_dtype != "fp32" or args.diffusion_forward_dtype != "bf16":
        raise ValueError("PDD uses FP32 master weights and BF16 forward")
    if args.teacher_load != args.hf_checkpoint:
        raise ValueError("PDD teacher must be the same base model as --hf-checkpoint")
    # Rollout engine: the full 8-step ODE trajectory, unguided, one sample per request.
    if args.diffusion_num_steps != 8 or args.diffusion_eval_num_steps not in (None, 8):
        raise ValueError("PDD requires 8 training/evaluation steps")
    if args.diffusion_sde_type != "ode" or args.diffusion_noise_level != 0:
        raise ValueError("PDD requires --diffusion-sde-type ode --diffusion-noise-level 0")
    if args.rollout_num_gpus_per_engine != 1 or args.rollout_microgroup_size != 1:
        raise ValueError("PDD requires one GPU per engine and rollout microgroup size 1")
    if args.sglang_dit_precision != family.engine_precision:
        raise ValueError(f"PDD requires --sglang-dit-precision {family.engine_precision}")
    if args.fsdp_param_dtype != family.fsdp_param_dtype:
        raise ValueError(f"PDD requires --fsdp-param-dtype {family.fsdp_param_dtype}")
    if args.diffusion_step_strategy_path or args.custom_generate_function_path or args.group_rm:
        raise ValueError("PDD requires the standard full-trajectory rollout without custom generation/group RM")
    if args.rollout_patch_groups != [family.patch_group]:
        raise ValueError(f"PDD requires only the {family.patch_group} rollout patch group")
    for key, value in {**_HOOK_DEFAULTS, "rollout_function_path": _DIFFUSION_ROLLOUT}.items():
        if getattr(args, key) != value:
            raise ValueError(f"PDD requires {key}={value}")
    if args.num_steps_per_rollout != args.pdd_num_intervals // args.pdd_block_size:
        raise ValueError("PDD performs one optimizer update per block (8 per rollout)")
    if args.global_batch_size != args.rollout_batch_size:
        raise ValueError("PDD global batch size equals the number of trajectory lanes (rollout batch size)")
