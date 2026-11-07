"""PDD protocol shared by the Miles actor, loader and the SGLang rollout patch.

SGLang rolls out each 8-block trajectory with fixed student weights; the actor then
takes one optimizer update per block, in block order. Checkpoints sit at rollout
boundaries, so no trajectory is discarded on resume.
"""

import functools
import hashlib
from pathlib import Path

import torch

from .pdd import ParallelLinear, validate_grid


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    """Qwen-Image 2.1 output layer as N heads, emitted as [B, S, N*C] for the joint sequence."""

    def __init__(self, source, num_intervals):
        super().__init__()
        self.parallel = ParallelLinear(source, num_intervals)

    def forward(self, hidden):
        return self.parallel(hidden).permute(1, 2, 0, 3).flatten(2)


def install_student(model, args):
    if not isinstance(model.proj_out, torch.nn.Linear) or model.config.patch_size != 1:
        raise ValueError("PDD needs an unmodified Qwen-Image 2.1 transformer")
    if model.proj_out.out_features != model.config.in_channels:
        raise ValueError("PDD requires equal input and output latent channels")
    model.proj_out = TokenParallelProjection(model.proj_out, args.pdd_num_intervals)


def expand_trajectories(pairs, *, intervals, block_size):
    """Called AFTER DP splitting; each local window keeps the same ordered prompts."""
    if len({p["sample_index"] for p in pairs}) != len(pairs):
        raise ValueError("PDD needs a distinct lane ID per sampled prompt occurrence")
    return [dict(pair, pdd_start=start) for start in range(0, intervals, block_size) for pair in pairs]


def resume_signature(args):
    fields = (
        "seed",
        "hf_checkpoint",
        "teacher_load",
        "pdd_num_intervals",
        "pdd_block_size",
        "diffusion_height",
        "diffusion_width",
        "diffusion_forward_dtype",
        "fsdp_master_dtype",
        "fsdp_reduce_dtype",
        "micro_batch_size",
        "rollout_batch_size",
        "train_dp_split_mode",
        "sequence_parallel_size",
        "actor_num_nodes",
        "actor_num_gpus_per_node",
        "lr",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "clip_grad",
    )
    rollout = ("rollout_seed", "diffusion_num_steps", "diffusion_generator_device", "diffusion_quality")
    return {
        "version": 3,
        "boundary": "rollout_trajectory",
        "settings": {k: getattr(args, k) for k in fields + rollout},
        "prompt_sha256": file_sha256(args.prompt_data),
        "grid": make_grid(args.hf_checkpoint, args.pdd_num_intervals, (args.diffusion_height // 16) ** 2).tolist(),
    }


def configure_args(args):
    """Narrow, validated integration; unsupported modes fail before loading GPUs."""
    if args.diffusion_model_family != "qwen_image21" or args.diffusion_task != "t2i":
        raise ValueError("PDD supports Qwen-Image 2.1 T2I only")
    if args.train_only or args.train_backend != "fsdp":
        raise ValueError("PDD trains on FSDP from SGLang rollouts; drop --train-only")
    if args.update_weight_target_modules != ["transformer"]:
        raise ValueError("PDD trains only the transformer component")
    if (args.pdd_num_intervals, args.pdd_block_size) != (32, 4):
        raise ValueError("PDD uses 32 intervals and block size 4 (8 student forwards)")
    if args.diffusion_height != 1024 or args.diffusion_width != 1024:
        raise ValueError("PDD requires 1024 x 1024")
    if args.micro_batch_size != 1 or args.micro_batch_size_sample is not None:
        raise ValueError("PDD requires --micro-batch-size 1 and no legacy sample/timestep tiling")
    if args.sequence_parallel_size != 1:
        raise ValueError("PDD sequence parallelism is not yet validated; use FSDP with SP=1")
    if args.use_lora or args.use_ema or args.ref_load or args.ref_mode != "none":
        raise ValueError("PDD uses full student weights and a frozen teacher; no LoRA, EMA or reference model")
    if args.teacher_lora_adapter_paths or args.lora_adapter_paths:
        raise ValueError("PDD does not accept LoRA adapters")
    if args.diffusion_recompute_old_log_prob or args.diffusion_kl_beta:
        raise ValueError("PDD does not use RL log probabilities or KL")
    if args.diffusion_guidance_scale != 1 or args.diffusion_true_cfg_scale not in (None, 1):
        raise ValueError("PDD requires CFG=1")
    if not args.prompt_data or not args.rollout_global_dataset:
        raise ValueError("PDD requires --prompt-data and the global dataset")
    if args.n_samples_per_prompt != 1:
        raise ValueError("PDD requires --n-samples-per-prompt 1")
    if args.no_save_optim or args.no_load_optim or args.no_load_rng:
        raise ValueError("PDD resumption requires optimizer and RNG checkpoints")
    if args.fsdp_master_dtype != "fp32" or args.diffusion_forward_dtype != "bf16":
        raise ValueError("PDD uses FP32 master weights and BF16 forward")
    if args.teacher_load is None:
        args.teacher_load = args.hf_checkpoint
    if args.teacher_load != args.hf_checkpoint:
        raise ValueError("PDD teacher must be the same base model as --hf-checkpoint")
    # Rollout engine: the full 8-step ODE trajectory, unguided, one sample per request.
    if args.diffusion_num_steps != 8 or args.diffusion_eval_num_steps not in (None, 8):
        raise ValueError("PDD requires 8 training/evaluation steps")
    if args.diffusion_sde_type != "ode" or args.diffusion_noise_level != 0:
        raise ValueError("PDD requires --diffusion-sde-type ode --diffusion-noise-level 0")
    if args.rollout_num_gpus_per_engine != 1 or args.rollout_microgroup_size != 1:
        raise ValueError("PDD requires one GPU per engine and rollout microgroup size 1")
    if args.sglang_dit_precision != "fp32":
        raise ValueError("PDD requires --sglang-dit-precision fp32")
    if args.diffusion_step_strategy_path or args.custom_generate_function_path or args.group_rm:
        raise ValueError("PDD requires the standard full-trajectory rollout without custom generation/group RM")
    if args.rollout_patch_groups not in (None, [], ["pdd_qwen_image21"]):
        raise ValueError("PDD requires only the pdd_qwen_image21 rollout patch group")
    args.rollout_patch_groups = ["pdd_qwen_image21"]
    args.rollout_return_full_trajectory = True
    paths = {
        "rollout_function_path": "miles.rollout.sglang_diffusion_rollout.generate_rollout",
        "custom_convert_samples_to_train_data_path": "miles.rollout.pdd_rollout.convert_samples_to_train_data",
        "custom_rm_path": "miles.rollout.pdd_rollout.no_reward",
        "custom_prepare_train_batch_path": "miles.backends.fsdp_utils.loss_hub.pdd.prepare_pdd_batch",
        "custom_loss_function_path": "miles.backends.fsdp_utils.loss_hub.pdd.pdd_loss_formula",
    }
    for key, value in paths.items():
        old = getattr(args, key)
        if old not in (None, value, "miles.rollout.sglang_rollout.generate_rollout"):
            raise ValueError(f"PDD requires {key}={value}")
        setattr(args, key, value)
    if args.num_steps_per_rollout not in (None, 8):
        raise ValueError("PDD performs 8 optimizer updates per rollout trajectory")
    args.num_steps_per_rollout = 8
    if args.global_batch_size not in (None, args.rollout_batch_size):
        raise ValueError("PDD global batch size equals the number of trajectory lanes (rollout batch size)")
    args.global_batch_size = args.rollout_batch_size
