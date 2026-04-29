from __future__ import annotations

import logging
import json
import os
from argparse import Namespace
from typing import Any

import torch
import ray
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
import numpy as np

from flow_grpo import rewards as flow_rewards
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.stat_tracking import PerPromptStatTracker

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.diffusion_protocol import validate_rollout_metadata
from miles.utils import tracking_utils
from miles.utils.types import CondKwargs, DenoisingEnv, DiTTrajectory, Sample
from miles.backends.fsdp_utils.configs.train_pipeline_config import get_train_pipeline_config
import miles.backends.fsdp_utils.configs.sd3  # noqa: F401 — register SD3 train config
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

__all__ = ["generate_rollout"]
__all__.extend(["offload_rollout", "onload_rollout"])

logger = logging.getLogger(__name__)

_PIPELINE = None
_REWARD_FN = None
_LOGGED_ROLLOUT_IDS: set[int] = set()
_STAT_TRACKER: PerPromptStatTracker | None = None
_REWARD_SPEC = None
_ROLLOUT_PG = None
_ROLLOUT_WORKERS = None
_LAST_ROLLOUT_WEIGHT_VERSION = -1


def _get_rollout_weight_paths(args: Namespace) -> tuple[str, str]:
    base_dir = getattr(args, "save", None) or os.path.join("/tmp", "miles_rollout_weights")
    adapter_path = os.path.join(base_dir, "diffusion_lora_adapter.pt")
    full_path = os.path.join(base_dir, "diffusion_transformer.pt")
    weights_path = adapter_path if getattr(args, "use_lora", False) or os.path.exists(adapter_path) else full_path
    meta_path = os.path.join(base_dir, "diffusion_transformer.meta.json")
    return weights_path, meta_path


def _ensure_rollout_lora(args: Namespace, pipeline: StableDiffusion3Pipeline) -> None:
    if hasattr(pipeline.transformer, "peft_config"):
        return

    from peft import LoraConfig, get_peft_model

    train_pipeline_config = get_train_pipeline_config(args.diffusion_model)
    targets = getattr(args, "lora_target_modules", None) or train_pipeline_config.lora_target_modules
    pipeline.transformer = get_peft_model(
        pipeline.transformer,
        LoraConfig(
            r=getattr(args, "lora_rank", 64),
            lora_alpha=getattr(args, "lora_alpha", 64),
            target_modules=targets,
            init_lora_weights=True,
        ),
    )


def _maybe_load_rollout_weights(args: Namespace, pipeline: StableDiffusion3Pipeline) -> None:
    global _LAST_ROLLOUT_WEIGHT_VERSION

    weights_path, meta_path = _get_rollout_weight_paths(args)
    if not os.path.exists(meta_path) or not os.path.exists(weights_path):
        return

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        version = int(meta.get("version", -1))
        weight_type = str(meta.get("weight_type", "full_transformer"))
    except Exception:
        return

    if version <= _LAST_ROLLOUT_WEIGHT_VERSION:
        return

    if weight_type == "peft_lora":
        _ensure_rollout_lora(args, pipeline)
    state = torch.load(weights_path, map_location="cpu")
    try:
        dtype = next(pipeline.transformer.parameters()).dtype
    except StopIteration:
        dtype = None
    if dtype is not None:
        state = {k: v.to(dtype=dtype) if isinstance(v, torch.Tensor) else v for k, v in state.items()}

    if weight_type == "peft_lora":
        missing, unexpected = pipeline.transformer.load_state_dict(state, strict=False)
        unexpected_lora = [key for key in unexpected if "lora_" in key]
        missing_lora = [key for key in missing if "lora_" in key]
        if unexpected_lora or missing_lora:
            raise RuntimeError(
                f"LoRA adapter state mismatch: missing={missing_lora[:5]} unexpected={unexpected_lora[:5]}"
            )
    else:
        pipeline.transformer.load_state_dict(state, strict=True)
    _LAST_ROLLOUT_WEIGHT_VERSION = version
    logger.info("[rollout] loaded %s weights version=%s from %s", weight_type, version, weights_path)
    if getattr(args, "diffusion_debug_mode", False):
        print(f"[rollout] loaded {weight_type} weights version={version} from {weights_path}", flush=True)


def set_rollout_pg(pg) -> None:
    """Provide rollout placement group info for multi-GPU diffusion rollout."""
    global _ROLLOUT_PG
    _ROLLOUT_PG = pg


@ray.remote(num_gpus=1)
class DiffusionRolloutWorker:
    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run_group(self, rollout_id: int, group: list[Sample], evaluation: bool = False) -> list[Sample]:
        return _run_rollout_group(self.args, rollout_id, group, evaluation)


def _get_rollout_workers(args: Namespace):
    global _ROLLOUT_WORKERS
    if getattr(args, "rollout_num_gpus", 1) <= 1:
        return None
    if _ROLLOUT_WORKERS is not None:
        return _ROLLOUT_WORKERS

    num_workers = int(args.rollout_num_gpus)
    workers = []
    scheduling_strategy = None
    if _ROLLOUT_PG is not None:
        if isinstance(_ROLLOUT_PG, PlacementGroup):
            pg = _ROLLOUT_PG
            reordered_bundle_indices = list(range(num_workers))
        else:
            pg, reordered_bundle_indices, _ = _ROLLOUT_PG
        for i in range(num_workers):
            bundle_index = reordered_bundle_indices[i] if i < len(reordered_bundle_indices) else 0
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_index,
            )
            workers.append(
                DiffusionRolloutWorker.options(scheduling_strategy=scheduling_strategy).remote(args)
            )
    else:
        for _ in range(num_workers):
            workers.append(DiffusionRolloutWorker.remote(args))

    _ROLLOUT_WORKERS = workers
    return _ROLLOUT_WORKERS


def _get_device(args: Namespace) -> torch.device:
    if getattr(args, "diffusion_device", None):
        requested = str(args.diffusion_device)
        if requested == "cuda":
            return torch.device("cuda:0")
        return torch.device(requested)
    if torch.cuda.is_available():
        # Ensure an explicit device index for torch.cuda.set_device in Ray actors.
        return torch.device("cuda:0")
    return torch.device("cpu")


def _get_dtype(args: Namespace) -> torch.dtype:
    dtype = getattr(args, "diffusion_dtype", "fp16")
    if dtype == "fp32":
        return torch.float32
    if dtype == "bf16":
        return torch.bfloat16
    return torch.float16


def _get_pipeline(args: Namespace) -> StableDiffusion3Pipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    # Load SD3 pipeline once and reuse across rollout calls.
    model_id = getattr(args, "diffusion_model", "stabilityai/stable-diffusion-3.5-medium")
    dtype = _get_dtype(args)
    device = _get_device(args)
    _PIPELINE = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    _PIPELINE.to(device)
    return _PIPELINE


def offload_rollout(args: Namespace) -> None:
    """Move diffusion rollout pipeline to CPU to free GPU memory."""
    global _PIPELINE
    if _PIPELINE is None:
        return
    _PIPELINE.to("cpu")


def onload_rollout(args: Namespace) -> None:
    """Move diffusion rollout pipeline back to the target device."""
    global _PIPELINE
    if _PIPELINE is None:
        return
    _PIPELINE.to(_get_device(args))


def _get_reward_fn(args: Namespace):
    global _REWARD_FN
    global _REWARD_SPEC
    if _REWARD_FN is not None:
        return _REWARD_FN

    reward_spec = _parse_reward_spec(args)
    _REWARD_SPEC = reward_spec
    device = getattr(args, "diffusion_reward_device", None) or str(_get_device(args))
    _REWARD_FN = flow_rewards.multi_score(device, reward_spec)
    return _REWARD_FN


def _parse_reward_spec(args: Namespace) -> dict[str, float]:
    spec = getattr(args, "diffusion_reward", "pickscore")
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, str):
        text = spec.strip()
        if text.startswith("{"):
            return json.loads(text)
        if ":" in text:
            pairs = [p for p in text.split(",") if p]
            reward_dict: dict[str, float] = {}
            for pair in pairs:
                name, weight = pair.split(":")
                reward_dict[name.strip()] = float(weight)
            return reward_dict
        return {text: 1.0}
    return {"pickscore": 1.0}


def _make_generators(prompts: list[str], base_seed: int, seed_offset: int) -> list[torch.Generator]:
    generators = []
    for idx, prompt in enumerate(prompts):
        del prompt
        # Per-sample generator ensures diverse images within the same prompt group.
        seed = (base_seed + seed_offset + idx) % (2**31)
        generator = torch.Generator().manual_seed(seed)
        generators.append(generator)
    return generators


def _fill_sample_metadata(
    samples: list[Sample],
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    latents: torch.Tensor,
    next_latents: torch.Tensor,
    log_prob_old: torch.Tensor,
    prev_latents_mean: torch.Tensor | None,
    prompt_embeds_pos: torch.Tensor | None = None,
    pooled_embeds_pos: torch.Tensor | None = None,
    prompt_embeds_neg: torch.Tensor | None = None,
    pooled_embeds_neg: torch.Tensor | None = None,
) -> None:
    # Move large rollout tensors to CPU and store per-sample slices in metadata.
    timesteps_cpu = timesteps.cpu()
    sigmas_cpu = sigmas.cpu()
    latents_cpu = latents.cpu()
    next_latents_cpu = next_latents.cpu()
    log_prob_old_cpu = log_prob_old.cpu()
    prev_latents_mean_cpu = prev_latents_mean.cpu() if prev_latents_mean is not None else None

    for i, sample in enumerate(samples):
        metadata = {
            "timesteps": timesteps_cpu.clone(),
            "sigmas": sigmas_cpu.clone(),
            "latents": latents_cpu[i].clone(),
            "next_latents": next_latents_cpu[i].clone(),
            "log_prob_old": log_prob_old_cpu[i].clone(),
        }
        if prev_latents_mean_cpu is not None:
            metadata["prev_latents_mean"] = prev_latents_mean_cpu[i].clone()
        sample.metadata.update(metadata)
        sample.train_metadata = metadata

        all_latents = torch.cat([latents_cpu[i], next_latents_cpu[i, -1:]], dim=0)
        sample.dit_trajectory = DiTTrajectory(
            latents=all_latents,
            timesteps=timesteps_cpu.clone(),
        )
        sample.rollout_log_probs = log_prob_old_cpu[i].clone()

        if prompt_embeds_pos is not None:
            pos_cond = CondKwargs(
                encoder_hidden_states=[prompt_embeds_pos[i].detach().cpu()],
                pooled_projections=[pooled_embeds_pos[i].detach().cpu()]
                if pooled_embeds_pos is not None
                else None,
            )
            neg_cond = None
            if prompt_embeds_neg is not None:
                neg_cond = CondKwargs(
                    encoder_hidden_states=[prompt_embeds_neg[i].detach().cpu()],
                    pooled_projections=[pooled_embeds_neg[i].detach().cpu()]
                    if pooled_embeds_neg is not None
                    else None,
                )
            sample.denoising_env = DenoisingEnv(
                pos_cond_kwargs=pos_cond,
                neg_cond_kwargs=neg_cond,
            )

        # Sanity check required keys and shape alignment.
        errors = validate_rollout_metadata(sample.metadata)
        if errors:
            raise ValueError(f"Invalid diffusion rollout metadata: {errors}")


def _get_stat_tracker(args: Namespace) -> PerPromptStatTracker:
    global _STAT_TRACKER
    if _STAT_TRACKER is None:
        _STAT_TRACKER = PerPromptStatTracker(global_std=False)
    return _STAT_TRACKER


def _calculate_zero_std_ratio(prompts: list[str], rewards: list[float]) -> tuple[float, float]:
    prompt_array = np.array(prompts)
    rewards_array = np.array(rewards, dtype=np.float64)
    unique, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    if len(unique) == 0:
        return 0.0, 0.0
    grouped_rewards = rewards_array[np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return float(zero_std_ratio), float(prompt_std_devs.mean())

def _log_wandb_images_if_enabled(
    args: Namespace,
    rollout_id: int,
    group: list[Sample],
    images: list,
    rewards: list[float],
) -> None:
    if not getattr(args, "use_wandb", False):
        return
    log_n = int(getattr(args, "diffusion_log_images", 0) or 0)
    if log_n <= 0:
        return
    interval = int(getattr(args, "diffusion_log_image_interval", 1) or 1)
    if interval <= 0:
        interval = 1
    if rollout_id % interval != 0:
        return
    group_index = getattr(group[0], "group_index", None)
    if group_index is not None and group_index != 0:
        return
    if group_index is None:
        if rollout_id in _LOGGED_ROLLOUT_IDS:
            return
        _LOGGED_ROLLOUT_IDS.add(rollout_id)

    import wandb
    if wandb.run is None:
        return

    log_images = []
    for idx, (img, sample, reward) in enumerate(zip(images, group, rewards, strict=False)):
        if idx >= log_n:
            break
        caption = f"reward={reward:.4f} prompt={sample.prompt}"
        log_images.append(wandb.Image(img, caption=caption))

    if not log_images:
        return

    metrics = {
        "images": log_images,
        "rollout/step": compute_rollout_step(args, rollout_id),
    }
    wandb.log(metrics)


def _run_rollout_group(
    args: Namespace, rollout_id: int, group: list[Sample], evaluation: bool
) -> list[Sample]:
    device = _get_device(args)
    pipeline = _get_pipeline(args)
    _maybe_load_rollout_weights(args, pipeline)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Each group is multiple samples of the same prompt.
    prompts = [sample.prompt for sample in group]
    num_steps = getattr(args, "diffusion_num_steps", 10)
    if evaluation and getattr(args, "diffusion_eval_num_steps", None) is not None:
        num_steps = args.diffusion_eval_num_steps

    # Deterministic seeding per rollout/group/sample.
    seed_offset = getattr(group[0], "group_index", 0) or 0
    seed_offset += rollout_id * 1000
    generators = _make_generators(prompts, getattr(args, "rollout_seed", 0), seed_offset)
    guidance_scale = getattr(args, "diffusion_guidance_scale", 4.5)
    noise_level = getattr(
        args,
        "diffusion_rollout_noise_level",
        getattr(args, "diffusion_noise_level", 0.7),
    )
    height = getattr(args, "diffusion_height", 512)
    width = getattr(args, "diffusion_width", 512)
    return_prev_latents_mean = getattr(args, "diffusion_return_prev_latents_mean", False)
    do_cfg = guidance_scale > 1.0
    negative_prompts = [""] * len(prompts) if do_cfg else None

    with torch.no_grad():
        prompt_embeds_pos, prompt_embeds_neg, pooled_embeds_pos, pooled_embeds_neg = pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompts,
            do_classifier_free_guidance=do_cfg,
            device=device,
        )

    # Run patched pipeline to get images + trajectory + per-step log-prob.
    output = pipeline_with_logprob(
        pipeline,
        prompt_embeds=prompt_embeds_pos,
        pooled_prompt_embeds=pooled_embeds_pos,
        negative_prompt_embeds=prompt_embeds_neg,
        negative_pooled_prompt_embeds=pooled_embeds_neg,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generators,
        output_type="pil",
        noise_level=noise_level,
        return_prev_sample_mean=return_prev_latents_mean,
    )

    if return_prev_latents_mean:
        images, all_latents, all_log_probs, all_prev_latents_mean = output
    else:
        images, all_latents, all_log_probs = output
        all_prev_latents_mean = None

    # Reconstruct timesteps from scheduler so training can recompute log_prob_new.
    timesteps, _ = retrieve_timesteps(pipeline.scheduler, num_steps, torch.device("cpu"))
    sigmas = getattr(pipeline.scheduler, "sigmas", None)
    if sigmas is None:
        raise ValueError("Scheduler missing sigmas; cannot align diffusion rollout metadata.")
    sigmas = torch.as_tensor(sigmas)
    if not getattr(_run_rollout_group, "_logged_sigmas_len", False):
        print(
            f"[rollout] timesteps_len={int(timesteps.numel())} sigmas_len={int(sigmas.numel())}",
            flush=True,
        )
        _run_rollout_group._logged_sigmas_len = True

    # Convert list trajectories into (B, T, C, H, W) tensors.
    latents = torch.stack(all_latents[:-1], dim=1)
    next_latents = torch.stack(all_latents[1:], dim=1)
    log_prob_old = torch.stack(all_log_probs, dim=1)
    prev_latents_mean = None
    if all_prev_latents_mean is not None:
        prev_latents_mean = torch.stack(all_prev_latents_mean, dim=1)

    _fill_sample_metadata(
        group,
        timesteps=timesteps,
        sigmas=sigmas,
        latents=latents,
        next_latents=next_latents,
        log_prob_old=log_prob_old,
        prev_latents_mean=prev_latents_mean,
        prompt_embeds_pos=prompt_embeds_pos,
        pooled_embeds_pos=pooled_embeds_pos,
        prompt_embeds_neg=prompt_embeds_neg,
        pooled_embeds_neg=pooled_embeds_neg,
    )

    reward_fn = _get_reward_fn(args)
    raw_reward_dict, _ = reward_fn(images, prompts, [{} for _ in range(len(prompts))])
    reward_dict = {}
    for key, value in raw_reward_dict.items():
        if torch.is_tensor(value):
            reward_dict[key] = value.detach().cpu().numpy().astype(np.float64).tolist()
        elif isinstance(value, np.ndarray):
            reward_dict[key] = value.astype(np.float64).tolist()
        else:
            reward_dict[key] = [
                float(item.item()) if torch.is_tensor(item) else float(item)
                for item in value
            ]
    rewards_avg = reward_dict.get("avg", reward_dict.get("ocr", []))

    _log_wandb_images_if_enabled(args, rollout_id, group, images, rewards_avg)

    for idx, sample in enumerate(group):
        per_sample_reward = {k: float(v[idx]) for k, v in reward_dict.items()}
        sample.reward = per_sample_reward
        sample.status = Sample.Status.COMPLETED

    return group


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    assert args.rollout_global_dataset

    num_batches = getattr(args, "diffusion_num_batches_per_epoch", 1)
    if num_batches is None or num_batches <= 0:
        num_batches = 1

    output_groups = []
    workers = _get_rollout_workers(args)
    for _ in range(num_batches):
        groups = data_source.get_samples(args.rollout_batch_size)
        if workers:
            tasks = []
            for idx, group in enumerate(groups):
                worker = workers[idx % len(workers)]
                tasks.append(worker.run_group.remote(rollout_id, group, evaluation))
            output_groups.extend(ray.get(tasks))
        else:
            for group in groups:
                output_groups.append(_run_rollout_group(args, rollout_id, group, evaluation=evaluation))

    flat = [sample for group in output_groups for sample in group]
    prompts = [sample.prompt for sample in flat]
    rewards_avg = [sample.reward.get("avg", sample.reward.get("ocr", 0.0)) for sample in flat]
    reward_ocr = [sample.reward.get("ocr", None) for sample in flat]

    log_dict = {
        "reward_avg": float(np.mean(rewards_avg)) if rewards_avg else 0.0,
        "rollout/step": compute_rollout_step(args, rollout_id),
    }
    if any(r is not None for r in reward_ocr):
        reward_ocr_vals = [r for r in reward_ocr if r is not None]
        if reward_ocr_vals:
            log_dict["reward_ocr"] = float(np.mean(reward_ocr_vals))

    # Avoid logging rollout metrics to W&B so all curves align on training global_step.
    if not getattr(args, "use_wandb", False):
        tracking_utils.log(args, log_dict, step_key="rollout/step")

    if evaluation:
        return RolloutFnEvalOutput(
            data={
                "diffusion_eval": {
                    "rewards": [sample.reward for sample in flat],
                    "truncated": [sample.status == Sample.Status.TRUNCATED for sample in flat],
                    "samples": flat,
                }
            }
        )

    return RolloutFnTrainOutput(samples=output_groups)
