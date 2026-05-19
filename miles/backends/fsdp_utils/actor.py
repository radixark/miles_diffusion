import logging
from argparse import Namespace
from collections import defaultdict

import ray
import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline

from miles.ray.train_actor import TrainRayActor
from miles.utils.context_utils import with_defer
from miles.utils import train_metric_utils
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.sde_log_prob import sde_step_with_logprob
from miles.utils.train_data_utils import scheduler_meta_from_rollout, stack_train_pair_rollout_debug
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils import tracking_utils
from miles.utils.profile_utils import TrainProfiler

from .configs.train_pipeline_config import get_train_pipeline_config
import miles.backends.fsdp_utils.configs.qwen_image  # noqa: F401 — register pipeline config

from . import checkpoint
from .lr_scheduler import get_lr_scheduler
from .parallel import create_fsdp_parallel_state
from .diffusion_update_weight_utils import DiffusionUpdateWeightFromTensor, DiffusionUpdateWeightFromTensorLoRA

logger = logging.getLogger(__name__)


def build_microbatch_schedule(
    *,
    num_pairs_per_optim_step: int,
    num_optim_steps_per_rollout: int,
    micro_batch_size: int,
) -> list[list[tuple[int, int]]]:
    """Absolute train-pair ranges for every optimizer step and micro-batch."""
    schedule: list[list[tuple[int, int]]] = []
    for step_id in range(num_optim_steps_per_rollout):
        step_pair_lo = step_id * num_pairs_per_optim_step
        step_pair_hi = step_pair_lo + num_pairs_per_optim_step
        step_ranges = []
        for pair_lo in range(step_pair_lo, step_pair_hi, micro_batch_size):
            pair_hi = min(step_pair_hi, pair_lo + micro_batch_size)
            step_ranges.append((pair_lo, pair_hi))
        schedule.append(step_ranges)
    return schedule


def validate_same_microbatch_counts_across_dp(
    *,
    microbatch_schedule: list[list[tuple[int, int]]],
    parallel_state,
) -> None:
    """Ensure every DP rank will run the same number of FSDP micro-batches."""
    local_microbatch_counts = [len(step_ranges) for step_ranges in microbatch_schedule]
    gathered_microbatch_counts = [None] * parallel_state.dp_cp_size
    dist.all_gather_object(
        gathered_microbatch_counts,
        local_microbatch_counts,
        group=parallel_state.dp_cp_group_gloo,
    )
    if any(counts != local_microbatch_counts for counts in gathered_microbatch_counts):
        raise ValueError(
            "Uneven train-pair counts would make DP ranks run different numbers of FSDP "
            f"micro-batches per optimizer step: {gathered_microbatch_counts}"
        )


class FSDPTrainRayActor(TrainRayActor):
    """FSDP training actor for diffusion GRPO.

    Loads only the DiT (transformer) from a diffusers pipeline, wraps it with
    FSDP, and trains with a PPO-clipped objective aligned with flow GRPO.
    """

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False) -> int:  # type: ignore[override]
        super().init(args, role, with_ref)

        self.parallel_state = create_fsdp_parallel_state(args)
        torch.manual_seed(args.seed)

        self.train_parallel_config = {
            "dp_size": self.parallel_state.dp_size,
        }

        if self.args.debug_rollout_only:
            return 0

        if self.args.offload_train and self.args.fsdp_cpu_offload:
            self.args.offload_train = False

        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        if self.args.start_rollout_id is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)

        self._master_dtype = _resolve_dtype(args.fsdp_master_dtype)
        self._forward_dtype = _resolve_dtype(args.diffusion_forward_dtype)

        with self._get_init_weight_context_manager():
            pipeline = DiffusionPipeline.from_pretrained(
                self.args.hf_checkpoint,
                torch_dtype=self._master_dtype,
                trust_remote_code=True,
                text_encoder=None,
                vae=None,
                tokenizer=None,
            )
            model = pipeline.transformer
            self.scheduler = pipeline.scheduler
            del pipeline

        self.train_pipeline_config = get_train_pipeline_config(args.diffusion_model)

        if args.use_lora:
            model = apply_lora(model, args, self.train_pipeline_config)

        model.train()

        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        model.to(torch.cuda.current_device())

        self.train_pipeline_config.preprocess_model_before_fsdp(model)

        model = apply_fsdp2(
            model,
            mesh=self.parallel_state.dp_mesh,
            cpu_offload=self.args.fsdp_cpu_offload,
            args=self.args,
        )
        # Force a sync to ensure sharding is complete and old memory is freed.
        torch.cuda.synchronize()
        clear_memory()
        self.model = model

        if args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                (p for p in self.model.parameters() if p.requires_grad),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)
        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)

        # sglang-d now supports /update_weights_from_tensor (PR #20464).
        update_weight_target_module = self.train_pipeline_config.update_weight_target_module
        self.weight_updater = (
            DiffusionUpdateWeightFromTensorLoRA(self.args, self.model, update_weight_target_module)
            if self.args.use_lora
            else DiffusionUpdateWeightFromTensor(self.args, self.model, update_weight_target_module)
        )

        checkpoint.finalize_load(self, checkpoint_payload)

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()

        return self.args.start_rollout_id

    def _get_parallel_config(self) -> dict:
        return {"dp_size": getattr(self.parallel_state, "dp_size", 1)}

    @timer
    def sleep(self) -> None:
        if not self.args.offload_train:
            return

        print_memory("before offload DiT")

        self.model.cpu()
        move_torch_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after sleep DiT")

    @timer
    def wake_up(self) -> None:
        if not self.args.offload_train:
            return

        self.model.cuda()
        move_torch_optimizer(self.optimizer, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up DiT")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:  # type: ignore[override]
        if self.args.save is None:
            return
        checkpoint.save(self, iteration=rollout_id)

    @timer
    def update_weights(self) -> None:  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.weight_updater is None:
            dist.barrier(group=get_gloo_group())
            return

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

        self.weight_updater.update_weights()
        clear_memory()
    
    def _get_init_weight_context_manager(self):
        """Return a context manager for model initialization.

        Non-rank-0 ranks use accelerate's ``init_empty_weights`` (params on
        meta device, no allocation). Rank 0 uses ``torch.device("cpu")``
        (already a context manager since PyTorch 1.X — sets default device
        for tensor construction inside the block).
        """
        from accelerate import init_empty_weights

        if dist.get_rank() != 0:
            return init_empty_weights()
        return torch.device("cpu")

    def _gather_and_log_metrics(self, rollout_id: int, log_dict: dict[str, float], step: int) -> None:
        """Reduce per-rank scalars and log."""
        if "train/lr" not in log_dict and hasattr(self, "optimizer"):
            try:
                log_dict["train/lr"] = float(self.optimizer.param_groups[0]["lr"])
            except Exception:
                pass
        if self.parallel_state.dp_cp_rank == 0:
            dp_size = self.parallel_state.dp_cp_size
            gathered = [None] * dp_size
            dist.gather_object(
                log_dict,
                gathered,
                dst=self.parallel_state.dp_src_rank,
                group=self.parallel_state.dp_cp_group_gloo,
            )
            reduced = {k: sum(d[k] for d in gathered) / dp_size for k in log_dict}
            reduced["train/epoch"] = float(rollout_id)
            reduced["rollout/step"] = compute_rollout_step(self.args, rollout_id)
            reduced["train/step"] = float(step)
            tracking_utils.log(self.args, reduced, step_key="train/step")

            logger.info(
                f"[train step {int(step)}] rollout={rollout_id} "
                + " ".join(f"{k}={v:.6e}" for k, v in sorted(reduced.items()) if k not in ("train/epoch", "rollout/step", "train/step"))
            )
        else:
            dist.gather_object(
                log_dict,
                None,
                dst=self.parallel_state.dp_src_rank,
                group=self.parallel_state.dp_cp_group_gloo,
            )

    def train(self, rollout_id: int, rollout_data_ref) -> None:  # type: ignore[override]
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            rollout_data = ray.get(rollout_data_ref[self.parallel_state.dp_rank].inner)
            if self.args.debug_rollout_only:
                return
            self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)
        
        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
        )

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        """Diffusion GRPO: ``rollout_data[train_data]`` is a flat list of train-pair dicts.

        Optimizer windows are contiguous groups of train pairs. Within a window, consecutive microbatches of
        size ``--micro-batch-size`` drive one forward+backward each; gradients
        scale as mean over all train pairs in the window (``loss_chunk / W``).
        """
        device = torch.cuda.current_device()

        train_pairs: list = rollout_data["train_data"]
        if not train_pairs:
            raise ValueError("rollout_data['train_data'] is empty")

        num_pairs = len(train_pairs)

        # ------------- CFG Scale -------------
        guidance_scale = self.args.diffusion_guidance_scale
        true_cfg_scale = self.args.diffusion_true_cfg_scale
        cfg_scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        use_cfg = cfg_scale > 0

        # ------------- Loss / SDE Parameters -------------
        clip_range = self.args.diffusion_clip_range
        noise_level = self.args.diffusion_noise_level
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        # ------------- Rollout Scheduler Metadata -------------
        scheduler_timesteps, scheduler_sigmas = scheduler_meta_from_rollout(
            rollout_data,
            device=device,
            num_train_timesteps=num_train_timesteps,
        )
        self.scheduler.timesteps = scheduler_timesteps
        self.scheduler.sigmas = scheduler_sigmas
        self.scheduler._step_index = None
        self.scheduler._begin_index = None

        # ------------- Optimizer Windows -------------
        num_optim_steps_per_rollout = self.args.num_steps_per_rollout
        if num_pairs % num_optim_steps_per_rollout != 0:
            raise ValueError(
                f"num_pairs_shard={num_pairs} not divisible by "
                f"num_steps_per_rollout={num_optim_steps_per_rollout}"
            )
        num_pairs_per_optim_step = num_pairs // num_optim_steps_per_rollout

        # ------------- Microbatch Synchronization -------------
        micro_bs = self.args.micro_batch_size
        if micro_bs <= 0:
            raise ValueError(f"micro_batch_size must be positive, got {micro_bs}")
        microbatch_schedule = build_microbatch_schedule(
            num_pairs_per_optim_step=num_pairs_per_optim_step,
            num_optim_steps_per_rollout=num_optim_steps_per_rollout,
            micro_batch_size=micro_bs,
        )
        validate_same_microbatch_counts_across_dp(
            microbatch_schedule=microbatch_schedule,
            parallel_state=self.parallel_state,
        )

        # ------------- Forward / Backward -------------
        with timer("actor_train"):
            for step_id, microbatch_ranges in enumerate(microbatch_schedule):
                self.optimizer.zero_grad(set_to_none=True)

                w_pairs = sum(pair_hi - pair_lo for pair_lo, pair_hi in microbatch_ranges)

                log_stats: dict[str, list[torch.Tensor]] = defaultdict(list)

                for pair_lo, pair_hi in microbatch_ranges:
                    chunk = train_pairs[pair_lo:pair_hi]
                    loss_sum = self._forward_train_pair_batch(
                        chunk,
                        use_cfg=use_cfg,
                        guidance_scale=guidance_scale,
                        true_cfg_scale=true_cfg_scale,
                        clip_range=clip_range,
                        noise_level=noise_level,
                        num_train_timesteps=num_train_timesteps,
                        log_stats=log_stats,
                        device=device,
                    )
                    if not self.args.debug_skip_optimizer_step:
                        (loss_sum / float(w_pairs)).backward()

                self.prof.step(rollout_id=rollout_id)
                if not self.args.debug_skip_optimizer_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    log_stats["grad_norm"].append(grad_norm.detach())
                    self.optimizer.step()
                    self.lr_scheduler.step()
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # Do mean over all ranks for now, may need to be updated for p99, max, etc.
                reduced = {f"train/{k}": torch.stack(v).mean().item() for k, v in log_stats.items()}
                self._gather_and_log_metrics(rollout_id, reduced, step=self.global_step)

    def _forward_train_pair_batch(
        self,
        batch: list,
        *,
        use_cfg: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
        clip_range: float,
        noise_level: float,
        num_train_timesteps: int,
        log_stats: dict[str, list[torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        """One DiT forward + PPO loss over ``len(batch)`` train pairs. Returns sum of per-pair losses."""
        forward_dtype = self._forward_dtype
        train_pipeline_config = self.train_pipeline_config
        bsz = len(batch)
        latents_microbatch = torch.stack([batch[i]["latent"] for i in range(bsz)]).to(
            device=device, dtype=torch.float32
        )
        next_latents_microbatch = torch.stack([batch[i]["next_latent"] for i in range(bsz)]).to(
            device=device, dtype=torch.float32
        )
        timesteps_microbatch = torch.stack([batch[i]["timestep"] for i in range(bsz)]).to(
            device=device, dtype=torch.float32
        ).reshape(bsz)
        log_prob_old_microbatch = torch.stack([batch[i]["log_prob_old"] for i in range(bsz)]).to(
            device=device, dtype=torch.float32
        ).reshape(bsz)

        advantage = torch.tensor(
            [float(batch[i]["advantage"]) for i in range(bsz)],
            device=device,
            dtype=torch.float32,
        )
        advantage = torch.clamp(advantage, -self.args.diffusion_adv_clip_max, self.args.diffusion_adv_clip_max)

        # sgl-d's Qwen DiT divides timestep by num_train_timesteps inside
        # forward; diffusers' does not — pre-scale to match.
        timesteps_normalized = timesteps_microbatch / float(num_train_timesteps)

        pos_list = [
            train_pipeline_config.prepare_cond_kwargs(batch[i]["denoising_env"].pos_cond_kwargs, device)
            for i in range(bsz)
        ]
        neg_list = (
            [
                train_pipeline_config.prepare_cond_kwargs(batch[i]["denoising_env"].neg_cond_kwargs, device)
                for i in range(bsz)
            ]
            if use_cfg
            else None
        )

        first_sample_index = batch[0]["sample_index"]
        same_sample_microbatch = all(batch[i]["sample_index"] == first_sample_index for i in range(1, bsz))

        if same_sample_microbatch:
            pos_cond_microbatch = train_pipeline_config.expand_cond_for_timestep_batch(pos_list[0], bsz)
            neg_cond_microbatch = (
                train_pipeline_config.expand_cond_for_timestep_batch(neg_list[0], bsz)
                if use_cfg and neg_list is not None
                else None
            )
        elif use_cfg and neg_list is not None:
            pos_cond_microbatch = train_pipeline_config.collate_cond_for_sample_batch(pos_list, device)
            neg_cond_microbatch = train_pipeline_config.collate_cond_for_sample_batch(neg_list, device)
        else:
            pos_cond_microbatch = train_pipeline_config.collate_cond_for_sample_batch(pos_list, device)
            neg_cond_microbatch = None

        pos_cond_microbatch = _cast_cond_to_dtype(pos_cond_microbatch, forward_dtype)
        if neg_cond_microbatch is not None:
            neg_cond_microbatch = _cast_cond_to_dtype(neg_cond_microbatch, forward_dtype)

        # Cast inputs explicitly: FSDP MixedPrecisionPolicy casts params but
        # leaves fp32 inputs, which would run first matmul at higher precision
        # than rollout → systematic noise_pred drift.
        latents_input = latents_microbatch.to(forward_dtype)
        timesteps_input = timesteps_normalized.to(forward_dtype)

        def _forward(cond: dict) -> torch.Tensor:
            return self.model(
                hidden_states=latents_input,
                timestep=timesteps_input,
                return_dict=False,
                **cond,
            )[0]

        if not use_cfg:
            noise_pred_microbatch = _forward(pos_cond_microbatch)
        elif self.args.fsdp_cfg_batching:
            joint_cond = train_pipeline_config.collate_cond_for_sample_batch(pos_list + neg_list, device)
            joint_cond = _cast_cond_to_dtype(joint_cond, forward_dtype)
            # forward as a batch to align with some implementations in sglang-d
            joint_out = self.model(
                hidden_states=torch.cat([latents_input, latents_input], dim=0),
                timestep=torch.cat([timesteps_input, timesteps_input], dim=0),
                return_dict=False,
                **joint_cond,
            )[0]
            noise_pred_pos, noise_pred_neg = joint_out.chunk(2, dim=0)
            noise_pred_microbatch = train_pipeline_config.cfg_combine(
                noise_pred_pos,
                noise_pred_neg,
                guidance_scale,
                true_cfg_scale=true_cfg_scale,
            )
        else:
            noise_pred_pos = _forward(pos_cond_microbatch)
            noise_pred_neg = _forward(neg_cond_microbatch)
            noise_pred_microbatch = train_pipeline_config.cfg_combine(
                noise_pred_pos,
                noise_pred_neg,
                guidance_scale,
                true_cfg_scale=true_cfg_scale,
            )

        _, log_prob_new_microbatch, prev_sample_mean_new, std_dev_t_new = sde_step_with_logprob(
            self.scheduler,
            noise_pred_microbatch.float(),
            timesteps_microbatch,
            latents_microbatch.float(),
            prev_sample=next_latents_microbatch.float(),
            noise_level=noise_level,
        )

        log_prob_new = log_prob_new_microbatch.reshape(bsz)
        log_prob_old = log_prob_old_microbatch.reshape(bsz)
        ratio = torch.exp(log_prob_new - log_prob_old)
        unclipped = -advantage * ratio
        clipped = -advantage * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        per_pair_loss = torch.maximum(unclipped, clipped)
        loss_sum = per_pair_loss.sum()

        with torch.no_grad():
            log_stats["loss"].append(per_pair_loss.mean().detach())
            log_stats["loss_abs_mean"].append(per_pair_loss.abs().mean().detach())
            log_stats["adv_abs_mean"].append(advantage.abs().mean().detach())
            log_stats["ratio_abs_minus_1"].append((ratio - 1.0).abs().mean().detach())
            log_stats["approx_kl"].append(0.5 * torch.mean((log_prob_new - log_prob_old) ** 2).detach())
            log_stats["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).detach())
            log_stats["log_prob_new_idx_0"].append(log_prob_new.reshape(-1)[0].detach())
            log_stats["log_prob_old_idx_0"].append(log_prob_old.reshape(-1)[0].detach())
            log_stats["log_prob_mean_abs_diff"].append(torch.mean(torch.abs(log_prob_new - log_prob_old)).detach())

            for debug_key, train_tensor in (
                ("rollout_step_model_output", noise_pred_microbatch),
                ("rollout_step_prev_sample_mean", prev_sample_mean_new),
                ("rollout_step_noise_std_dev", std_dev_t_new),
            ):
                rollout_tensor = stack_train_pair_rollout_debug(batch, debug_key)
                if rollout_tensor is None:
                    continue
                _append_rollout_train_abs_diff_stats(
                    log_stats,
                    debug_key,
                    train_tensor.float(),
                    rollout_tensor.to(device=device, dtype=torch.float32).float(),
                )

        return loss_sum


def _append_rollout_train_abs_diff_stats(
    log_stats: dict[str, list],
    prefix: str,
    train: torch.Tensor,
    rollout: torch.Tensor,
) -> None:
    bsz = train.shape[0]
    diff = (train.reshape(bsz, -1).float() - rollout.reshape(bsz, -1).float()).abs()
    ref_max = rollout.reshape(bsz, -1).float().abs().max() + 1e-30
    log_stats[f"{prefix}_max_abs_diff"].append(diff.max().detach())
    log_stats[f"{prefix}_mean_abs_diff"].append(diff.mean().detach())
    log_stats[f"{prefix}_rel_max"].append((diff.max() / ref_max).detach())

def _cast_cond_to_dtype(cond: dict, dtype: torch.dtype) -> dict:
    """Cast floating-point tensors to the model's compute dtype; leave bool
    masks / int / list / scalar values untouched. The bool
    encoder_hidden_states_mask must NOT be cast. 
    """
    out: dict = {}
    for k, v in cond.items():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            out[k] = v.to(dtype)
        else:
            out[k] = v
    return out


@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    """ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py"""
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)

    torch.cuda.synchronize()


def _resolve_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]

def apply_lora(model: torch.nn.Module, args: Namespace, train_pipeline_config) -> None:
    """Apply PEFT LoRA to the model.

    Args:
        model: The model to apply LoRA to.
        args: Arguments containing LoRA settings.
        train_pipeline_config: The train pipeline config.
    """
    from peft import LoraConfig, get_peft_model

    # Per-model fallback when --lora-target-modules is unset (runtime inference: depends on loaded pipeline).
    targets = args.lora_target_modules or train_pipeline_config.lora_target_modules
    init_lora_weight = args.diffusion_init_lora_weight
    if init_lora_weight == "kaiming-uniform":
        init_lora_weight = True # namely kaiming-uniform
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=targets,
        init_lora_weights=init_lora_weight,
    ))
    if dist.get_rank() == 0:
        model.print_trainable_parameters()
    return model

def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None):
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
    ]

    param_dtype = _resolve_dtype(args.diffusion_forward_dtype)
    reduce_dtype = _resolve_dtype(args.fsdp_reduce_dtype)
    logger.info(f"FSDP: wrapping {len(modules)} modules of type {layer_cls_to_wrap}, param_dtype={param_dtype}, reduce_dtype={reduce_dtype}")

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)

    return model
