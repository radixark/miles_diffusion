import logging
import warnings
from argparse import Namespace
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from itertools import chain

import ray
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

import miles.backends.fsdp_utils.configs.qwen_image  # noqa: F401 — register pipeline config
import miles.backends.fsdp_utils.configs.sd3  # noqa: F401 — register pipeline config
import miles.backends.fsdp_utils.configs.wan2_2  # noqa: F401 — register pipeline config
from miles.ray.train_actor import TrainRayActor
from miles.utils import tracking_utils, train_metric_utils
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.profile_utils import TrainProfiler
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils.train_data_utils import (
    build_microbatch_schedule,
    scheduler_meta_from_rollout,
    stack_train_pair_rollout_debug,
    validate_same_microbatch_counts_across_dp,
)
from . import checkpoint
from .diffusion_update_weight_utils import (
    DiffusionUpdateWeightFromTensor,
    DiffusionUpdateWeightFromTensorLoRA,
    DiffusionUpdateWeightFromTensorLoRAIPC,
)
from .lr_scheduler import get_lr_scheduler
from .parallel import create_fsdp_parallel_state

logger = logging.getLogger(__name__)


def _enable_deterministic_training(args: Namespace) -> None:
    """Train-actor deterministic mode. NCCL/CUBLAS env is set at spawn (actor_group);
    here we set the torch-runtime knobs."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=False is required: SDPA's deterministic backward is gated on
    # !warnOnly (aten attention_backward.cu), so warn_only=True is a no-op on native.
    torch.use_deterministic_algorithms(True, warn_only=False)


class FSDPTrainRayActor(TrainRayActor):
    """FSDP training actor for diffusion GRPO.

    Loads only the DiT (transformer) from a diffusers pipeline, wraps it with
    FSDP, and trains with a PPO-clipped objective aligned with flow GRPO.
    """

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False) -> int:  # type: ignore[override]
        super().init(args, role, with_ref)

        if args.deterministic_mode:
            _enable_deterministic_training(args)

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

        from miles.utils.misc import load_function

        self.train_pipeline_config = load_function(args.train_pipeline_config_path)()
        self.train_pipeline_config.configure(args)
        self.model_backend = load_function(args.model_backend_path)(self.train_pipeline_config)
        if args.deterministic_mode:
            # flash-attn is opaque to torch's determinism flag; backends patch their own dispatch.
            self.model_backend.enable_deterministic_attention(args.fsdp_attention_backend)
        self.scheduler = self.model_backend.load_scheduler(args)
        rank = dist.get_rank()

        self.models: dict[str, torch.nn.Module] = {}
        for component in args.update_weight_target_modules:
            # per raw component (wan2.2 has two transformers), before LoRA/FSDP wrap
            with self._init_weight_context():
                model = self.model_backend.load_component(component, args, master_dtype=self._master_dtype)
            if args.fsdp_attention_backend is not None:
                self.model_backend.set_attention_backend(model, args.fsdp_attention_backend)

            if args.use_lora:
                model = apply_lora(model, args, self.train_pipeline_config)

            model.train()

            if args.gradient_checkpointing:
                self.model_backend.enable_gradient_checkpointing(model)

            if rank != 0 and any(not parameter.is_meta for parameter in model.parameters()):
                raise RuntimeError(f"{component} did not honor meta initialization")
            sync_model_dtypes(model)
            full_state = model.state_dict() if rank == 0 else {}
            model = apply_fsdp2(
                model,
                mesh=self.parallel_state.dp_mesh,
                cpu_offload=self.args.fsdp_cpu_offload,
                args=self.args,
                no_split_modules=self.model_backend.fsdp_no_split_modules(model),
            )
            load_sharded_model(model, full_state, cpu_offload=self.args.fsdp_cpu_offload)
            del full_state
            self.train_pipeline_config.postprocess_model_after_materialize(model)
            self.models[component] = model
        # Force a sync to ensure sharding is complete and old memory is freed.
        torch.cuda.synchronize()
        clear_memory()

        if len(self.models) == 1:
            self.model = next(iter(self.models.values()))
        else:
            self.model = torch.nn.ModuleDict(self.models)

        from miles.utils.misc import load_function

        self.sde_backend = load_function(args.sde_step_backend_path)(
            self.scheduler,
            sde_timestep_divisor=self.train_pipeline_config.sde_timestep_divisor,
        )

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

        # fp16 policy gradients are small enough to underflow without scaling.
        # ShardedGradScaler keeps the found_inf decision synchronized across
        # FSDP ranks; it is a no-op for bf16/fp32.
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        self.scaler = ShardedGradScaler(
            enabled=(self._forward_dtype == torch.float16),
        )

        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)
        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)

        # sglang-d now supports /update_weights_from_tensor (PR #20464).
        if self.args.debug_train_only:
            self.weight_updater = None
        elif self.args.use_lora and self.args.lora_ipc_weight_sync:
            self.weight_updater = DiffusionUpdateWeightFromTensorLoRAIPC(self.args, self.models)
        elif self.args.use_lora:
            self.weight_updater = DiffusionUpdateWeightFromTensorLoRA(self.args, self.models)
        else:
            self.weight_updater = DiffusionUpdateWeightFromTensor(self.args, self.models)

        checkpoint.finalize_load(self, checkpoint_payload)

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()

        return self.args.start_rollout_id

    @contextmanager
    def _init_weight_context(self):
        """Build real weights on rank0 and allocation-free meta weights elsewhere."""
        if dist.get_rank() == 0:
            with torch.device("cpu"):
                yield
            return

        from accelerate import init_empty_weights

        # Some models compute buffer values during __init__, which cannot run on meta.
        with init_empty_weights(include_buffers=False), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"for .*: copying from a non-meta parameter in the checkpoint to a meta parameter.*",
            )
            yield

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
                + " ".join(
                    f"{k}={v:.6e}"
                    for k, v in sorted(reduced.items())
                    if k not in ("train/epoch", "rollout/step", "train/step")
                )
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
        scale as mean over all train pairs in the window (``loss_chunk / num_local_pairs``).
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

        # ------------- KL loss -------------
        kl_beta = float(self.args.diffusion_kl_beta)
        if kl_beta > 0 and not self.args.use_lora:
            raise ValueError(
                "--diffusion-kl-beta currently requires --use-lora so the base model can be used as reference."
            )
        if kl_beta > 0 and not all(hasattr(m, "disable_adapter") for m in self.models.values()):
            raise RuntimeError("Diffusion KL requires PEFT models exposing disable_adapter() after FSDP wrapping.")

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

        # ------------- Micro-batch schedule -------------
        num_optim_steps_per_rollout = self.args.num_steps_per_rollout
        if num_pairs % num_optim_steps_per_rollout != 0:
            raise ValueError(
                f"num_pairs_shard={num_pairs} not divisible by " f"num_steps_per_rollout={num_optim_steps_per_rollout}"
            )
        num_pairs_per_optim_step = num_pairs // num_optim_steps_per_rollout
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
            for microbatch_ranges in microbatch_schedule:
                self.optimizer.zero_grad(set_to_none=True)

                num_local_pairs = sum(pair_hi - pair_lo for pair_lo, pair_hi in microbatch_ranges)

                # LEGACY 2D parity: pad cond to the whole-window width. TODO: remove with legacy 2D path.
                legacy_pad_to_len = self._maybe_legacy_window_pad_len(train_pairs, microbatch_ranges)

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
                        kl_beta=kl_beta,
                        pad_to_len=legacy_pad_to_len,
                    )
                    if not self.args.debug_skip_optimizer_step:
                        # ShardedGradScaler keeps fp16 policy grads from underflowing
                        # (required for SD3.5 fp16 forward); no-op for bf16/fp32.
                        self.scaler.scale(loss_sum / float(num_local_pairs)).backward()

                self.prof.step(rollout_id=rollout_id)
                if not self.args.debug_skip_optimizer_step:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    if isinstance(grad_norm, DTensor):
                        # clip returns a lazily-reduced partial norm; materialize it,
                        # otherwise the logged metric leaks the local shard's value.
                        grad_norm = grad_norm.full_tensor()
                    log_stats["grad_norm"].append(grad_norm.detach())
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # Do mean over all ranks for now, may need to be updated for p99, max, etc.
                reduced = {f"train/{k}": torch.stack(v).mean().item() for k, v in log_stats.items()}
                self._gather_and_log_metrics(rollout_id, reduced, step=self.global_step)

    def _maybe_legacy_window_pad_len(self, train_pairs: list, microbatch_ranges: list) -> int | None:
        """LEGACY 2D parity: the whole-window max cond seq_len (like the legacy tile path), or
        None unless the legacy --micro-batch-size-sample>1 path is active. TODO: remove with it."""
        if self.args.micro_batch_size_sample is None or self.args.micro_batch_size_sample <= 1:
            return None
        conds = []
        for pair_lo, pair_hi in microbatch_ranges:
            for pair in train_pairs[pair_lo:pair_hi]:
                env = pair["denoising_env"]
                conds.append(env.pos_cond_kwargs)
                if env.neg_cond_kwargs is not None:
                    conds.append(env.neg_cond_kwargs)
        return self.train_pipeline_config.maybe_legacy_window_pad_len(conds)

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
        kl_beta: float = 0.0,
        pad_to_len: int | None = None,
    ) -> torch.Tensor:
        """One DiT forward + PPO loss over ``len(batch)`` train pairs. Returns sum of per-pair losses."""
        forward_dtype = self._forward_dtype
        train_pipeline_config = self.train_pipeline_config
        bsz = len(batch)

        def _stack(key):
            return torch.stack([pair[key] for pair in batch]).to(device=device, dtype=torch.float32)

        latents_microbatch = _stack("latent")  # (bsz, *latent_dims)
        next_latents_microbatch = _stack("next_latent")  # (bsz, *latent_dims)
        timesteps_microbatch = _stack("timestep")  # (bsz,) -- per-pair timestep is scalar
        next_timesteps_microbatch = _stack("next_timestep")  # (bsz,) -- next rollout timestep (0 at terminal)
        log_prob_old_microbatch = _stack("log_prob_old")  # (bsz,) -- per-pair log_prob is scalar

        advantage = torch.tensor(  # (bsz,)
            [float(pair["advantage"]) for pair in batch],
            device=device,
            dtype=torch.float32,
        )
        advantage = torch.clamp(advantage, -self.args.diffusion_adv_clip_max, self.args.diffusion_adv_clip_max)

        if len(self.models) == 1:
            component, model = next(iter(self.models.items()))
        else:
            components = {
                train_pipeline_config.component_for_timestep(t, num_train_timesteps)
                for t in timesteps_microbatch.tolist()
            }
            # to prevent mixing denoising phases in a single micro-batch
            # Just in case when some customized step strategy is used that
            # may violate the assumption of one phase per micro-batch, we raise an error here
            if len(components) > 1:
                raise ValueError(
                    f"Micro-batch mixes denoising phases {sorted(components)}; set "
                    "--micro-batch-size 1 so each forward is phase-pure (one DiT, one CFG scale)."
                )
            component = components.pop()
            model = self.models[component]
            guidance_scale = train_pipeline_config.select_guidance_scale(
                float(timesteps_microbatch[0]),
                num_train_timesteps,
                guidance_scale,
                self.args.diffusion_guidance_scale_2,
            )

        # sgl-d's Qwen DiT divides timestep by num_train_timesteps inside
        # forward; diffusers' does not. SD3 already expects raw timesteps.
        if train_pipeline_config.needs_timestep_scaling:
            timesteps_for_model = timesteps_microbatch / float(num_train_timesteps)
        else:
            timesteps_for_model = timesteps_microbatch

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

        # Collate cond once, up front. With CFG batching, pos+neg must share one
        # padded width and go through a single joint forward, so build that joint cond
        # directly; otherwise build pos (and neg) separately. (A single-sample
        # timestep-stacked micro-batch is just collate of bsz copies of one sample --
        # bitwise-equivalent to the old expand_cond_for_timestep_batch path; the
        # all-True mask qwen adds is a verified forward no-op, see
        # tests/manual/check_mask_equivalence.py.)
        cfg_batching = use_cfg and bool(self.args.fsdp_cfg_batching)
        joint_cond = None
        pos_cond_microbatch = None
        neg_cond_microbatch = None
        if cfg_batching:
            joint_cond = _cast_cond_to_dtype(
                train_pipeline_config.collate_cond_for_sample_batch(
                    pos_list + neg_list, device, pad_to_len=pad_to_len
                ),
                forward_dtype,
            )
        else:
            pos_cond_microbatch = _cast_cond_to_dtype(
                train_pipeline_config.collate_cond_for_sample_batch(pos_list, device, pad_to_len=pad_to_len),
                forward_dtype,
            )
            if use_cfg and neg_list is not None:
                neg_cond_microbatch = _cast_cond_to_dtype(
                    train_pipeline_config.collate_cond_for_sample_batch(neg_list, device, pad_to_len=pad_to_len),
                    forward_dtype,
                )

        # Cast inputs explicitly: FSDP MixedPrecisionPolicy casts params but
        # leaves fp32 inputs, which would run first matmul at higher precision
        # than rollout → systematic noise_pred drift.
        latents_input = latents_microbatch.to(forward_dtype)
        timesteps_input = timesteps_for_model.to(forward_dtype)

        def _compute_noise_pred(disable_adapter: bool = False) -> torch.Tensor:
            adapter_ctx = model.disable_adapter() if disable_adapter else nullcontext()
            with adapter_ctx:
                return train_pipeline_config.compute_noise_pred(
                    model=model,
                    latents_input=latents_input,
                    timesteps_input=timesteps_input,
                    pos_cond=pos_cond_microbatch,
                    neg_cond=neg_cond_microbatch,
                    joint_cond=joint_cond,
                    use_cfg=use_cfg,
                    cfg_batching=cfg_batching,
                    guidance_scale=guidance_scale,
                    true_cfg_scale=true_cfg_scale,
                )

        noise_pred_microbatch = _compute_noise_pred()

        _, log_prob_new_microbatch, prev_sample_mean_new, std_dev_t_new = self.sde_backend.sde_step_logprob(
            noise_pred_microbatch.float(),
            timesteps_microbatch,
            next_timesteps_microbatch,
            latents_microbatch.float(),
            prev_sample=next_latents_microbatch.float(),
            noise_level=noise_level,
        )

        log_prob_new = log_prob_new_microbatch  # (bsz,) -- sde_step_with_logprob means over non-batch dims
        log_prob_old = log_prob_old_microbatch  # (bsz,)
        ratio = torch.exp(log_prob_new - log_prob_old)  # (bsz,)
        unclipped = -advantage * ratio
        clipped = -advantage * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        per_pair_loss = torch.maximum(unclipped, clipped)
        loss_sum = per_pair_loss.sum()

        # ------------- KL loss (vs LoRA base model as reference) -------------
        kl_loss = loss_sum.new_zeros(())
        if kl_beta > 0:
            with torch.no_grad():
                ref_noise_pred_microbatch = _compute_noise_pred(disable_adapter=True)
                # TODO: unify sde_step_with_logprob with rollout and trainer forward paths.
                _, _, prev_sample_mean_ref, _ = self.sde_backend.sde_step_logprob(
                    ref_noise_pred_microbatch.float(),
                    timesteps_microbatch,
                    next_timesteps_microbatch,
                    latents_microbatch.float(),
                    prev_sample=next_latents_microbatch.float(),
                    noise_level=noise_level,
                )
            kl_per_pair = ((prev_sample_mean_new - prev_sample_mean_ref) ** 2).mean(
                dim=tuple(range(1, prev_sample_mean_new.ndim)),
                keepdim=True,
            ) / (2 * std_dev_t_new**2)
            loss_sum = loss_sum + kl_beta * kl_per_pair.sum()
            kl_loss = kl_per_pair.mean()

        with torch.no_grad():
            log_stats["loss"].append((per_pair_loss.mean() + kl_beta * kl_loss).detach())
            log_stats["policy_loss"].append(per_pair_loss.mean().detach())
            log_stats["kl_loss"].append(kl_loss.detach())
            log_stats["loss_abs_mean"].append(per_pair_loss.abs().mean().detach())
            log_stats["adv_abs_mean"].append(advantage.abs().mean().detach())
            log_stats["ratio_abs_minus_1"].append((ratio - 1.0).abs().mean().detach())
            log_stats["approx_kl"].append(0.5 * torch.mean((log_prob_new - log_prob_old) ** 2).detach())
            log_stats["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).detach())
            log_stats["log_prob_new_idx_0"].append(log_prob_new[0].detach())
            log_stats["log_prob_old_idx_0"].append(log_prob_old[0].detach())
            log_prob_mean_abs_diff = torch.mean(torch.abs(log_prob_new - log_prob_old)).detach()
            log_stats["log_prob_mean_abs_diff"].append(log_prob_mean_abs_diff)
            if len(self.models) > 1:
                log_stats[f"log_prob_mean_abs_diff_{component}"].append(log_prob_mean_abs_diff)

            # model_output_* checks the train forward reproduces the rollout forward -- the only
            # model-dependent consistency metric (std_dev/prev_sample_mean are deterministic
            # functions of it). Matches the legacy actor metric name.
            rollout_model_output = stack_train_pair_rollout_debug(batch, "rollout_step_model_output")
            if rollout_model_output is not None:
                mean_abs_diff = _append_rollout_train_abs_diff_stats(
                    log_stats,
                    "model_output",
                    noise_pred_microbatch.float(),
                    rollout_model_output.to(device=device, dtype=torch.float32),
                )
                if len(self.models) > 1:
                    log_stats[f"model_output_mean_abs_diff_{component}"].append(mean_abs_diff)

        return loss_sum


def _append_rollout_train_abs_diff_stats(
    log_stats: dict[str, list],
    prefix: str,
    train: torch.Tensor,
    rollout: torch.Tensor,
) -> torch.Tensor:
    bsz = train.shape[0]
    diff = (train.reshape(bsz, -1).float() - rollout.reshape(bsz, -1).float()).abs()
    ref_max = rollout.reshape(bsz, -1).float().abs().max() + 1e-30
    mean_abs_diff = diff.mean().detach()
    log_stats[f"{prefix}_max_abs_diff"].append(diff.max().detach())
    log_stats[f"{prefix}_mean_abs_diff"].append(mean_abs_diff)
    log_stats[f"{prefix}_rel_max"].append((diff.max() / ref_max).detach())
    return mean_abs_diff


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


def apply_lora(model: torch.nn.Module, args: Namespace, train_pipeline_config) -> torch.nn.Module:
    """Apply PEFT LoRA, leaving non-rank0 adapters uninitialized on meta."""
    from peft import LoraConfig, get_peft_model

    on_meta = dist.get_rank() != 0
    # Per-model fallback when --lora-target-modules is unset (runtime inference: depends on loaded pipeline).
    targets = args.lora_target_modules or train_pipeline_config.lora_target_modules
    init_lora_weight = args.diffusion_init_lora_weight
    if init_lora_weight == "kaiming-uniform":
        init_lora_weight = True  # namely kaiming-uniform
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=targets,
            init_lora_weights=False if on_meta else init_lora_weight,
        ),
        low_cpu_mem_usage=on_meta,
    )
    if dist.get_rank() == 0:
        model.print_trainable_parameters()
    return model


def load_sharded_model(model: torch.nn.Module, full_state: dict, cpu_offload: bool) -> None:
    """Materialize FSDP2 shards from rank0's full state dict."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    if dist.get_rank() == 0:
        # Rank 0 was sharded on real CPU weights; move them (and real buffers) along.
        model.to(device=torch.cuda.current_device(), non_blocking=True)
    else:
        # to_empty creates tensors on device without initializing memory.
        model.to_empty(device=torch.cuda.current_device())

    set_model_state_dict(
        model,
        full_state,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=cpu_offload,
            broadcast_from_rank0=True,
        ),
    )
    # set_model_state_dict only covers state_dict entries; non-persistent buffers
    # (e.g. Wan's rope tables) exist in no state_dict and were wiped by to_empty
    # on non-rank0 ranks — take rank0's real values for every buffer.
    for buffer in model.buffers():
        dist.broadcast(buffer, src=0)

    if cpu_offload:
        model.to("cpu", non_blocking=True)
        # CPUOffloadPolicy manages params only; buffers must live on GPU for forward.
        for buffer in model.buffers():
            buffer.data = buffer.data.to(torch.cuda.current_device())


def sync_model_dtypes(model: torch.nn.Module) -> None:
    """Match meta parameter and buffer dtypes to rank0 before sharding."""
    rank = dist.get_rank()
    tensors = list(chain(model.parameters(), model.buffers()))
    dtypes = [tensor.dtype for tensor in tensors] if rank == 0 else None
    objects = [dtypes]
    dist.broadcast_object_list(objects, src=0)
    if rank != 0:
        for tensor, dtype in zip(tensors, objects[0], strict=True):
            tensor.data = tensor.data.to(dtype)


def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None, no_split_modules=None):
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = no_split_modules if no_split_modules is not None else model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [module for name, module in model.named_modules() if module.__class__.__name__ in layer_cls_to_wrap]

    param_dtype = _resolve_dtype(args.diffusion_forward_dtype)
    reduce_dtype = _resolve_dtype(args.fsdp_reduce_dtype)
    logger.info(
        f"FSDP: wrapping {len(modules)} modules of type {layer_cls_to_wrap}, param_dtype={param_dtype}, reduce_dtype={reduce_dtype}"
    )

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    if args.gradient_checkpointing:
        # MixedPrecisionPolicy does not cast buffers; a buffer above param_dtype
        # makes the ckpt recompute dtype-diverge from the forward and abort.
        for module in model.modules():
            for name, buf in module.named_buffers(recurse=False):
                if buf.is_floating_point() and buf.dtype != param_dtype:
                    persistent = name not in module._non_persistent_buffers_set
                    module.register_buffer(name, buf.to(param_dtype), persistent=persistent)

    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)

    return model
