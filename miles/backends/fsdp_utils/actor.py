import logging
import warnings
from argparse import Namespace
from contextlib import contextmanager, nullcontext

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
from miles.utils.metric_buffer import MetricBuffer
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils.train_data_utils import (
    build_microbatch_schedule,
    scheduler_meta_from_rollout,
    validate_same_microbatch_counts_across_train_ranks,
    validate_sample_aligned_windows,
)

from . import checkpoint
from .diffusion_update_weight_utils import (
    DiffusionUpdateWeightFromTensor,
    DiffusionUpdateWeightFromTensorLoRA,
    DiffusionUpdateWeightFromTensorLoRAIPC,
)
from .ema import EmaShadow
from .input_dtype_policy import apply_input_dtype_policy
from .loss_hub import DiffusionLossContext, flow_grpo_loss_formula, prepare_flow_grpo_batch
from .lr_scheduler import get_lr_scheduler
from .metrics import new_metric_buffer
from .mixed_precision import compile_param_dtype_maps, parse_dtype_from_str
from .parallel import create_fsdp_parallel_state
from .sequence_parallel.plan import apply_sequence_parallel

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

        # Offline dashboard: record Timer phases + (rank 0) NVML GPU util when explicitly enabled.
        from miles.dashboard import hooks

        hooks.register_train_actor(args, role)

        self.train_parallel_config = {
            "dp_size": self.parallel_state.get_mesh("dp").size(),
        }

        if self.args.debug_rollout_only:
            return 0

        if self.args.offload_train and self.args.fsdp_cpu_offload:
            self.args.offload_train = False

        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        if self.args.start_rollout_id is None:
            self.args.start_rollout_id = 0

        self._master_dtype = parse_dtype_from_str(args.fsdp_master_dtype)
        self._forward_dtype = parse_dtype_from_str(args.diffusion_forward_dtype)

        from miles.utils.misc import load_function

        self.train_pipeline_config = load_function(args.train_pipeline_config_path)()
        self.train_pipeline_config.configure(args)
        self.model_backend = load_function(args.model_backend_path)(self.train_pipeline_config)
        if args.deterministic_mode:
            # flash-attn is opaque to torch's determinism flag; backends patch their own dispatch.
            self.model_backend.enable_deterministic_attention(args.fsdp_attention_backend)
        self.scheduler = self.model_backend.load_scheduler(args)
        rank = dist.get_rank()
        materialize_weights = rank == 0

        self.models: dict[str, torch.nn.Module] = {}
        for component in args.update_weight_target_modules:
            # per raw component (wan2.2 has two transformers), before LoRA/FSDP wrap
            with self._model_init_context(materialize_weights=materialize_weights):
                model = self.model_backend.load_component(
                    component,
                    args,
                    master_dtype=self._master_dtype,
                    materialize_weights=materialize_weights,
                )
            if args.fsdp_attention_backend is not None:
                self.model_backend.set_attention_backend(model, args.fsdp_attention_backend)

            # Enable checkpointing on the raw model before PEFT wraps it. The flag
            # is consumed when transformer blocks run, so LoRA layers inserted
            # below remain inside the checkpointed block forward.
            if args.gradient_checkpointing:
                self.model_backend.enable_gradient_checkpointing(model)

            if args.use_lora:
                model = apply_lora(model, args, self.train_pipeline_config)

            model.train()

            if rank != 0 and any(not parameter.is_meta for parameter in model.parameters()):
                raise RuntimeError(f"{component} did not honor meta initialization")
            checkpoint.sync_model_dtypes(model)
            full_state = model.state_dict() if rank == 0 else {}
            model = apply_fsdp2(
                model,
                self.model_backend.fsdp_parallel_plan(model),
                mesh=self.parallel_state.get_mesh("fsdp"),
                cpu_offload=self.args.fsdp_cpu_offload,
                args=self.args,
            )
            checkpoint.broadcast_full_state_to_fsdp(
                model,
                full_state,
                cpu_offload=self.args.fsdp_cpu_offload,
            )
            del full_state
            self.train_pipeline_config.postprocess_model_after_materialize(model)
            self.models[component] = model

        if self.parallel_state.get_optional_mesh("sp") is not None:
            for model in self.models.values():
                plan = self.model_backend.sequence_parallel_plan(model)
                apply_sequence_parallel(
                    model,
                    self.parallel_state,
                    plan,
                    self.model_backend.install_sequence_parallel_attention,
                )

        # Force a sync to ensure sharding is complete and old memory is freed.
        torch.cuda.synchronize()
        clear_memory()

        if len(self.models) == 1:
            self.model = next(iter(self.models.values()))
        else:
            self.model = torch.nn.ModuleDict(self.models)

        self.sde_backend = load_function(args.sde_step_backend_path)(
            self.scheduler,
            sde_timestep_divisor=self.train_pipeline_config.sde_timestep_divisor,
        )

        self.custom_prepare_train_batch_func = (
            load_function(args.custom_prepare_train_batch_path)
            if args.custom_prepare_train_batch_path is not None
            else None
        )
        self.custom_loss_formula_func = (
            load_function(args.custom_loss_function_path) if args.custom_loss_function_path is not None else None
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

        self.ema_shadow = None
        if self.args.use_ema:
            self.ema_shadow = EmaShadow(
                (p for m in self.models.values() for p in m.parameters()),
                decay=self.args.ema_decay_init,
                uprate=self.args.ema_decay_ramp,
                uphold=self.args.ema_decay_max,
                flat_steps=self.args.ema_decay_flat_steps,
            )

        # sglang-d now supports /update_weights_from_tensor (PR #20464).
        if self.args.train_only:
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

        return self.args.start_rollout_id

    @contextmanager
    def _model_init_context(self, *, materialize_weights: bool):
        """Build real CPU weights on rank 0 and meta weights elsewhere."""
        if materialize_weights:
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
        if self.args.train_only or self.args.debug_rollout_only:
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

        ema_shadow = self.ema_shadow
        if ema_shadow is not None:
            delta = ema_shadow.update()
            if dist.get_rank() == 0:
                logger.info("EMA shadow updated (decay=%.4f step=%d)", delta, ema_shadow.step)
        rollout_weight_context = (
            ema_shadow.swap_in() if ema_shadow is not None and self.args.ema_rollout_policy == "ema" else nullcontext()
        )
        with rollout_weight_context:
            self.weight_updater.update_weights()
        clear_memory()

    def _log_metrics(self, rollout_id: int, log_dict: dict[str, float], step: int) -> None:
        """Emit already-reduced metrics; every DP group computed the same values."""
        if dist.get_rank() != 0:
            return
        log_dict["train/lr"] = float(self.optimizer.param_groups[0]["lr"])
        log_dict["train/epoch"] = float(rollout_id)
        log_dict["rollout/step"] = compute_rollout_step(self.args, rollout_id)
        log_dict["train/step"] = float(step)
        tracking_utils.log(self.args, log_dict, step_key="train/step")

        logger.info(
            f"[train step {int(step)}] rollout={rollout_id} "
            + " ".join(
                f"{k}={v:.6e}"
                for k, v in sorted(log_dict.items())
                if k not in ("train/epoch", "rollout/step", "train/step")
            )
        )

    def train(self, rollout_id: int, rollout_data_ref) -> None:  # type: ignore[override]
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            rollout_data = ray.get(rollout_data_ref[self.parallel_state.get_mesh("dp").get_local_rank()].inner)
            if self.args.debug_rollout_only:
                return
            self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
        )

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        """Run the shared diffusion training loop."""
        device = torch.device("cuda", torch.cuda.current_device())

        train_pairs: list = rollout_data["train_data"]
        if not train_pairs:
            raise ValueError("rollout_data['train_data'] is empty")

        num_pairs = len(train_pairs)

        ref_mode = self.args.ref_mode
        if ref_mode == "lora_base" and not all(hasattr(m, "disable_adapter") for m in self.models.values()):
            raise RuntimeError(
                "--ref-mode lora_base requires PEFT models exposing disable_adapter() after FSDP wrapping."
            )

        # ------------- Rollout Scheduler Metadata -------------
        scheduler_timesteps, scheduler_sigmas = scheduler_meta_from_rollout(
            rollout_data,
            device=device,
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
        validate_same_microbatch_counts_across_train_ranks(
            microbatch_schedule=microbatch_schedule,
            parallel_state=self.parallel_state,
        )
        if self.args.loss_type == "nft":
            validate_sample_aligned_windows(
                train_pairs=train_pairs,
                microbatch_schedule=microbatch_schedule,
            )

        loss_ctx = DiffusionLossContext(
            models=self.models,
            train_pipeline_config=self.train_pipeline_config,
            sde_backend=self.sde_backend,
            scheduler=self.scheduler,
            args=self.args,
            forward_dtype=self._forward_dtype,
            device=device,
            rollout_id=rollout_id,
            dp_rank=self.parallel_state.dp_rank,
        )

        # ------------- Recompute old log-probs (impl-consistent PPO ratio) -------------
        if self.args.diffusion_recompute_old_log_prob:
            with timer("recompute_old_log_prob"), torch.no_grad():
                # write_old_log_prob returns before recording; this is never reduced.
                unused_metrics = new_metric_buffer(self.parallel_state.dp_group, device, self.models)
                # Skip window 0: its training forward runs on the same pre-update weights and doubles as the recompute.
                # Start the id after window 0's micro-batches to stay aligned with the training loop.
                microbatch_id = len(microbatch_schedule[0])
                for microbatch_ranges in microbatch_schedule[1:]:
                    legacy_pad_to_len = self._maybe_legacy_window_pad_len(train_pairs, microbatch_ranges)
                    for pair_lo, pair_hi in microbatch_ranges:
                        loss_ctx.microbatch_id = microbatch_id
                        microbatch_id += 1
                        self._forward_train_pair_batch(
                            loss_ctx,
                            train_pairs[pair_lo:pair_hi],
                            metrics=unused_metrics,
                            pad_to_len=legacy_pad_to_len,
                            write_old_log_prob=True,
                        )

        # ------------- Forward / Backward -------------
        with timer("actor_train"):
            microbatch_id = 0
            for optim_step_idx, microbatch_ranges in enumerate(microbatch_schedule):
                self.optimizer.zero_grad(set_to_none=True)

                old_log_prob_from_new = self.args.diffusion_recompute_old_log_prob and optim_step_idx == 0

                num_local_pairs = sum(pair_hi - pair_lo for pair_lo, pair_hi in microbatch_ranges)

                # LEGACY 2D parity: pad cond to the whole-window width. TODO: remove with legacy 2D path.
                legacy_pad_to_len = self._maybe_legacy_window_pad_len(train_pairs, microbatch_ranges)

                metrics = new_metric_buffer(self.parallel_state.dp_group, device, self.models)

                for pair_lo, pair_hi in microbatch_ranges:
                    chunk = train_pairs[pair_lo:pair_hi]
                    loss_ctx.microbatch_id = microbatch_id
                    microbatch_id += 1
                    loss_sum = self._forward_train_pair_batch(
                        loss_ctx,
                        chunk,
                        metrics=metrics,
                        pad_to_len=legacy_pad_to_len,
                        old_log_prob_from_new=old_log_prob_from_new,
                    )
                    if not self.args.debug_skip_optimizer_step:
                        # ShardedGradScaler keeps fp16 policy grads from underflowing
                        # (required for SD3.5 fp16 forward); no-op for bf16/fp32.
                        self.scaler.scale(loss_sum / float(num_local_pairs)).backward()

                if not self.args.debug_skip_optimizer_step:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    if isinstance(grad_norm, DTensor):
                        # clip returns a lazily-reduced partial norm; materialize it,
                        # otherwise the logged metric leaks the local shard's value.
                        grad_norm = grad_norm.full_tensor()
                    metrics.emit_replicated("grad_norm", grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                reduced = {f"train/{key}": value for key, value in metrics.reduce().items()}
                self._log_metrics(rollout_id, reduced, step=self.global_step)

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
        ctx: DiffusionLossContext,
        batch: list,
        *,
        metrics: MetricBuffer,
        pad_to_len: int | None = None,
        write_old_log_prob: bool = False,
        old_log_prob_from_new: bool = False,
    ) -> torch.Tensor | None:
        """Run one prepared diffusion micro-batch."""
        if self.custom_prepare_train_batch_func is not None:
            prepared = self.custom_prepare_train_batch_func(ctx, batch, pad_to_len=pad_to_len)
        else:
            prepared = prepare_flow_grpo_batch(ctx, batch, pad_to_len=pad_to_len)
        train_pipeline_config = self.train_pipeline_config
        forward_dtype = self._forward_dtype

        # Boundary dtypes are family policy; op interiors stay autocast-managed.
        latents_in, timesteps_in, (pos_cond_in, neg_cond_in, joint_cond_in) = apply_input_dtype_policy(
            train_pipeline_config.input_dtype_policy,
            latents=prepared.latents,
            timesteps=prepared.timesteps_for_model,
            conds=(prepared.pos_cond, prepared.neg_cond, prepared.joint_cond),
            default_dtype=forward_dtype,
        )

        def _compute_noise_pred() -> torch.Tensor:
            with torch.autocast("cuda", dtype=forward_dtype, enabled=forward_dtype != torch.float32):
                return train_pipeline_config.compute_noise_pred(
                    model=prepared.model,
                    latents_input=latents_in,
                    timesteps_input=timesteps_in,
                    pos_cond=pos_cond_in,
                    neg_cond=neg_cond_in,
                    joint_cond=joint_cond_in,
                    use_cfg=prepared.use_cfg,
                    cfg_batching=prepared.cfg_batching,
                    guidance_scale=prepared.guidance_scale,
                    true_cfg_scale=prepared.true_cfg_scale,
                )

        new_pred = _compute_noise_pred()

        ref_pred = None
        ref_mode = self.args.ref_mode
        if ref_mode != "none":
            if ref_mode == "ema":
                ref_ctx = self.ema_shadow.swap_in()
            else:
                ref_ctx = prepared.model.disable_adapter()
            with torch.no_grad(), ref_ctx:
                ref_pred = _compute_noise_pred().detach()

        if self.custom_loss_formula_func is not None:
            return self.custom_loss_formula_func(
                ctx,
                batch,
                prepared,
                new_pred=new_pred,
                ref_pred=ref_pred,
                metrics=metrics,
                write_old_log_prob=write_old_log_prob,
                old_log_prob_from_new=old_log_prob_from_new,
            )
        return flow_grpo_loss_formula(
            ctx,
            batch,
            prepared,
            new_pred=new_pred,
            ref_pred=ref_pred,
            metrics=metrics,
            write_old_log_prob=write_old_log_prob,
            old_log_prob_from_new=old_log_prob_from_new,
        )


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


def apply_lora(model: torch.nn.Module, args: Namespace, train_pipeline_config) -> torch.nn.Module:
    """Apply PEFT LoRA, leaving non-rank0 adapters uninitialized on meta."""
    from peft import LoraConfig, get_peft_model

    on_meta = dist.get_rank() != 0
    # Per-model fallback when --lora-target-modules is unset (runtime inference: depends on loaded pipeline).
    targets = args.lora_target_modules or train_pipeline_config.lora_target_modules
    init_lora_weight = args.lora_init_weights
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


def apply_fsdp2(
    model,
    parallel_plan,
    mesh=None,
    cpu_offload=False,
    args=None,
):
    """Apply FSDP2 per the model's FSDPParallelPlan.

    ``parallel_plan.param_dtype_patterns`` is matched against FQNs from ``model``. Each child
    ``fully_shard`` call receives exact FQNs relative to that child module, while
    parameters managed by the root call retain their root-relative FQNs.
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = parallel_plan.no_split_modules
    assert layer_cls_to_wrap is not None and len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [module for name, module in model.named_modules() if module.__class__.__name__ in layer_cls_to_wrap]

    param_dtype = parse_dtype_from_str(args.diffusion_forward_dtype)
    reduce_dtype = parse_dtype_from_str(args.fsdp_reduce_dtype)
    # A wrap entry may also be a module LIST — fully_shard can group several modules into one wrap
    # (one shared all-gather); today every wrap holds a single block.
    param_dtype_maps = compile_param_dtype_maps(
        model,
        modules,
        parallel_plan.param_dtype_patterns,
        param_dtype,
    )
    has_param_dtype_overrides = bool(any(param_dtype_maps.wrap_maps) or param_dtype_maps.root_map)
    param_dtype_policy_cls = None
    if has_param_dtype_overrides:
        from .monkey_patches.fsdp_param_dtype_patch import ParamDtypeMixedPrecisionPolicy, apply_param_dtype_map_patch

        apply_param_dtype_map_patch()
        param_dtype_policy_cls = ParamDtypeMixedPrecisionPolicy
    logger.info(
        f"FSDP: wrapping {len(modules)} modules of type {layer_cls_to_wrap}, "
        f"param_dtype={param_dtype}, reduce_dtype={reduce_dtype}, "
        f"param_dtype_overrides={param_dtype_maps.override_count} "
        f"({param_dtype_maps.override_numel:,} parameters)"
    )

    fsdp_kwargs = {
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    # input_dtype_policy owns boundary casts; autocast owns compute and keeps grad-ckpt recompute consistent.
    def make_mp_policy(param_dtype_map):
        if param_dtype_map:
            assert param_dtype_policy_cls is not None
            return param_dtype_policy_cls(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                cast_forward_inputs=False,
                param_dtype_map=param_dtype_map,
            )
        return MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=False,
        )

    for module, wrap_map in zip(modules, param_dtype_maps.wrap_maps, strict=True):
        fully_shard(
            module,
            mp_policy=make_mp_policy(wrap_map),
            **fsdp_kwargs,
        )

    fully_shard(
        model,
        mp_policy=make_mp_policy(param_dtype_maps.root_map),
        **fsdp_kwargs,
    )

    return model
