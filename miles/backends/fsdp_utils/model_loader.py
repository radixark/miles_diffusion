from __future__ import annotations

import logging
import warnings
from argparse import Namespace
from contextlib import contextmanager
from functools import partial

import torch
import torch.distributed as dist

from . import checkpoint
from .device_move import move_model
from .mixed_precision import compile_param_dtype_maps, parse_dtype_from_str
from .sequence_parallel.plan import apply_sequence_parallel

logger = logging.getLogger(__name__)


def load_fsdp_models(
    args: Namespace,
    model_backend,
    train_pipeline_config,
    parallel_state,
    *,
    checkpoint_path: str,
    lora_adapter_paths: list[str] | None = None,
    trainable: bool = False,
    cpu_offload: bool = False,
) -> dict[str, torch.nn.Module]:
    models = {}
    for component_index, component in enumerate(args.update_weight_target_modules):
        lora_adapter_path = None if lora_adapter_paths is None else lora_adapter_paths[component_index]
        build = partial(
            build_component,
            args,
            model_backend,
            train_pipeline_config,
            component,
            checkpoint_path=checkpoint_path,
            lora_adapter_path=lora_adapter_path,
            trainable=trainable,
        )
        # Every rank reads its own shards from the safetensors files; else rank 0 loads and broadcasts.
        has_lora = lora_adapter_path is not None or (trainable and args.use_lora)
        read_per_rank = not has_lora or lora_needs_no_base_weights(args, lora_adapter_path)
        safetensors_files = None
        if read_per_rank:
            safetensors_files = model_backend.safetensors_files(component, checkpoint_path=checkpoint_path)
        # Each rank reads its own adapter config and files, and the collectives below depend on the answer.
        if not checkpoint.holds_on_every_rank(safetensors_files is not None):
            safetensors_files = None
        materialize_weights = safetensors_files is None and dist.get_rank() == 0
        rng_state = torch.get_rng_state()
        lora_on_every_rank = has_lora and safetensors_files is not None
        model = build(materialize_weights=materialize_weights, lora_on_every_rank=lora_on_every_rank)
        if safetensors_files is not None and not checkpoint.holds_on_every_rank(
            checkpoint.safetensors_hold_state(model, safetensors_files)
        ):
            if has_lora or dist.get_rank() == 0:
                # A constructor may draw random numbers: the second build draws what the first one did.
                torch.set_rng_state(rng_state)
                materialize_weights = dist.get_rank() == 0
                model = build(materialize_weights=materialize_weights, lora_on_every_rank=False)
            safetensors_files = None
        if dist.get_rank() == 0:
            source = "rank 0 loads and broadcasts"
            if safetensors_files is not None:
                source = (
                    f"every rank reads {len(safetensors_files)} safetensors files of {safetensors_files[0].parent}"
                )
            logger.info(f"FSDP: {component} weights of {checkpoint_path}: {source}")
        checkpoint.sync_model_dtypes(model)
        full_state = model.state_dict() if materialize_weights else {}
        model = apply_fsdp2(
            model,
            model_backend.fsdp_parallel_plan(model),
            mesh=parallel_state.get_mesh("fsdp"),
            cpu_offload=cpu_offload,
            args=args,
        )
        if safetensors_files is not None:
            checkpoint.load_safetensors_to_fsdp(model, safetensors_files, cpu_offload=cpu_offload)
        else:
            checkpoint.broadcast_full_state_to_fsdp(model, full_state, cpu_offload=cpu_offload)
        del full_state
        train_pipeline_config.postprocess_model_after_materialize(model)
        if parallel_state.get_optional_mesh("sp") is not None:
            apply_sequence_parallel(
                model,
                parallel_state,
                model_backend.sequence_parallel_plan(model),
                model_backend.install_sequence_parallel_attention,
            )
        finish_fsdp_lazy_init(model)
        # A natively CPU-offloaded model never sleeps, so its buffers stay on the GPU.
        if args.offload_train and not cpu_offload:
            move_model(model, "cpu")
        models[component] = model
    return models


LORA_INITS_WITHOUT_BASE_WEIGHTS = ("gaussian", "kaiming-uniform")


def lora_needs_no_base_weights(args: Namespace, lora_adapter_path: str | None) -> bool:
    """Whether PEFT builds the adapter without reading the base weights (PiSSA, OLoRA, LoftQ... decompose them)."""
    if lora_adapter_path is None:
        return args.lora_init_weights in LORA_INITS_WITHOUT_BASE_WEIGHTS
    from peft import PeftConfig

    return PeftConfig.from_pretrained(lora_adapter_path).init_lora_weights in (True, False, "gaussian")


def build_component(
    args: Namespace,
    model_backend,
    train_pipeline_config,
    component: str,
    *,
    checkpoint_path: str,
    lora_adapter_path: str | None,
    trainable: bool,
    materialize_weights: bool,
    lora_on_every_rank: bool,
) -> torch.nn.Module:
    """``lora_on_every_rank``: every rank builds the adapter's values as rank 0 does, on a base without weights.

    It is for a model that has an adapter, and for all ranks at once: the build then runs a collective.
    """
    with model_init_context(materialize_weights=materialize_weights):
        model = model_backend.load_component(
            component,
            checkpoint_path=checkpoint_path,
            master_dtype=parse_dtype_from_str(args.fsdp_master_dtype),
            materialize_weights=materialize_weights,
        )
    if args.fsdp_attention_backend is not None:
        model_backend.set_attention_backend(model, args.fsdp_attention_backend)
    if trainable and args.gradient_checkpointing:
        model_backend.enable_gradient_checkpointing(model)
    materialize_lora = lora_on_every_rank or dist.get_rank() == 0
    with base_storage_for_lora(model, enabled=lora_on_every_rank):
        if lora_adapter_path is not None:
            model = load_lora_adapter(model, lora_adapter_path, trainable=trainable, materialize=materialize_lora)
        elif trainable and args.use_lora:
            model = apply_lora(model, args, train_pipeline_config, materialize=materialize_lora)
    model.train(trainable)
    if not trainable:
        model.requires_grad_(False)
    if not materialize_weights and any(
        not parameter.is_meta
        for name, parameter in model.named_parameters()
        if not (lora_on_every_rank and checkpoint.is_lora_param_name(name))
    ):
        raise RuntimeError(f"{component} did not honor meta initialization")
    return model


@contextmanager
def base_storage_for_lora(model: torch.nn.Module, *, enabled: bool):
    """Give a meta base a CPU stand-in while PEFT builds an adapter on it, from rank 0's generator state.

    PEFT moves a new adapter to its base layer's device, and meta would drop the adapter's values. It reads the
    base layer's device and dtype only, so each base parameter stands in as one uninitialized element expanded to
    its shape: a full ``torch.empty`` is written through under ``--deterministic-mode`` (the NaN fill), a copy of
    the model on every rank. PEFT's initialization draws, and only rank 0 has run it so far: the other ranks draw
    the same values from rank 0's generator state and keep their own.
    """
    if not enabled:
        yield
        return
    rng_state = torch.get_rng_state()
    torch.set_rng_state(checkpoint.rank0_rng_state())
    replace_base_parameters(model, lambda parameter: torch.empty((), dtype=parameter.dtype).expand(parameter.shape))
    try:
        yield
    finally:
        replace_base_parameters(model, lambda parameter: parameter.data.to("meta"))
        if dist.get_rank() != 0:
            torch.set_rng_state(rng_state)


def replace_base_parameters(model: torch.nn.Module, data_of) -> None:
    # A meta parameter cannot take CPU data in place (nor the reverse), so each one is registered anew.
    for module_name, module in model.named_modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            if not checkpoint.is_lora_param_name(f"{module_name}.{name}"):
                data = data_of(parameter)
                module.register_parameter(name, torch.nn.Parameter(data, requires_grad=parameter.requires_grad))


def finish_fsdp_lazy_init(model: torch.nn.Module) -> None:
    """Run FSDP's one-time lazy init now instead of at the first forward.

    It fixes each group's gradient dtype from the parameters that require grad at that moment;
    a first forward with PEFT adapters disabled would see nothing trainable and leave it unset.
    """
    model._get_fsdp_state()._lazy_init()


@contextmanager
def model_init_context(*, materialize_weights: bool):
    if materialize_weights:
        with torch.device("cpu"):
            yield
        return
    from accelerate import init_empty_weights

    # Some constructors compute buffer values and cannot initialize them on meta.
    with init_empty_weights(include_buffers=False), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"for .*: copying from a non-meta parameter in the checkpoint to a meta parameter.*",
        )
        yield


def load_lora_adapter(model, adapter_path: str, *, trainable: bool, materialize: bool):
    from peft import PeftConfig, PeftModel, get_peft_model

    if materialize:
        return PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable, autocast_adapter_dtype=False)
    config = PeftConfig.from_pretrained(adapter_path)
    config.inference_mode = not trainable
    return get_peft_model(model, config, low_cpu_mem_usage=True, autocast_adapter_dtype=False)


def apply_lora(
    model: torch.nn.Module, args: Namespace, train_pipeline_config, *, materialize: bool
) -> torch.nn.Module:
    """Apply PEFT LoRA; without ``materialize`` the adapters stay uninitialized on meta."""
    from peft import LoraConfig, get_peft_model

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
            init_lora_weights=init_lora_weight if materialize else False,
        ),
        low_cpu_mem_usage=not materialize,
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
