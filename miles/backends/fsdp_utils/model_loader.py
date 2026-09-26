from __future__ import annotations

import logging
import warnings
from argparse import Namespace
from contextlib import contextmanager

import torch
import torch.distributed as dist

from . import checkpoint
from .mixed_precision import compile_param_dtype_maps, parse_dtype_from_str
from .offload import offload_model
from .sequence_parallel.plan import apply_sequence_parallel

logger = logging.getLogger(__name__)


def load_fsdp_models(
    args: Namespace,
    model_backend,
    train_pipeline_config,
    parallel_state,
    *,
    checkpoint_path: str,
    lora_adapter_path: str | None = None,
    trainable: bool = False,
    cpu_offload: bool = False,
) -> dict[str, torch.nn.Module]:
    materialize_weights = dist.get_rank() == 0
    master_dtype = parse_dtype_from_str(args.fsdp_master_dtype)
    models = {}
    for component in args.update_weight_target_modules:
        with model_init_context(materialize_weights=materialize_weights):
            model = model_backend.load_component(
                component,
                checkpoint_path=checkpoint_path,
                master_dtype=master_dtype,
                materialize_weights=materialize_weights,
            )
        if args.fsdp_attention_backend is not None:
            model_backend.set_attention_backend(model, args.fsdp_attention_backend)
        if trainable and args.gradient_checkpointing:
            model_backend.enable_gradient_checkpointing(model)
        if lora_adapter_path is not None:
            subfolder = component if len(args.update_weight_target_modules) > 1 else None
            model = load_lora_adapter(model, lora_adapter_path, trainable=trainable, subfolder=subfolder)
        elif trainable and args.use_lora:
            model = apply_lora(model, args, train_pipeline_config)
        model.train(trainable)
        if not trainable:
            model.requires_grad_(False)

        if not materialize_weights and any(not parameter.is_meta for parameter in model.parameters()):
            raise RuntimeError(f"{component} did not honor meta initialization")
        checkpoint.sync_model_dtypes(model)
        full_state = model.state_dict() if materialize_weights else {}
        model = apply_fsdp2(
            model,
            model_backend.fsdp_parallel_plan(model),
            mesh=parallel_state.get_mesh("fsdp"),
            cpu_offload=cpu_offload,
            args=args,
        )
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
        if args.offload_train:
            offload_model(model)
        models[component] = model
    return models


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


def load_lora_adapter(model, adapter_path: str, *, trainable: bool, subfolder: str | None = None):
    from peft import PeftConfig, PeftModel, get_peft_model

    if dist.get_rank() == 0:
        return PeftModel.from_pretrained(
            model, adapter_path, subfolder=subfolder, is_trainable=trainable, autocast_adapter_dtype=False
        )
    config = PeftConfig.from_pretrained(adapter_path, subfolder=subfolder)
    config.inference_mode = not trainable
    return get_peft_model(model, config, low_cpu_mem_usage=True, autocast_adapter_dtype=False)


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
