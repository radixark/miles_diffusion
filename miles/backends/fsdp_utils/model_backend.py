"""Model backend: owns model-side behavior for the FSDP trainer.

Selected via ``--model-backend-path`` (miles custom-function style); the
family config declares the default. The concerns are all properties of the
concrete modeling rather than of the training loop:

  - ``build_models_and_scheduler``: checkpoint -> ``({component: model}, scheduler)``
    with parameters on the meta device (``--fsdp-load-mode stream``)
  - ``stream_load_weights``: fill one already-sharded component's weights,
    reading the checkpoint on rank 0 only
  - ``load_models_and_scheduler``: legacy path — full weights on every rank
    (``--fsdp-load-mode legacy``)
  - ``enable_gradient_checkpointing``: how this model turns on grad ckpt
  - ``fsdp_no_split_modules``: which block classes FSDP wraps

Defaults implement the diffusers protocol (see ``models/__init__.py``); a
native model overrides methods here instead of retrofitting its instances.
"""

from __future__ import annotations

import abc
import functools
import inspect
import json
import logging
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)

_SAFETENSORS_INDEX = "diffusion_pytorch_model.safetensors.index.json"
_SAFETENSORS_SINGLE = "diffusion_pytorch_model.safetensors"

class ModelBackend(abc.ABC):
    def __init__(self, train_pipeline_config):
        self.config = train_pipeline_config

    def enable_deterministic_attention(self, backend: str | None) -> None:
        """Deterministic-mode hook: flash kernels need per-backend patching; native/math need none."""
        name = "" if backend is None else backend.lower()
        if "flash" in name or name.startswith("fa"):
            self._enable_deterministic_flash_attention(name)

    def _enable_deterministic_flash_attention(self, name: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has no deterministic hook for flash backend "
            f"{name!r}; use a native/math attention backend under deterministic mode."
        )

    @abc.abstractmethod
    def load_models_and_scheduler(
        self,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[dict[str, torch.nn.Module], Any]:
        """Legacy path: return ``({component: model}, scheduler)`` fully
        materialized on CPU, on every rank."""

    def build_models_and_scheduler(
        self,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[dict[str, torch.nn.Module], Any]:
        """Return ``({component: model}, scheduler)`` with component
        parameters on the meta device (buffers real, computed by __init__).
        Weights are filled in later via ``stream_load_weights``, after FSDP
        sharding, so no rank ever holds a full unsharded copy."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement meta-device init; run with --fsdp-load-mode legacy"
        )

    def stream_load_weights(
        self,
        model: torch.nn.Module,
        component: str,
        args,
        *,
        master_dtype: torch.dtype,
        key_map: Callable[[str], str] | None = None,
    ) -> None:
        """Fill ``model``'s weights from the checkpoint. ``model`` is already
        FSDP-sharded and materialized; rank 0 reads, everyone receives."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement streaming load; run with --fsdp-load-mode legacy"
        )

    def enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
        """Turn on grad checkpointing; default = the diffusers protocol method."""
        model.enable_gradient_checkpointing()

    def fsdp_no_split_modules(self, model: torch.nn.Module) -> list[str]:
        """Block class names FSDP wraps; default = the model's own declaration."""
        return model._no_split_modules

    def set_attention_backend(self, model: torch.nn.Module, backend: str) -> None:
        raise NotImplementedError


class DiffusersModelBackend(ModelBackend):
    """Load trainable components from a diffusers pipeline checkpoint."""

    def set_attention_backend(self, model: torch.nn.Module, backend: str) -> None:
        model.set_attention_backend(backend)

    def _enable_deterministic_flash_attention(self, name: str) -> None:
        """Patch diffusers flash entrypoints to deterministic=True (backward only; idempotent)."""
        import diffusers.models.attention_dispatch as ad

        from .arguments import deterministic_capable_flash_fns

        names = deterministic_capable_flash_fns()
        for fn_name in names:
            setattr(ad, fn_name, functools.partial(getattr(ad, fn_name), deterministic=True))
        logger.info("Enabled deterministic flash attention backward for: %s", ", ".join(names))

    def load_models_and_scheduler(
        self,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[dict[str, torch.nn.Module], Any]:
        pipeline = DiffusionPipeline.from_pretrained(
            args.hf_checkpoint,
            torch_dtype=master_dtype,
            trust_remote_code=True,
            text_encoder=None,
            vae=None,
            tokenizer=None,
        )
        raw_models: dict[str, torch.nn.Module] = {}
        for component in args.update_weight_target_modules:
            sub_model = getattr(pipeline, component, None)
            if sub_model is None:
                raise ValueError(
                    f"--update-weight-target-module: pipeline {args.hf_checkpoint} " f"has no component '{component}'"
                )
            raw_models[component] = sub_model
        scheduler = pipeline.scheduler
        del pipeline
        return raw_models, scheduler

    def build_models_and_scheduler(
        self,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[dict[str, torch.nn.Module], Any]:
        import diffusers
        from accelerate import init_empty_weights

        root = self._resolve_checkpoint_dir(args)
        with open(os.path.join(root, "model_index.json")) as f:
            model_index = json.load(f)

        raw_models: dict[str, torch.nn.Module] = {}
        for component in args.update_weight_target_modules:
            if component not in model_index:
                raise ValueError(
                    f"--update-weight-target-module: pipeline {args.hf_checkpoint} " f"has no component '{component}'"
                )
            library, cls_name = model_index[component]
            if library != "diffusers":
                raise ValueError(
                    f"component '{component}' comes from library '{library}'; only diffusers "
                    f"components support meta init — run with --fsdp-load-mode legacy"
                )
            cls = getattr(diffusers, cls_name, None)
            if cls is None:
                raise ValueError(
                    f"component class '{cls_name}' not found in diffusers (remote code?); "
                    f"run with --fsdp-load-mode legacy"
                )
            config = cls.load_config(root, subfolder=component)
            # include_buffers=False: parameters land on meta (zero memory), but
            # buffers and plain tensor attributes are computed for real by
            # __init__ (Wan's rope freq tables, Qwen's pos/neg freqs).  This is
            # the same context diffusers' own from_pretrained builds under, so
            # buffer values match the legacy path bit-exactly.
            with init_empty_weights(include_buffers=False):
                model = cls.from_config(config)
            raw_models[component] = model.to(master_dtype)

        _, scheduler_cls_name = model_index["scheduler"]
        scheduler = getattr(diffusers, scheduler_cls_name).from_pretrained(root, subfolder="scheduler")
        return raw_models, scheduler

    def stream_load_weights(
        self,
        model: torch.nn.Module,
        component: str,
        args,
        *,
        master_dtype: torch.dtype,
        key_map: Callable[[str], str] | None = None,
    ) -> None:
        from safetensors.torch import load_file
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        component_dir = os.path.join(self._resolve_checkpoint_dir(args), component)
        index_path = os.path.join(component_dir, _SAFETENSORS_INDEX)
        rank = dist.get_rank()
        if rank == 0:
            if os.path.exists(index_path):
                with open(index_path) as f:
                    files = sorted(set(json.load(f)["weight_map"].values()))
            elif os.path.exists(os.path.join(component_dir, _SAFETENSORS_SINGLE)):
                files = [_SAFETENSORS_SINGLE]
            else:
                raise FileNotFoundError(f"no safetensors checkpoint under {component_dir}")
        else:
            files = None
        # rank0 discovered the file list; everyone must loop the same number of
        # times since set_model_state_dict is collective.
        obj = [files]
        dist.broadcast_object_list(obj, src=0)
        files = obj[0]

        options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False)
        loaded_keys: set[str] = set()
        for fname in files:
            partial: dict[str, torch.Tensor] = {}
            if rank == 0:
                # load_file mmaps; peak CPU ~= one file (+ a dtype-cast copy)
                for key, tensor in load_file(os.path.join(component_dir, fname)).items():
                    partial[key_map(key) if key_map is not None else key] = tensor.to(master_dtype)
            set_model_state_dict(model, partial, options=options)
            key_obj = [set(partial.keys())]
            dist.broadcast_object_list(key_obj, src=0)
            loaded_keys |= key_obj[0]
            del partial

        # strict=False enables the per-file partial loads above, so missing
        # weights would pass silently — verify full coverage ourselves.
        expected = {name for name, _ in model.named_parameters() if "lora_" not in name}
        missing = expected - loaded_keys
        if missing:
            raise RuntimeError(
                f"{component}: {len(missing)} parameters not covered by checkpoint, " f"e.g. {sorted(missing)[:3]}"
            )
        logger.info("[FSDP] %s: streamed %d files, %d tensors", component, len(files), len(loaded_keys))

    @staticmethod
    def _resolve_checkpoint_dir(args) -> str:
        ckpt = args.hf_checkpoint
        if os.path.isdir(ckpt):
            return ckpt
        from huggingface_hub import snapshot_download

        # every rank needs the configs/index; weight files only on rank 0,
        # where stream_load_weights reads them.
        patterns = ["model_index.json", "*/*.json"]
        if not dist.is_initialized() or dist.get_rank() == 0:
            patterns += [f"{component}/*" for component in args.update_weight_target_modules]
        return snapshot_download(ckpt, allow_patterns=patterns)


class LTXModelBackend(ModelBackend):
    """Native LTX-2 loading via ltx_core; model instances stay unmodified."""

    def _enable_deterministic_flash_attention(self, name: str) -> None:
        # ltx_core binds flash kernels via module globals; wrap them with deterministic=True.
        import ltx_core.model.transformer.attention as ltx_attn

        patched: list[str] = []
        f3 = ltx_attn.flash_attn_interface
        if f3 is not None and "deterministic" in inspect.signature(f3.flash_attn_func).parameters:
            f3.flash_attn_func = functools.partial(f3.flash_attn_func, deterministic=True)
            patched.append("flash_attention_3")
        f4 = ltx_attn.flash_attn_4_func
        if f4 is not None and "deterministic" in inspect.signature(f4).parameters:
            ltx_attn.flash_attn_4_func = functools.partial(f4, deterministic=True)
            patched.append("flash_attention_4")
        wanted = "flash_attention_3" if ("3" in name) else "flash_attention_4" if ("4" in name) else None
        if wanted is not None and wanted not in patched:
            raise RuntimeError(
                f"deterministic_mode: ltx_core backend {name!r} maps to {wanted}, but its kernel "
                f"is unavailable or exposes no deterministic argument (patched: {patched or None}). "
                f"Use --fsdp-attention-backend math for a deterministic backward."
            )
        logger.info("Enabled deterministic ltx_core flash attention backward for: %s", ", ".join(patched))

    def load_models_and_scheduler(
        self,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[dict[str, torch.nn.Module], Any]:
        from miles.backends.fsdp_utils.models.ltx2 import (
            build_ltx_train_scheduler,
            load_ltx_transformer_for_train,
            resolve_transformer_checkpoint,
        )

        modules = list(args.update_weight_target_modules)
        if modules != ["transformer"]:
            raise ValueError(f"LTX trains the single DiT ('transformer'); got {modules}")
        # TODO: meta-init on non-rank-0 before multi-node runs (every rank loads the full weights).
        checkpoint = resolve_transformer_checkpoint(str(args.diffusion_model))
        model = load_ltx_transformer_for_train(checkpoint, device="cpu", dtype=master_dtype)
        return {"transformer": model}, build_ltx_train_scheduler(args)

    def enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
        model.set_gradient_checkpointing(True)

    def fsdp_no_split_modules(self, model: torch.nn.Module) -> list[str]:
        return ["BasicAVTransformerBlock"]

    def set_attention_backend(self, model: torch.nn.Module, backend: str) -> None:
        # ltx_core selects attention via AttentionFunction (not diffusers' set_attention_backend(str));
        # map the flag and reuse ltx_core's own module op to swap it on every Attention submodule.
        from ltx_core.loader.attention_ops import set_attention_module_op
        from ltx_core.model.transformer.attention import AttentionFunction, MaskedAttentionFunction

        aliases = {
            "fa3": "FLASH_ATTENTION_3",
            "fa4": "FLASH_ATTENTION_4",
            "sdpa": "PYTORCH",
            "native": "PYTORCH",
            "math": "SDPA_MATH",
            "sdpa_math": "SDPA_MATH",
        }
        name = aliases.get(backend.strip().lower(), backend.strip().upper())
        if name not in AttentionFunction.__members__:
            valid = ", ".join(m.name.lower() for m in AttentionFunction)
            raise ValueError(
                f"LTX --fsdp-attention-backend='{backend}' is not an ltx_core backend; "
                f"choose one of {{{valid}}} (aliases: fa3, fa4, sdpa)."
            )
        masked = MaskedAttentionFunction[name] if name in MaskedAttentionFunction.__members__ else None
        set_attention_module_op(attention=AttentionFunction[name], masked_attention=masked).mutator(model)
