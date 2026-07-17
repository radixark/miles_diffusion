from __future__ import annotations

import abc
import functools
import importlib
import inspect
import logging
from typing import Any

import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)


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
    def load_component(
        self,
        component: str,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def load_scheduler(self, args) -> Any:
        raise NotImplementedError

    def enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
        model.enable_gradient_checkpointing()

    def fsdp_no_split_modules(self, model: torch.nn.Module) -> list[str]:
        return model._no_split_modules

    def set_attention_backend(self, model: torch.nn.Module, backend: str) -> None:
        raise NotImplementedError


class DiffusersModelBackend(ModelBackend):
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

    def load_component(
        self,
        component: str,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> torch.nn.Module:
        config = DiffusionPipeline.load_config(args.hf_checkpoint)
        if component not in config:
            raise ValueError(f"pipeline {args.hf_checkpoint} has no component {component!r}")

        pipeline = self._load_pipeline(args, config, component, master_dtype)
        model = getattr(pipeline, component)
        del pipeline
        return model

    def load_scheduler(self, args) -> Any:
        config = DiffusionPipeline.load_config(args.hf_checkpoint)
        pipeline = self._load_pipeline(args, config, "scheduler")
        scheduler = pipeline.scheduler
        del pipeline
        return scheduler

    @classmethod
    def _load_pipeline(cls, args, config: dict, component: str, master_dtype=None):
        skipped = {
            name: None for name, value in config.items() if name != component and isinstance(value, (list, tuple))
        }
        kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": dist.get_rank() == 0, **skipped}
        if master_dtype is not None:
            kwargs["torch_dtype"] = master_dtype

        model_cls = cls._component_class(config.get(component))
        keep_in_fp32 = getattr(model_cls, "_keep_in_fp32_modules", None)
        if dist.get_rank() != 0 and keep_in_fp32 is not None:
            # Diffusers otherwise forces low_cpu_mem_usage and materializes meta ranks.
            model_cls._keep_in_fp32_modules = None
        try:
            return DiffusionPipeline.from_pretrained(args.hf_checkpoint, **kwargs)
        finally:
            if dist.get_rank() != 0 and keep_in_fp32 is not None:
                model_cls._keep_in_fp32_modules = keep_in_fp32

    @staticmethod
    def _component_class(spec):
        if not isinstance(spec, (list, tuple)) or len(spec) != 2:
            return None
        library, class_name = spec
        if not library or not class_name:
            return None
        try:
            module = importlib.import_module(library)
        except ImportError:
            try:
                module = importlib.import_module(f"diffusers.pipelines.{library}")
            except ImportError:
                return None
        return getattr(module, class_name, None)


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

    def load_component(
        self,
        component: str,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> torch.nn.Module:
        from miles.backends.fsdp_utils.models.ltx2 import (
            load_ltx_transformer_for_train,
            resolve_transformer_checkpoint,
        )

        if component != "transformer":
            raise ValueError(f"LTX trains the single DiT ('transformer'); got {component!r}")
        checkpoint = resolve_transformer_checkpoint(str(args.diffusion_model))
        return load_ltx_transformer_for_train(checkpoint, device="cpu", dtype=master_dtype)

    def load_scheduler(self, args) -> Any:
        from miles.backends.fsdp_utils.models.ltx2 import build_ltx_train_scheduler

        return build_ltx_train_scheduler(args)

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
