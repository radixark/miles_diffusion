"""Model backend: owns model-side behavior for the FSDP trainer.

Selected via ``--model-backend-path`` (miles custom-function style); the
family config declares the default. Four concerns, all properties of the
concrete modeling rather than of the training loop:

  - ``load_component`` / ``load_scheduler``: checkpoint -> model components and scheduler
  - ``enable_gradient_checkpointing``: how this model turns on grad ckpt
  - ``fsdp_no_split_modules``: which block classes FSDP wraps
  - ``sequence_parallel_plan`` / ``install_sequence_parallel_attention``:
    the model's SP declaration and attention integration

``MilesModelBackend`` loads native model packages (``models/<family>/``); set
``TrainPipelineConfig.model_package`` to the package import path.
``DiffusersModelBackend`` adapts the diffusers protocol for HF checkpoints.
"""

from __future__ import annotations

import abc
import functools
import importlib
import logging
from typing import Any

import torch
from diffusers import DiffusionPipeline

from .sequence_parallel.diffusers_dispatch import install_diffusers_usp_patch
from .sequence_parallel.plan import MILES_SP_PLAN_ATTR, SequenceParallelPlan

logger = logging.getLogger(__name__)


class BaseModelBackend(abc.ABC):
    """Contract consumed by the FSDP actor for model-family integration."""

    def __init__(self, train_pipeline_config):
        self.config = train_pipeline_config

    @abc.abstractmethod
    def enable_deterministic_attention(self, backend: str | None) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def load_component(
        self,
        component: str,
        args,
        *,
        master_dtype: torch.dtype,
        materialize_weights: bool,
    ) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def load_scheduler(self, args) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def fsdp_no_split_modules(self, model: torch.nn.Module) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def set_attention_backend(self, model: torch.nn.Module, backend: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def sequence_parallel_plan(self, model: torch.nn.Module) -> SequenceParallelPlan:
        raise NotImplementedError

    @abc.abstractmethod
    def install_sequence_parallel_attention(self, model: torch.nn.Module, parallel_state) -> None:
        raise NotImplementedError


class MilesModelBackend(BaseModelBackend):
    """Load native model families via the standardized ``models/<family>/`` package."""

    def __init__(self, train_pipeline_config):
        super().__init__(train_pipeline_config)
        pkg_path = getattr(train_pipeline_config, "model_package", None)
        if not pkg_path:
            raise ValueError(
                f"{type(train_pipeline_config).__name__} uses MilesModelBackend but "
                f"declares no model_package; set model_package to e.g. "
                f"'miles.backends.fsdp_utils.models.<family>'"
            )
        from .models.package import load_model_package

        self._pkg = load_model_package(pkg_path)

    def enable_deterministic_attention(self, backend: str | None) -> None:
        """Deterministic-mode hook: flash kernels need per-backend patching; native/math need none."""
        name = "" if backend is None else backend.lower()
        if "flash" in name or name.startswith("fa"):
            from .models.deterministic_attention import patch_modeling_flash_attention_deterministic

            patch_modeling_flash_attention_deterministic(self._pkg.modeling, name)

    def load_component(
        self,
        component: str,
        args,
        *,
        master_dtype: torch.dtype,
        materialize_weights: bool,
    ) -> torch.nn.Module:
        return self._pkg.loading.load_component(
            component,
            args,
            master_dtype=master_dtype,
            materialize_weights=materialize_weights,
        )

    def load_scheduler(self, args) -> Any:
        return self._pkg.modeling.load_scheduler(args)

    def enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
        self._pkg.modeling.enable_gradient_checkpointing(model)

    def fsdp_no_split_modules(self, model: torch.nn.Module) -> list[str]:
        return list(self._pkg.parallel_plan.FSDP_NO_SPLIT_MODULES)

    def set_attention_backend(self, model: torch.nn.Module, backend: str) -> None:
        self._pkg.attention.set_attention_backend(model, backend)

    def sequence_parallel_plan(self, model: torch.nn.Module) -> SequenceParallelPlan:
        return self._pkg.parallel_plan.sequence_parallel_plan(model)

    def install_sequence_parallel_attention(self, model: torch.nn.Module, parallel_state) -> None:
        install = getattr(self._pkg.parallel_plan, "install_sequence_parallel_attention", None)
        if install is None:
            raise NotImplementedError(
                f"{self._pkg.root.__name__}.parallel_plan does not provide " f"install_sequence_parallel_attention"
            )
        install(model, parallel_state)


class DiffusersModelBackend(BaseModelBackend):
    """Load trainable components from a diffusers pipeline checkpoint."""

    def __init__(self, train_pipeline_config):
        super().__init__(train_pipeline_config)

    def set_attention_backend(self, model: torch.nn.Module, backend: str) -> None:
        model.set_attention_backend(backend)

    def enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
        model.enable_gradient_checkpointing()

    def fsdp_no_split_modules(self, model: torch.nn.Module) -> list[str]:
        no_split_modules = getattr(model, "_no_split_modules", None)
        if not no_split_modules:
            raise ValueError(f"{model.__class__.__name__} declares no _no_split_modules for FSDP wrapping")
        return list(no_split_modules)

    def install_sequence_parallel_attention(self, model: torch.nn.Module, parallel_state) -> None:
        install_diffusers_usp_patch(model, parallel_state)

    def enable_deterministic_attention(self, backend: str | None) -> None:
        # Configure every installed kernel we know how to control. Native/SDPA
        # determinism is handled by torch.use_deterministic_algorithms; unsupported
        # opaque kernels are rejected by argument validation before actor startup.
        self._enable_deterministic_flash_attention()

    def _enable_deterministic_flash_attention(self) -> None:
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
        materialize_weights: bool,
    ) -> torch.nn.Module:
        model_cls = self._resolve_component_class(args, component)
        kwargs = {
            "subfolder": component,
            "torch_dtype": master_dtype,
            "low_cpu_mem_usage": materialize_weights,
        }

        # Non-rank0 loads with low_cpu_mem_usage=False so the ambient meta-device
        # context keeps params on meta; diffusers forbids that combination when the
        # class pins modules to fp32, so disable the pin for the duration (dtypes are
        # re-synced from rank0 afterwards, see ``sync_model_dtypes``).
        keep_in_fp32 = getattr(model_cls, "_keep_in_fp32_modules", None)
        if not materialize_weights and keep_in_fp32 is not None:
            model_cls._keep_in_fp32_modules = None
        try:
            return model_cls.from_pretrained(args.hf_checkpoint, **kwargs)
        finally:
            if not materialize_weights and keep_in_fp32 is not None:
                model_cls._keep_in_fp32_modules = keep_in_fp32

    def load_scheduler(self, args) -> Any:
        scheduler_cls = self._resolve_component_class(args, "scheduler")
        return scheduler_cls.from_pretrained(args.hf_checkpoint, subfolder="scheduler")

    @classmethod
    def _resolve_component_class(cls, args, component: str):
        """Resolve ``component``'s class from ``model_index.json``.

        Components load individually via ``cls.from_pretrained(subfolder=...)`` rather
        than through ``DiffusionPipeline.from_pretrained`` with the siblings passed as
        ``None``: pipelines that declare a component optional with a ``None`` default
        (e.g. ``WanPipeline.transformer``/``transformer_2``) drop it from
        ``expected_modules``, so the ``None`` is silently ignored and the sibling is
        loaded from disk anyway — on every rank, and with ``low_cpu_mem_usage=False``
        it also trips diffusers' ``_keep_in_fp32_modules`` guard.
        """
        config = DiffusionPipeline.load_config(args.hf_checkpoint)
        if component not in config:
            raise ValueError(f"pipeline {args.hf_checkpoint} has no component {component!r}")
        component_cls = cls._component_class(config[component])
        if component_cls is None:
            raise ValueError(
                f"cannot resolve the class for component {component!r} of {args.hf_checkpoint} "
                f"from spec {config[component]!r}; remote-code components are not supported"
            )
        return component_cls

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

    def sequence_parallel_plan(self, model: torch.nn.Module) -> SequenceParallelPlan:
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        plan = getattr(base, MILES_SP_PLAN_ATTR, None)
        if plan is not None:
            if not isinstance(plan, SequenceParallelPlan):
                raise TypeError(
                    f"{base.__class__.__name__}.{MILES_SP_PLAN_ATTR} must be a SequenceParallelPlan, "
                    f"got {type(plan).__name__}"
                )
            return plan

        boundaries = getattr(base, "_cp_plan", None)
        if not boundaries:
            raise ValueError(f"{base.__class__.__name__} declares no _cp_plan; sequence parallelism unavailable")
        plan = SequenceParallelPlan(
            boundaries=boundaries,
            num_attention_heads=base.config.num_attention_heads,
        )
        setattr(base, MILES_SP_PLAN_ATTR, plan)
        return plan
