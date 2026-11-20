"""Model backend: owns model-side behavior for the FSDP trainer.

Selected via ``--model-backend-path`` (miles custom-function style); the
family config declares the default. Four concerns, all properties of the
concrete modeling rather than of the training loop:

  - ``load_component``: checkpoint -> model components
  - ``enable_gradient_checkpointing``: how this model turns on grad ckpt
  - ``fsdp_parallel_plan``: FSDP wrapping and parameter precision policy
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
import json
import logging
import os
from dataclasses import replace

import torch
from diffusers import DiffusionPipeline, ModularPipeline
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, HFValidationError, RemoteEntryNotFoundError

from .models.parallel_plan import FSDPParallelPlan
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
        *,
        checkpoint_path: str,
        master_dtype: torch.dtype,
        materialize_weights: bool,
    ) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def fsdp_parallel_plan(self, model: torch.nn.Module) -> FSDPParallelPlan:
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
        *,
        checkpoint_path: str,
        master_dtype: torch.dtype,
        materialize_weights: bool,
    ) -> torch.nn.Module:
        return self._pkg.loading.load_component(
            component,
            checkpoint_path=checkpoint_path,
            master_dtype=master_dtype,
            materialize_weights=materialize_weights,
        )

    def enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
        self._pkg.modeling.enable_gradient_checkpointing(model)

    def fsdp_parallel_plan(self, model: torch.nn.Module) -> FSDPParallelPlan:
        return self._pkg.parallel_plan.FSDP_PARALLEL_PLAN

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

    def fsdp_parallel_plan(self, model: torch.nn.Module) -> FSDPParallelPlan:
        if self.config is not None and self.config.model_family is not None:
            from .models.diffusers import load_fsdp_parallel_plan

            plan = load_fsdp_parallel_plan(self.config.model_family)
        else:
            plan = FSDPParallelPlan()
        if plan.no_split_modules is not None:
            return plan
        no_split_modules = getattr(model, "_no_split_modules", None)
        if not no_split_modules:
            raise ValueError(f"{model.__class__.__name__} declares no _no_split_modules for FSDP wrapping")
        return replace(plan, no_split_modules=tuple(no_split_modules))

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
        *,
        checkpoint_path: str,
        master_dtype: torch.dtype,
        materialize_weights: bool,
    ) -> torch.nn.Module:
        """Load one component of a Diffusers checkpoint by itself.

        Components load individually via ``cls.from_pretrained(subfolder=...)`` rather
        than through ``DiffusionPipeline.from_pretrained`` with the siblings passed as
        ``None``: pipelines that declare a component optional with a ``None`` default
        (e.g. ``WanPipeline.transformer``/``transformer_2``) drop it from
        ``expected_modules``, so the ``None`` is silently ignored and the sibling is
        loaded from disk anyway — on every rank, and with ``low_cpu_mem_usage=False``
        it also trips diffusers' ``_keep_in_fp32_modules`` guard.
        """
        model_cls, subfolder = self._resolve_component(checkpoint_path, component)
        kwargs = {
            "subfolder": subfolder,
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
            return model_cls.from_pretrained(checkpoint_path, **kwargs)
        finally:
            if not materialize_weights and keep_in_fp32 is not None:
                model_cls._keep_in_fp32_modules = keep_in_fp32

    @classmethod
    def _resolve_component(cls, checkpoint_path: str, component: str) -> tuple[type, str]:
        """``component``'s class and its folder under ``checkpoint_path``, from the checkpoint's index.

        A checkpoint that ships ``modular_model_index.json`` is read from it, as diffusers' modular loader does:
        its root ``model_index.json`` may describe another layout of the same weights (MiniMax-H3-Image's names the
        engine's native classes). The folder is the ``subfolder`` of the component's loading spec; an entry without
        a loading spec uses the component's name. The repository a loading spec names is not followed: weights come
        from ``checkpoint_path``.
        """
        index_name = ModularPipeline.config_name
        index = cls._read_modular_index(checkpoint_path)
        if index is None:
            index_name = DiffusionPipeline.config_name
            index = DiffusionPipeline.load_config(checkpoint_path)
        if component not in index:
            raise ValueError(f"{index_name} of {checkpoint_path} has no component {component!r}")
        spec = index[component]
        component_cls = cls._component_class(spec)
        if component_cls is None:
            raise ValueError(
                f"cannot resolve the class for component {component!r} from {index_name} of {checkpoint_path}: "
                f"spec {spec!r} names a class the installed library does not provide (remote-code components are "
                f"not supported)"
            )
        if len(spec) == 2:
            return component_cls, component
        loading_spec = spec[2] if isinstance(spec[2], dict) else {}
        subfolder = loading_spec.get("subfolder")
        if not isinstance(subfolder, str) or not subfolder:
            raise ValueError(
                f"component {component!r} names no subfolder in its spec {spec!r} in {index_name} of {checkpoint_path}"
            )
        if loading_spec.get("variant") is not None:
            raise ValueError(
                f"component {component!r} names the weight variant {loading_spec['variant']!r} in its spec in "
                f"{index_name} of {checkpoint_path}; loading a variant is untested, and loading without it could "
                f"read other weights"
            )
        return component_cls, subfolder

    @staticmethod
    def _read_modular_index(checkpoint_path: str) -> dict | None:
        """``modular_model_index.json`` of a checkpoint; None when it ships none, or the Hub cannot be asked and no copy is cached."""
        if os.path.isdir(checkpoint_path):
            index_file = os.path.join(checkpoint_path, ModularPipeline.config_name)
            if not os.path.isfile(index_file):
                return None
        else:
            try:
                index_file = hf_hub_download(checkpoint_path, ModularPipeline.config_name)
            except RemoteEntryNotFoundError:
                return None
            # no answer (offline, gated, not a repo id) and no cached copy: the load goes on from the root index
            except (EntryNotFoundError, HfHubHTTPError, HFValidationError) as error:
                logger.info(
                    "%s of %s could not be read (%s: %s); reading %s",
                    ModularPipeline.config_name,
                    checkpoint_path,
                    type(error).__name__,
                    error,
                    DiffusionPipeline.config_name,
                )
                return None
        with open(index_file) as f:
            return json.load(f)

    @staticmethod
    def _component_class(spec):
        # Classic pipelines write ``[library, class_name]``; modular ones append a
        # loading-spec dict (H3's MiniMaxH3ModularPipeline). Nothing else is valid.
        if not isinstance(spec, (list, tuple)) or len(spec) not in (2, 3):
            return None
        library, class_name = spec[0], spec[1]
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
