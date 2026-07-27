"""Model package protocol consumed by ``MilesModelBackend``.

Each native model family is a Python package under ``models/<family>/`` with:

  - ``loading`` — checkpoint resolution and ``load_component(...,
    materialize_weights=...)``; distributed rank selection stays outside the package
  - ``modeling`` — ``load_scheduler``, ``enable_gradient_checkpointing``,
    optional ``flash_attention_entrypoints`` / ``required_flash_kernel_label``
  - ``parallel_plan`` — ``FSDP_NO_SPLIT_MODULES``, ``sequence_parallel_plan``
    (and optional ``install_sequence_parallel_attention``)
  - ``attention`` — ``set_attention_backend``

``geometry`` and other train-forward helpers are family-specific and consumed by
the family ``TrainPipelineConfig``, not by the backend loader.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class ModelPackage:
    root: ModuleType
    loading: ModuleType
    modeling: ModuleType
    parallel_plan: ModuleType
    attention: ModuleType


def load_model_package(package_path: str) -> ModelPackage:
    root = importlib.import_module(package_path)
    return ModelPackage(
        root=root,
        loading=importlib.import_module(f"{package_path}.loading"),
        modeling=importlib.import_module(f"{package_path}.modeling"),
        parallel_plan=importlib.import_module(f"{package_path}.parallel_plan"),
        attention=importlib.import_module(f"{package_path}.attention"),
    )
