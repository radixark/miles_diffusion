import importlib

from ..parallel_plan import FSDPParallelPlan


def load_cp_plan(model_family: str) -> dict | None:
    """The family's boundary declaration, or None to fall back to the model's own ``_cp_plan``."""
    module = importlib.import_module(f"{__name__}.{model_family}.parallel_plan")
    return module.CP_PLAN


def load_fsdp_parallel_plan(model_family: str) -> FSDPParallelPlan:
    module = importlib.import_module(f"{__name__}.{model_family}.parallel_plan")
    plan = module.FSDP_PARALLEL_PLAN
    if not isinstance(plan, FSDPParallelPlan):
        raise TypeError(f"{module.__name__}.FSDP_PARALLEL_PLAN must be an FSDPParallelPlan")
    return plan


__all__ = ["load_cp_plan", "load_fsdp_parallel_plan"]
