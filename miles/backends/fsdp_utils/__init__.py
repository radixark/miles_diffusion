"""Lazy entry points so CPU export tools do not import Ray/SGLang."""

import logging

__all__ = ["load_fsdp_args", "FSDPTrainRayActor"]


def __getattr__(name):
    if name == "FSDPTrainRayActor":
        from .actor import FSDPTrainRayActor

        return FSDPTrainRayActor
    if name == "load_fsdp_args":
        from .arguments import load_fsdp_args

        return load_fsdp_args
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


logging.getLogger().setLevel(logging.WARNING)
