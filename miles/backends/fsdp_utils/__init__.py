import logging

# Resolved lazily: importing the actor pulls in its whole training stack (and sglang for weight
# sync), which processes that only need fsdp_utils.arguments or a helper module do not want.
_TORCH_MEMORY_SAVER_AVAILABLE = True
_FSDP_AVAILABLE = True

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
