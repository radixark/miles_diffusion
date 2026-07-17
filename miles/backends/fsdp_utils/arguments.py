import argparse
import dataclasses
from dataclasses import dataclass

import yaml


@dataclass
class FSDPArgs:
    # Optim
    optimizer: str = "adam"  # Optimizer type: "adam" (AdamW)
    lr: float = 2e-5
    lr_warmup_init: float = 0.0
    min_lr: float = 0.0
    lr_decay_style: str = "constant"
    lr_decay_iters: int | None = None
    lr_warmup_iters: int = 0
    lr_warmup_fraction: float | None = None
    lr_wsd_decay_iters: int | None = None
    lr_wsd_decay_style: str | None = None
    use_checkpoint_lr_scheduler: bool = True
    override_lr_scheduler: bool = False
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    # Aligned with flow_grpo (config/base.py:80) and PyTorch's Adam paper default.
    # Old miles default was 0.95 (LLM-pretraining convention) — switched here so
    # users who forget --adam-beta2 don't silently fall out of sync with flow_grpo
    # diffusion comparisons.
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    warmup_ratio: float = 0.03

    attn_implementation: str = "flash_attention_2"

    # DiT attention backend, passed to diffusers set_attention_backend (e.g.
    # "flash", "sage", "native"). None keeps the diffusers default.
    fsdp_attention_backend: str | None = None

    # Logging
    wandb_project: str = "miles-fsdp"
    wandb_run_name: str | None = None

    # Precision
    gradient_checkpointing: bool = False
    fp16: bool = False

    # FSDP configuration
    fsdp_state_dict_cpu_offload: bool = True  # If True, offload full state dict to CPU during collection.
    fsdp_cpu_offload: bool = (
        False  # If True, offload parameters, gradients, and optimizer states to CPU (optimizer runs on CPU)
    )
    fsdp_cpu_backend: str | None = (
        "gloo"  # CPU backend for FSDP CPU offload (e.g., "gloo"). Set to None to disable hybrid backend.
    )

    # Train-actor deterministic mode; see validate_attention_args for the backend
    # support matrix. Name kept identical to Megatron's.
    deterministic_mode: bool = False

    # Context Parallelism
    context_parallel_size: int = 1  # Context Parallelism size

    # YAML bookkeeping
    config: str | None = None


def parse_fsdp_cli(extra_args_provider=None):
    parser = argparse.ArgumentParser("FSDP SFT Training (miles)")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    for f in dataclasses.fields(FSDPArgs):
        if f.name == "config":
            continue

        # Handle union types like int | None, str | None, etc.
        if hasattr(f.type, "__args__"):  # Check if it's a Union type
            # For T | None, use T as the type
            non_none_types = [t for t in f.type.__args__ if t is not type(None)]
            arg_type = non_none_types[0] if non_none_types else str
        else:
            arg_type = f.type

        if arg_type is bool:
            parser.add_argument(f"--{f.name.replace('_', '-')}", action="store_true")
        else:
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=arg_type, default=f.default)

    if extra_args_provider is not None:
        parser = extra_args_provider(parser)
    args = parser.parse_args()
    return args


# Deterministic-mode attention support matrix — KEEP IN SYNC. torch's flag only
# governs torch-native ops, so an unlisted custom kernel runs nondeterministic
# silently under deterministic mode.
#   native / _native_*  (SDPA)      : torch's flag (needs warn_only=False)
#   flash* / _flash_3*  (flash-attn): patch deterministic= on (flag can't reach it)
#   sage / xformers / flex / aiter  : opaque to torch, no hook -> reject (validate)

# diffusers dispatches flash through these module globals (FA3 op reads them too).
_FLASH_ATTN_DISPATCH_FNS = (
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_3_func",
    "flash_attn_3_varlen_func",
)


def deterministic_capable_flash_fns():
    """diffusers flash entry points whose signature accepts a `deterministic` arg."""
    import inspect

    import diffusers.models.attention_dispatch as ad

    out = []
    for name in _FLASH_ATTN_DISPATCH_FNS:
        fn = getattr(ad, name, None)
        if fn is None:
            continue
        try:
            if "deterministic" in inspect.signature(fn).parameters:
                out.append(name)
        except (TypeError, ValueError):
            continue
    return out


def validate_attention_args(args):
    """Fail fast (driver-side, before any actor launches) on deterministic-mode misconfig."""
    if not getattr(args, "deterministic_mode", False):
        return
    backend = args.fsdp_attention_backend
    name = "" if backend is None else backend.lower()
    # torch SDPA (diffusers default / native): torch's global determinism covers it.
    # 'math' backends (torch SDPBackend.MATH semantics) are deterministic by construction.
    if backend is None or "native" in name or "math" in name:
        return
    # flash-attn: torch's global flag can't reach it; we patch its deterministic= on.
    if "flash" in name:
        if not deterministic_capable_flash_fns():
            raise RuntimeError(
                "deterministic_mode with a flash attention backend, but no diffusers "
                "flash entry point exposes a deterministic argument (is flash-attn "
                "installed and recent enough?)."
            )
        return
    # Anything else is a custom kernel we can neither cover via torch nor patch.
    raise ValueError(
        f"deterministic_mode cannot guarantee a deterministic backward for attention "
        f"backend '{backend}': it is a custom kernel opaque to "
        f"torch.use_deterministic_algorithms with no deterministic hook here. Use a "
        f"flash (flash/_flash_3) or native (SDPA) backend."
    )


def load_fsdp_args(extra_args_provider=None):
    args = parse_fsdp_cli(extra_args_provider)
    if args.config:
        with open(args.config) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if not hasattr(args, k):
                setattr(args, k, v)
    validate_attention_args(args)
    return args
