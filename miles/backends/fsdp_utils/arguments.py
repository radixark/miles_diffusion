import argparse
import dataclasses
import json
import logging
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)


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
    # No effect in this backend: a saved LR-scheduler state is always restored on --load.
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

    # diffusers set_attention_backend value; None keeps the default. Ring attention accepts the RING_KERNELS subset.
    fsdp_attention_backend: str | None = None

    # Logging
    wandb_project: str = "miles-fsdp"

    # Precision
    gradient_checkpointing: bool = False

    # FSDP configuration
    fsdp_cpu_offload: bool = (
        False  # If True, offload parameters, gradients, and optimizer states to CPU (optimizer runs on CPU)
    )
    fsdp_cpu_backend: str | None = (
        "gloo"  # CPU backend for FSDP CPU offload (e.g., "gloo"). Set to None to disable hybrid backend.
    )
    # Hybrid sharding: parameter replica count; dp_shard uses the ranks left by this and SP.
    dp_replicate_size: int = 1

    # Train-actor deterministic mode; see validate_attention_args for the backend
    # support matrix. Name kept identical to Megatron's.
    deterministic_mode: bool = False

    # Sequence Parallelism (USP = Ulysses x Ring)
    sequence_parallel_size: int = 1
    # 0=auto: ulysses fills sp; ring = sp // ulysses. Ring degrees > 1 run on
    # torch's experimental (private) ring-attention implementation and require
    # torch >= 2.11 (the CI image's pin).
    ulysses_degree: int = 0

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
    return parser.parse_args(), parser


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
    """Fail fast (driver-side, before any actor launches) on deterministic-mode misconfig.

    Enforces the support matrix documented above _FLASH_ATTN_DISPATCH_FNS.
    """
    if not args.deterministic_mode:
        return
    backend = args.fsdp_attention_backend
    name = "" if backend is None else backend.lower()
    if backend is None or "native" in name or "math" in name:
        return
    if "flash" in name:
        if not deterministic_capable_flash_fns():
            raise RuntimeError(
                "deterministic_mode with a flash attention backend, but no diffusers "
                "flash entry point exposes a deterministic argument (is flash-attn "
                "installed and recent enough?)."
            )
        return
    raise ValueError(
        f"deterministic_mode cannot guarantee a deterministic backward for attention "
        f"backend '{backend}': it is a custom kernel opaque to "
        f"torch.use_deterministic_algorithms with no deterministic hook here. Use a "
        f"flash (flash/_flash_3) or native (SDPA) backend."
    )


def validate_sp_config(world_size, sequence_parallel_size, ulysses_degree=0):
    if sequence_parallel_size < 1:
        raise ValueError(f"sequence_parallel_size must be positive, got {sequence_parallel_size}")
    if ulysses_degree < 0:
        raise ValueError(f"ulysses_degree must be non-negative, got {ulysses_degree}")
    resolved_ulysses_degree = ulysses_degree or sequence_parallel_size
    if sequence_parallel_size % resolved_ulysses_degree:
        raise ValueError(
            f"sequence_parallel_size({sequence_parallel_size}) is not divisible by "
            f"ulysses_degree({resolved_ulysses_degree})"
        )
    if world_size % sequence_parallel_size:
        raise ValueError(
            f"world_size({world_size}) is not divisible by sequence_parallel_size({sequence_parallel_size})"
        )


def validate_hybrid_shard_args(args) -> None:
    """Fail fast on a dp_replicate_size the world size or SP degree cannot honor."""
    world_size = args.actor_num_gpus_per_node * args.actor_num_nodes
    validate_sp_config(world_size, args.sequence_parallel_size, args.ulysses_degree)
    if args.dp_replicate_size < 1:
        raise ValueError(f"dp_replicate_size must be at least 1, got {args.dp_replicate_size}")
    if world_size % (args.dp_replicate_size * args.sequence_parallel_size):
        raise ValueError(
            f"world_size({world_size}) is not divisible by dp_replicate_size({args.dp_replicate_size}) "
            f"* sequence_parallel_size({args.sequence_parallel_size})"
        )


def validate_sp_args(args) -> None:
    """Validate finalized SP topology and ring-kernel arguments on the driver.

    Model-instance constraints such as ``_cp_plan`` availability and attention
    dispatch compatibility are checked later, after the model is loaded.
    """
    from .sequence_parallel.attention import RING_KERNELS

    validate_sp_config(
        args.actor_num_gpus_per_node * args.actor_num_nodes,
        args.sequence_parallel_size,
        args.ulysses_degree,
    )
    if args.sequence_parallel_size == 1:
        return
    # ulysses SP by default
    ulysses_degree = args.ulysses_degree or args.sequence_parallel_size
    ring_degree = args.sequence_parallel_size // ulysses_degree
    if ring_degree > 1 and args.fsdp_attention_backend not in RING_KERNELS:
        raise ValueError(
            f"--fsdp-attention-backend {args.fsdp_attention_backend!r} cannot drive ring attention; "
            f"supported: {sorted(k for k in RING_KERNELS if k is not None)}"
        )


def load_fsdp_args(extra_args_provider=None):
    args, parser = parse_fsdp_cli(extra_args_provider)
    if args.config:
        with open(args.config) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if not hasattr(args, k):
                setattr(args, k, v)
    # Before any check or derived value, so a YAML value is treated like a command-line one.
    if args.custom_config_path:
        with open(args.custom_config_path) as f:
            apply_custom_config(args, parser, yaml.safe_load(f) or {})
    validate_attention_args(args)
    return args


_BOOLEAN_ACTIONS = (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)


def apply_custom_config(args, parser: argparse.ArgumentParser, data: dict) -> None:
    """Set each ``--custom-config-path`` key, an argument's dest, to its value parsed as the command line would.

    Args:
        args: the parsed namespace, updated in place.
        parser: the parser that produced ``args``; its actions give each dest's type, choices and nargs.
        data: the YAML mapping of dest to value.
    """
    actions = {action.dest: action for action in parser._actions if action.option_strings}
    for key, value in data.items():
        if key not in actions:
            raise ValueError(
                f"--custom-config-path sets {key!r}, which is no argument's dest (an argument's dest is its "
                "flag name with underscores, e.g. global_batch_size); put other keys in --config"
            )
        parsed = _parse_like_command_line(actions[key], value)
        if parsed != getattr(args, key):
            logger.info(f"--custom-config-path sets {key}={parsed!r} (was {getattr(args, key)!r})")
        setattr(args, key, parsed)


def _parse_like_command_line(action: argparse.Action, value):
    flag = action.option_strings[0]
    if isinstance(action, _BOOLEAN_ACTIONS):
        if not isinstance(value, bool):
            raise ValueError(f"--custom-config-path {action.dest}: {flag} takes a YAML boolean, got {value!r}")
        return value
    if value is None:
        return None
    if action.nargs in ("+", "*") or isinstance(action.nargs, int):
        if not isinstance(value, list):
            raise ValueError(f"--custom-config-path {action.dest}: {flag} takes a YAML list, got {value!r}")
        if isinstance(action.nargs, int) and len(value) != action.nargs:
            raise ValueError(f"--custom-config-path {action.dest}: {flag} takes {action.nargs} values, got {value!r}")
        parsed = [_parse_one(action, item) for item in value]
        for item in parsed:
            _check_choice(action, item)
        return parsed
    parsed = _parse_one(action, value)
    _check_choice(action, parsed)
    return parsed


def _parse_one(action: argparse.Action, value):
    """The value a command line would give: the YAML value written as its command-line text, through the type."""
    if action.type is json.loads and not isinstance(value, str):
        text = json.dumps(value)
    elif isinstance(value, dict):
        # A per-stream flag's command-line form, e.g. visual=3.0,audio=1.0.
        text = ",".join(f"{name}={item}" for name, item in value.items())
    elif isinstance(value, list):
        text = ",".join(str(item) for item in value)
    else:
        text = str(value)
    if action.type is None:
        return text
    try:
        return action.type(text)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as error:
        raise ValueError(
            f"--custom-config-path {action.dest}: {value!r} is not a valid {action.option_strings[0]}"
        ) from error


def _check_choice(action: argparse.Action, value) -> None:
    if action.choices is not None and value not in action.choices:
        raise ValueError(
            f"--custom-config-path {action.dest}: {value!r} is not one of {list(action.choices)} "
            f"for {action.option_strings[0]}"
        )
