"""Sequence-parallel plan: the per-family declaration and its interpreter.

A model family opts into SP through a ``SequenceParallelPlan`` (where the
sequence is sharded/gathered and its attention head count).
``apply_sequence_parallel`` consumes it: boundary hooks keep model outputs
full-sequence, so every parameter sees a partial gradient and loss/log_prob
code is untouched.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import torch
from diffusers.models._modeling_parallel import ContextParallelOutput

from .attention import gather_sequence, shard_sequence

MILES_SP_PLAN_ATTR = "_miles_sp_plan"


@dataclass(frozen=True)
class SequenceParallelPlan:
    """What one transformer family declares to run under SP.

    ``boundaries``: fqn -> ``ContextParallelInput``/``ContextParallelOutput``
    (the diffusers ``_cp_plan`` vocabulary) — where the sequence dim is sharded
    to S/sp and where full-sequence outputs are gathered back.
    Backends may attach one plan to a model instance as ``_miles_sp_plan``.
    The plan is topology-independent; ranks and process groups remain in the
    runtime parallel state passed to ``apply_sequence_parallel``.
    """

    boundaries: dict
    num_attention_heads: int

    def __post_init__(self) -> None:
        wildcards = [path for path in self.boundaries if "*" in path]
        if wildcards:
            raise ValueError(f"SequenceParallelPlan does not support wildcard boundary paths: {wildcards}")
        if self.num_attention_heads < 1:
            raise ValueError(f"num_attention_heads must be positive, got {self.num_attention_heads}")


def _split_if_expected(x, spec, parallel_state):
    if not isinstance(x, torch.Tensor):
        return x
    if spec.expected_dims is not None and x.ndim != spec.expected_dims:
        return x
    sp_mesh = parallel_state.get_mesh("sp")
    return shard_sequence(x, sp_mesh.get_local_rank(), sp_mesh.size(), dim=spec.split_dim)


def _resolve_submodule(root, path):
    # getattr-based walk: unlike get_submodule, it also traverses attribute
    # forwarding of wrappers such as peft's PeftModel.
    module = root
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _install_boundary_hooks(transformer, boundaries, parallel_state):
    """Install shard/gather hooks from the plan's boundary specs.

    ContextParallelInput entries split module inputs (or outputs when
    split_output=True); ContextParallelOutput entries gather module outputs.
    The gather's backward sums across SP, pairing with FSDP's 1/(dp*sp) mean.
    """
    sp_mesh = parallel_state.get_mesh("sp")
    for path, spec in boundaries.items():
        module = _resolve_submodule(transformer, path) if path else transformer

        if isinstance(spec, ContextParallelOutput):

            def gather_output(mod, args, output, _spec=spec):
                assert isinstance(output, torch.Tensor)
                return gather_sequence(
                    output,
                    sp_mesh.get_group(),
                    sp_mesh.get_local_rank(),
                    sp_mesh.size(),
                    dim=_spec.gather_dim,
                )

            module.register_forward_hook(gather_output)
            continue

        input_specs = {k: v for k, v in spec.items() if not v.split_output}
        output_specs = {k: v for k, v in spec.items() if v.split_output}

        if input_specs:
            # Wrappers like PeftModel expose forward(*args, **kwargs); the real
            # parameter names live on the base module.
            sig_module = module.get_base_model() if hasattr(module, "get_base_model") else module
            param_names = list(inspect.signature(sig_module.forward).parameters)
            missing = [k for k in input_specs if isinstance(k, str) and k not in param_names]
            if missing:
                raise ValueError(
                    f"boundary keys {missing} at '{path}' are not parameters of "
                    f"{type(sig_module).__name__}.forward"
                )

            def split_inputs(mod, args, kwargs, _specs=input_specs, _names=param_names):
                args = list(args)
                for key, s in _specs.items():
                    if isinstance(key, str) and key in kwargs:
                        kwargs[key] = _split_if_expected(kwargs[key], s, parallel_state)
                        continue
                    index = key if isinstance(key, int) else (_names.index(key) if key in _names else None)
                    if index is not None and index < len(args):
                        args[index] = _split_if_expected(args[index], s, parallel_state)
                return tuple(args), kwargs

            module.register_forward_pre_hook(split_inputs, with_kwargs=True)

        if output_specs:

            def split_outputs(mod, args, kwargs, output, _specs=output_specs):
                single = not isinstance(output, tuple)
                out = [output] if single else list(output)
                for index, s in _specs.items():
                    out[index] = _split_if_expected(out[index], s, parallel_state)
                return out[0] if single else tuple(out)

            module.register_forward_hook(split_outputs, with_kwargs=True)


def apply_sequence_parallel(transformer, parallel_state, plan, attention_installer: Callable):
    """Wire SP into one transformer per its plan: install the family's SP
    self-attention and the shard/gather boundary hooks. Call once per
    transformer after FSDP wrapping."""
    if plan.num_attention_heads % parallel_state.ulysses_degree != 0:
        raise ValueError(
            f"num_attention_heads({plan.num_attention_heads}) is not divisible by "
            f"ulysses_degree({parallel_state.ulysses_degree})"
        )
    attention_installer(transformer, parallel_state)
    _install_boundary_hooks(transformer, plan.boundaries, parallel_state)
