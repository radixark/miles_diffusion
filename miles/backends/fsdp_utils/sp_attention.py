"""Sequence parallelism for diffusion DiTs: self-attention runs USP (sp_ops).

Each sp rank holds S/sp latent tokens; attention internally gathers to the full
sequence via Ulysses all-to-all (+Ring) and scatters back. A model family opts in
through a ``SequenceParallelPlan``; model outputs stay full-sequence, so every
parameter sees a partial gradient and loss/log_prob code is untouched.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import torch
from diffusers.models._modeling_parallel import ContextParallelOutput

from .sp_ops import gather_sequence, shard_sequence


@dataclass(frozen=True)
class SequenceParallelPlan:
    """What one transformer family declares to run under SP.

    ``boundaries``: fqn -> ``ContextParallelInput``/``ContextParallelOutput``
    (the diffusers ``_cp_plan`` vocabulary) — where the sequence dim is sharded
    to S/sp and where full-sequence outputs are gathered back.
    ``attention``: called with (transformer, parallel_state); routes the model's
    self-attention through ``sp_ops.usp_attention``.
    """

    boundaries: dict
    attention: Callable[[torch.nn.Module, object], None]
    num_attention_heads: int


def _split_if_expected(x, spec, parallel_state):
    if not isinstance(x, torch.Tensor):
        return x
    if spec.expected_dims is not None and x.ndim != spec.expected_dims:
        return x
    return shard_sequence(x, parallel_state.sp_rank, parallel_state.sp_size, dim=spec.split_dim)


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
    The gather is a differentiable all-gather, so backward stays partial-grad.
    """
    for path, spec in boundaries.items():
        module = _resolve_submodule(transformer, path) if path else transformer

        if isinstance(spec, ContextParallelOutput):

            def gather_output(mod, args, output, _spec=spec):
                assert isinstance(output, torch.Tensor)
                return gather_sequence(
                    output,
                    parallel_state.sp_group,
                    parallel_state.sp_rank,
                    parallel_state.sp_size,
                    dim=_spec.gather_dim,
                    sum_grad=parallel_state.fsdp_shard_mode == "dp_sp",
                )

            module.register_forward_hook(gather_output)
            continue

        input_specs = {k: v for k, v in spec.items() if not v.split_output}
        output_specs = {k: v for k, v in spec.items() if v.split_output}

        if input_specs:
            param_names = list(inspect.signature(module.forward).parameters)

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


def apply_sequence_parallel(transformer, parallel_state, plan):
    """Wire SP into one transformer per its plan: install the family's SP
    self-attention and the shard/gather boundary hooks. Call once per
    transformer after FSDP wrapping."""
    if plan.num_attention_heads % parallel_state.ulysses_degree != 0:
        raise ValueError(
            f"num_attention_heads({plan.num_attention_heads}) is not divisible by "
            f"ulysses_degree({parallel_state.ulysses_degree})"
        )
    plan.attention(transformer, parallel_state)
    _install_boundary_hooks(transformer, plan.boundaries, parallel_state)
