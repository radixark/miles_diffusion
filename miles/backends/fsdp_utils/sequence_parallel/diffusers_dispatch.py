"""Diffusers dispatcher integration for USP self-attention.

Wraps the modeling module's ``dispatch_attention_fn`` so self-attention call
sites (which pass ``_parallel_config`` per upstream convention) route through
``attention.usp_attention``; cross-attention and the model's own processors run
untouched.
"""

import functools
import sys

from .attention import usp_attention


class _USPDispatchConfig:
    """Marker consumed by the wrapped dispatch_attention_fn; models pass it
    through ``_parallel_config`` for self-attention call sites only."""

    def __init__(self, parallel_state):
        self.ulysses_group = parallel_state.ulysses_group
        self.ring_group = parallel_state.ring_group


def _wrap_dispatch(module):
    original = module.dispatch_attention_fn
    if getattr(original, "_miles_usp_wrapped", False):
        return

    @functools.wraps(original)
    def dispatch(query, key, value, *args, parallel_config=None, **kwargs):
        if not isinstance(parallel_config, _USPDispatchConfig):
            return original(query, key, value, *args, parallel_config=parallel_config, **kwargs)
        attn_mask = args[0] if args else kwargs.get("attn_mask")
        if attn_mask is not None:
            raise ValueError("USP self-attention does not support attention masks")
        if (len(args) > 1 and args[1]) or kwargs.get("dropout_p"):
            raise ValueError("USP self-attention requires dropout_p=0 (per-rank RNG streams diverge)")
        if (len(args) > 2 and args[2]) or kwargs.get("is_causal"):
            raise ValueError("USP self-attention does not support is_causal yet")
        if (len(args) > 3 and args[3] is not None) or kwargs.get("scale") is not None:
            raise ValueError("USP self-attention uses the default 1/sqrt(d) scale")
        if parallel_config.ring_group is not None:
            if kwargs.get("backend") is not None:
                raise ValueError(
                    "An explicit attention backend is supported with pure Ulysses only, not Ring attention"
                )
            return usp_attention(query, key, value, parallel_config.ulysses_group, parallel_config.ring_group)

        def local_attention_fn(local_query, local_key, local_value):
            return original(
                local_query,
                local_key,
                local_value,
                *args,
                parallel_config=None,
                **kwargs,
            )

        return usp_attention(
            query,
            key,
            value,
            parallel_config.ulysses_group,
            local_attention_fn=local_attention_fn,
        )

    dispatch._miles_usp_wrapped = True
    module.dispatch_attention_fn = dispatch


def apply_dispatch_sp_attention(transformer, parallel_state):
    """Default SP attention: intercept the model module's dispatch_attention_fn
    so self-attention call sites (which pass ``_parallel_config`` per upstream
    convention) route through usp_attention; the model's own processors and
    cross-attention stay untouched."""
    base = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
    # fully_shard swizzles the class (FSDP<Name>, defined in torch's fsdp
    # module); the modeling module that imported dispatch_attention_fn is
    # found through the MRO.
    module = next(
        (
            mod
            for cls in type(base).__mro__
            if (mod := sys.modules.get(cls.__module__)) is not None and hasattr(mod, "dispatch_attention_fn")
        ),
        None,
    )
    if module is None:
        raise ValueError(
            f"{type(base).__name__} does not route attention through diffusers' "
            "dispatch_attention_fn; its SequenceParallelPlan must provide a custom attention_installer"
        )
    _wrap_dispatch(module)
    config = _USPDispatchConfig(parallel_state)
    for processor in base.attn_processors.values():
        processor._parallel_config = config
