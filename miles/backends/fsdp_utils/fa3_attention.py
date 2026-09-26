"""Autograd-preserving dense FA3 adapter for the pinned Diffusers dispatcher.

Diffusers wraps FA3 in a forward-only custom op and drops the deterministic
argument before that call. Use FA3's public autograd function directly, both
with and without deterministic mode. Miles handles pure Ulysses outside this
local attention call; Diffusers context parallelism and ring are not supported.
"""

import inspect
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from diffusers.models._modeling_parallel import ParallelConfig


def install_diffusers_fa3_attention(*, deterministic: bool = False) -> None:
    """Install before model execution/compilation, once per train actor.

    The Diffusers registry is process-wide. Reinstalling with a different value
    changes the forced mode for all `_flash_3` users in that process; concurrent
    models requiring different forced modes must use separate processes.
    """
    import diffusers.models.attention_dispatch as ad

    kernel = ad.flash_attn_3_func
    try:
        parameters = inspect.signature(kernel).parameters if callable(kernel) else {}
    except (TypeError, ValueError):
        parameters = {}
    required = {"deterministic", "num_splits", "return_attn_probs"}
    if not required.issubset(parameters):
        raise RuntimeError(
            "Miles _flash_3 requires flash_attn_interface.flash_attn_func with "
            "deterministic, num_splits, and return_attn_probs arguments. Install "
            "the compatible FA3 build in the GPU image; FA2 is not a substitute."
        )

    registry = ad._AttentionBackendRegistry
    backend = ad.AttentionBackendName._FLASH_3
    current = registry._backends[backend]
    if getattr(current, "_miles_fa3_deterministic", None) is deterministic:
        return

    force_deterministic = deterministic

    # Register through the public decorator of the private registry: replacing
    # only _backends leaves the dispatcher's cached keyword signature stale.
    @registry.register(backend, constraints=registry._constraints.get(backend), supports_context_parallel=False)
    def fa3_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        return_lse: bool = False,
        deterministic: bool = False,
        num_splits: int = 1,
        _parallel_config: "ParallelConfig | None" = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if attn_mask is not None:
            raise ValueError("Miles _flash_3 does not support attn_mask")
        if dropout_p != 0.0:
            raise ValueError("Miles _flash_3 requires dropout_p=0")
        if _parallel_config is not None:
            raise ValueError(
                "Miles _flash_3 requires local attention; use Miles pure Ulysses for sequence parallelism"
            )
        if num_splits != 1:
            raise ValueError("Miles _flash_3 requires num_splits=1")

        result = ad.flash_attn_3_func(
            q=query,
            k=key,
            v=value,
            softmax_scale=scale,
            causal=is_causal,
            num_splits=1,
            deterministic=force_deterministic or deterministic or torch.are_deterministic_algorithms_enabled(),
            return_attn_probs=return_lse,
        )
        if return_lse:
            out, lse, *_ = result
            # FA3 backward does not differentiate LSE. Expose it as a diagnostic,
            # in the [batch, sequence, heads] layout expected by Diffusers.
            return out, lse.detach().transpose(1, 2)
        return result

    fa3_attention._miles_fa3_deterministic = deterministic
