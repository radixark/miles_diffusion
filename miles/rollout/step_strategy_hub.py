"""Library of functions that fill ``rollout_sde_step_indices`` /
``rollout_return_step_indices`` for one sglang-diffusion rollout request.

Each function has signature ``(args, sample, num_steps, seed) -> (sde, ret)``
where ``sde`` and ``ret`` are ``list[int] | None`` (``None`` = all steps).
Point ``--diffusion-step-strategy-path`` at any such function.
"""
from __future__ import annotations

from argparse import Namespace

import numpy as np

from miles.utils.types import Sample


_WAN2_2_T2V_A14B_FLOW_SHIFT = 12.0
_WAN2_2_T2V_A14B_BOUNDARY_RATIO = 0.875
_WAN_NUM_TRAIN_TIMESTEPS = 1000


def _wan2_2_euler_timesteps(
    num_steps: int,
    *,
    shift: float = _WAN2_2_T2V_A14B_FLOW_SHIFT,
    num_train_timesteps: int = _WAN_NUM_TRAIN_TIMESTEPS,
) -> np.ndarray:
    """Rebuild SGLang's FlowMatchEulerDiscreteScheduler timesteps for Wan2.2."""
    train_timesteps = np.linspace(
        1, num_train_timesteps, num_train_timesteps, dtype=np.float32
    )[::-1].copy()
    train_sigmas = train_timesteps / float(num_train_timesteps)
    train_sigmas = shift * train_sigmas / (1 + (shift - 1) * train_sigmas)

    timesteps = np.linspace(
        train_sigmas[0] * num_train_timesteps,
        train_sigmas[-1] * num_train_timesteps,
        num_steps,
        dtype=np.float32,
    )
    sigmas = timesteps / float(num_train_timesteps)
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return sigmas * float(num_train_timesteps)


def sde_window(
    args: Namespace, sample: Sample, num_steps: int, seed: int
) -> tuple[list[int] | None, list[int] | None]:
    """flow_grpo-style random contiguous SDE window. Returns (sde=window, return=None)
    so sglang-d returns the full trajectory and log_probs; the trainer then slices
    to the window for loss / backprop. Keeping the full trajectory avoids the
    sglang-d-side trailing ``x_final`` aliasing issue when the window ends before
    the last denoising step."""
    window_size = int(args.diffusion_sde_window_size)
    range_raw = getattr(args, "diffusion_sde_window_range", None)
    if range_raw:
        parts = [int(x) for x in str(range_raw).split(",")]
        lo, hi = parts[0], parts[1]
    else:
        lo, hi = 0, num_steps
    rng = np.random.default_rng(seed)
    start = int(rng.integers(lo, hi - window_size + 1))
    indices = list(range(start, start + window_size))
    return indices, None


def wan_high_window(
    args: Namespace, sample: Sample, num_steps: int, seed: int
) -> tuple[list[int] | None, list[int] | None]:
    """Sample an SDE window only from Wan2.2 high-noise steps."""
    window_size = int(args.diffusion_sde_window_size)
    if window_size <= 0:
        raise ValueError("wan_high_window requires --diffusion-sde-window-size > 0")

    boundary = _WAN2_2_T2V_A14B_BOUNDARY_RATIO * _WAN_NUM_TRAIN_TIMESTEPS
    timesteps = _wan2_2_euler_timesteps(num_steps)
    high_indices = [
        int(i) for i, timestep in enumerate(timesteps) if timestep >= boundary
    ]

    range_raw = getattr(args, "diffusion_sde_window_range", None)
    if range_raw:
        parts = [int(x) for x in str(range_raw).split(",")]
        lo, hi = parts[0], parts[1]
        high_indices = [i for i in high_indices if lo <= i < hi]

    if len(high_indices) < window_size:
        raise ValueError(
            "Not enough Wan high-noise steps for requested SDE window: "
            f"available={len(high_indices)}, requested={window_size}, "
            f"num_steps={num_steps}, boundary={boundary}"
        )

    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(high_indices) - window_size + 1))
    return high_indices[start : start + window_size], None
