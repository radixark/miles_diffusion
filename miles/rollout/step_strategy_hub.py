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


def sde_window(
    args: Namespace, sample: Sample, num_steps: int, seed: int
) -> tuple[list[int] | None, list[int] | None]:
    """flow_grpo-style random contiguous SDE window. Returns (sde=window, return=None)
    so sglang-d returns the full trajectory and log_probs; the trainer then slices
    to the window for loss / backprop. Keeping the full trajectory avoids the
    sglang-d-side trailing ``x_final`` aliasing issue when the window ends before
    the last denoising step."""
    window_size = args.diffusion_sde_window_size
    range_raw = args.diffusion_sde_window_range
    if range_raw:
        parts = [int(x) for x in str(range_raw).split(",")]
        lo, hi = parts[0], parts[1]
    else:
        lo, hi = 0, num_steps
    rng = np.random.default_rng(seed)
    start = int(rng.integers(lo, hi - window_size + 1))
    indices = list(range(start, start + window_size))
    return indices, None
