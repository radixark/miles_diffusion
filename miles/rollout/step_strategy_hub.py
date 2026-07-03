"""Library of functions that fill ``rollout_sde_step_indices`` /
``rollout_return_step_indices`` for one sglang-diffusion rollout request.

Each function has signature ``(args, sample, num_steps, seed) -> (sde, ret)``
where ``sde`` and ``ret`` are ``list[int] | None`` (``None`` = all steps).
Point ``--diffusion-step-strategy-path`` at any such function.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch

from miles.utils.types import Sample


def sde_window(
    args: Namespace, sample: Sample, num_steps: int, seed: int
) -> tuple[list[int] | None, list[int] | None]:
    """flow_grpo-style random contiguous SDE window. Returns (sde=window, return=None)
    so sglang-d returns the full trajectory and log_probs; the trainer then slices
    to the window for loss / backprop. Keeping the full trajectory avoids the
    sglang-d-side trailing ``x_final`` aliasing issue when the window ends before
    the last denoising step."""
    window_size = int(args.diffusion_sde_window_size)
    if window_size <= 0:
        raise ValueError("sde_window requires --diffusion-sde-window-size > 0")
    range_raw = args.diffusion_sde_window_range
    if range_raw:
        parts = [int(x) for x in str(range_raw).split(",")]
        if len(parts) != 2:
            raise ValueError(f"--diffusion-sde-window-range must be 'lo,hi', got {range_raw!r}")
        lo, hi = parts
    else:
        lo, hi = 0, num_steps
    if not 0 <= lo < hi <= num_steps:
        raise ValueError(f"--diffusion-sde-window-range [{lo},{hi}) out of range for a {num_steps}-step schedule")
    if window_size > hi - lo:
        raise ValueError(f"--diffusion-sde-window-size {window_size} does not fit in window range [{lo},{hi})")
    rng = np.random.default_rng(seed)
    start = int(rng.integers(lo, hi - window_size + 1))
    indices = list(range(start, start + window_size))
    return indices, None


def epoch_global_window(
    args: Namespace, sample: Sample, num_steps: int, seed: int
) -> tuple[list[int] | None, list[int] | None]:
    """Per-epoch global SDE window: draw ``--diffusion-sde-window-size``
    indices from the candidate set once per epoch, so every sample in an
    epoch trains the same window."""
    candidates = _sde_candidate_steps(args, num_steps)
    window_size = int(args.diffusion_sde_window_size)
    if window_size <= 0:
        raise ValueError("epoch_global_window requires --diffusion-sde-window-size > 0")
    if window_size >= len(candidates):
        return sorted(candidates), None

    group_index = int(getattr(sample, "group_index", 0) or 0)
    epoch = group_index // int(args.rollout_batch_size)
    generator = torch.Generator().manual_seed(epoch + int(args.rollout_seed))
    selected = torch.randperm(len(candidates), generator=generator)[:window_size]
    return sorted(candidates[i] for i in selected.tolist()), None


def _sde_candidate_steps(args: Namespace, num_steps: int) -> list[int]:
    raw = getattr(args, "diffusion_sde_candidate_steps", None)
    if raw is None:
        raise ValueError(
            "epoch_global_window requires --diffusion-sde-candidate-steps "
            "(e.g. '1,2,3'); which indices are valid depends on the "
            "num-steps/flow-shift schedule; there is no safe default"
        )
    candidates = [int(step) for step in str(raw).split(",")]
    out_of_range = [step for step in candidates if not 0 <= step < num_steps]
    if out_of_range:
        raise ValueError(
            f"--diffusion-sde-candidate-steps {out_of_range} out of range for a {num_steps}-step schedule"
        )
    return candidates
