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


# Default candidate list, mirroring the local Flow-Factory wan22 dual recipe's
# `sde_steps` ({1,2,3} high window + all low steps {6..9} of the 10-step
# schedule). Override per run with --diffusion-sde-candidate-steps.
_FF_DUAL_SDE_CANDIDATES = [1, 2, 3, 6, 7, 8, 9]


def _sde_candidate_steps(args: Namespace) -> list[int]:
    raw = getattr(args, "diffusion_sde_candidate_steps", None)
    if raw is None:
        return _FF_DUAL_SDE_CANDIDATES
    return [int(step) for step in str(raw).split(",")]


def wan_ff_global_window(
    args: Namespace, sample: Sample, num_steps: int, seed: int
) -> tuple[list[int] | None, list[int] | None]:
    """Replicate Flow-Factory's per-epoch global SDE window for A/B comparison.

    FF (``FlowMatchEulerDiscreteSDEScheduler.current_sde_steps``) draws
    ``num_sde_steps`` indices once per epoch via
    ``torch.randperm(len(sde_steps), seed=epoch + train_seed)`` — every sample
    in the epoch shares the window, and phases are NOT balanced (a window can
    be all-high or all-low). Here ``epoch = group_index // rollout_batch_size``
    (group_index is 0-based and monotonic across the run), ``train_seed`` is
    ``--rollout-seed`` and the draw count is ``--diffusion-sde-window-size``
    (total, not per phase) — with both set to FF's values the two frameworks
    train the exact same window sequence epoch by epoch."""
    candidates = _sde_candidate_steps(args)
    window_size = int(args.diffusion_sde_window_size)
    if window_size <= 0:
        raise ValueError("wan_ff_global_window requires --diffusion-sde-window-size > 0")
    if window_size >= len(candidates):
        return sorted(candidates), None

    group_index = int(getattr(sample, "group_index", 0) or 0)
    epoch = group_index // int(args.rollout_batch_size)
    generator = torch.Generator().manual_seed(epoch + int(args.rollout_seed))
    selected = torch.randperm(len(candidates), generator=generator)[:window_size]
    return sorted(candidates[i] for i in selected.tolist()), None
