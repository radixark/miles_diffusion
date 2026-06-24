"""Library of functions that fill ``rollout_sde_step_indices`` /
``rollout_return_step_indices`` for one sglang-diffusion rollout request.

Each function has signature ``(args, sample, num_steps, seed) -> (sde, ret)``
where ``sde`` and ``ret`` are ``list[int] | None`` (``None`` = all steps).
Strategies that must match trainer-rollout (``ltx_sde_candidates``) accept
``rollout_id`` as a keyword argument — see ``miles.rollout.sglang_diffusion_rollout``.
All strategies in this hub should accept ``*, rollout_id=0`` for a uniform call site.

Point ``--diffusion-step-strategy-path`` at any such function.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch

from miles.utils.types import Sample


def _normalize_sde_step_candidates(candidates, num_steps: int) -> list[int] | None:
    if candidates is None or candidates == "":
        return None
    if isinstance(candidates, str):
        candidates = [int(x.strip()) for x in candidates.split(",") if x.strip()]
    else:
        candidates = [int(x) for x in candidates]
    invalid = [step for step in candidates if step < 0 or step >= num_steps]
    if invalid:
        raise ValueError(f"sde_step_candidates must be in [0, {num_steps}), got {invalid}")
    return list(dict.fromkeys(candidates))


def ltx_sde_candidates(
    args: Namespace,
    sample: Sample,
    num_steps: int,
    seed: int,
    *,
    rollout_id: int = 0,
) -> tuple[list[int] | None, list[int] | None]:
    """verl-omni / trainer-rollout SDE step pick: ``--ltx-num-sde-steps`` random
    draws from ``--ltx-sde-step-candidates``, keyed by ``rollout_seed + rollout_id``.

    Uses ``torch.randperm`` (not numpy) so the chosen indices match
    ``miles.rollout.ltx_rollout._select_sde_step_set`` bit-for-bit.

    Non-candidate steps run as deterministic Euler in sglang; only listed indices
    inject SDE noise and contribute log_probs — same as trainer-rollout.
    """
    del sample, seed  # trainer keys off rollout_id, not per-sample generation seed
    candidates = _normalize_sde_step_candidates(getattr(args, "ltx_sde_step_candidates", None), num_steps)
    if candidates is None:
        raise ValueError("ltx_sde_candidates requires --ltx-sde-step-candidates " "(e.g. 0,1,2,3,4,5,6,7,8,9)")
    num_sde = int(getattr(args, "ltx_num_sde_steps", 0) or len(candidates))
    num_sde = min(max(num_sde, 1), len(candidates))
    rng_seed = int(getattr(args, "rollout_seed", 42)) + int(rollout_id)
    g = torch.Generator().manual_seed(rng_seed)
    selected = torch.randperm(len(candidates), generator=g)[:num_sde]
    indices = [candidates[i] for i in selected.tolist()]
    return indices, None


def sde_window(
    args: Namespace, sample: Sample, num_steps: int, seed: int, *, rollout_id: int = 0
) -> tuple[list[int] | None, list[int] | None]:
    """flow_grpo-style random contiguous SDE window. Returns (sde=window, return=None)
    so sglang-d returns the full trajectory and log_probs; the trainer then slices
    to the window for loss / backprop. Keeping the full trajectory avoids the
    sglang-d-side trailing ``x_final`` aliasing issue when the window ends before
    the last denoising step."""
    del sample, rollout_id
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
