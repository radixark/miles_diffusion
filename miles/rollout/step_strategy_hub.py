"""Library of functions that fill ``rollout_sde_step_indices`` /
``rollout_return_step_indices`` for one sglang-diffusion rollout request.

Each function has signature ``(args, sample, num_steps, seed) -> (sde, ret)``,
both ``list[int] | None``. A strategy owns both halves of the request:
``sde`` picks the steps sampled as SDE (``None`` = no SDE steps, a pure ODE
rollout), and ``ret`` picks the trajectory latents the engine ships back for
the trainer (``None`` = all steps).
Point ``--diffusion-step-strategy-path`` at any such function.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch

from miles.utils.types import Sample


def _sde_steps_to_latent_indices(indices: list[int]) -> list[int]:
    # The GRPO trainer consumes (x_i, x_{i+1}, log_prob_i) per SDE step, so the
    # engine must return every latent those steps touch: S U (S+1).
    return sorted({int(i) for i in indices} | {int(i) + 1 for i in indices})


def sde_window(
    args: Namespace, sample: Sample, num_steps: int, seed: int
) -> tuple[list[int] | None, list[int] | None]:
    """flow_grpo-style random contiguous SDE window; the engine returns all
    latents the window's steps touch (x_i and x_{i+1} for each step i)."""
    window_size = int(args.diffusion_num_sde_steps)
    if window_size <= 0:
        raise ValueError("sde_window requires --diffusion-num-sde-steps > 0")
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
        raise ValueError(f"--diffusion-num-sde-steps {window_size} does not fit in window range [{lo},{hi})")
    rng = np.random.default_rng(seed)
    start = int(rng.integers(lo, hi - window_size + 1))
    indices = list(range(start, start + window_size))
    return indices, _sde_steps_to_latent_indices(indices)


def epoch_global_random_choice(
    args: Namespace, sample: Sample, num_steps: int, seed: int
) -> tuple[list[int] | None, list[int] | None]:
    """Per-epoch global random SDE subset: draw ``--diffusion-num-sde-steps`` candidate
    steps at random once per epoch (every sample in the epoch shares them), returned
    in ascending step order (matches the wan2.2 recipe)."""
    candidates = _sde_candidate_steps(args, num_steps)
    num_sde_steps = int(args.diffusion_num_sde_steps)
    if num_sde_steps <= 0:
        raise ValueError("epoch_global_random_choice requires --diffusion-num-sde-steps > 0")
    if num_sde_steps >= len(candidates):
        indices = sorted(candidates)
    else:
        epoch = int(sample.group_index or 0) // int(args.rollout_batch_size)
        generator = torch.Generator().manual_seed(epoch + int(args.rollout_seed))
        selected = torch.randperm(len(candidates), generator=generator)[:num_sde_steps]
        indices = sorted(candidates[i] for i in selected.tolist())
    return indices, _sde_steps_to_latent_indices(indices)


def ode_and_return_last(
    args: Namespace, sample: Sample, num_steps: int, seed: int
) -> tuple[list[int] | None, list[int] | None]:
    """DiffusionNFT-style request: no SDE steps (pure ODE rollout) and only
    the final clean x0 shipped back."""
    return None, [num_steps]


def _sde_candidate_steps(args: Namespace, num_steps: int) -> list[int]:
    raw = args.diffusion_sde_candidate_steps
    if raw is None:
        raise ValueError(
            "epoch_global_random_choice requires --diffusion-sde-candidate-steps "
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
