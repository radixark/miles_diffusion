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


# sgl-d's serving default for Wan2.2-T2V-A14B (Wan2_2_T2V_A14B_Config.flow_shift).
# The server applies this shift to ANY sigma schedule, including per-request
# custom sigmas — wan_request_sigmas() composes it out when overriding.
_WAN2_2_T2V_A14B_FLOW_SHIFT = 12.0
_WAN2_2_T2V_A14B_BOUNDARY_RATIO = 0.875
_WAN_NUM_TRAIN_TIMESTEPS = 1000


def _flow_shift_transform(sigmas: np.ndarray, shift: float) -> np.ndarray:
    """Exponential flow shift. Composes multiplicatively:
    shift_b(shift_a(x)) == shift_{a*b}(x)."""
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def wan_effective_shift(args: Namespace) -> float:
    """The shift the schedule actually runs with (request override or server default)."""
    flow_shift = getattr(args, "diffusion_flow_shift", None)
    return float(flow_shift) if flow_shift is not None else _WAN2_2_T2V_A14B_FLOW_SHIFT


def wan_request_sigmas(args: Namespace, num_steps: int) -> list[float] | None:
    """Sigma schedule to attach to rollout requests, or None for server default.

    The sgl-d scheduler applies its configured shift (12.0) on top of custom
    sigmas (diffusers convention), so to realize an effective shift S we send
    shift_{S/12}(raw): shift_12(shift_{S/12}(raw)) == shift_S(raw)."""
    flow_shift = getattr(args, "diffusion_flow_shift", None)
    if flow_shift is None:
        return None
    target = float(flow_shift)
    # Raw (pre-shift) grid exactly as the scheduler builds it: linspace between
    # the target-shifted train-sigma endpoints, in sigma units.
    endpoint_hi = _flow_shift_transform(np.array(1.0), target)
    endpoint_lo = _flow_shift_transform(np.array(1.0 / _WAN_NUM_TRAIN_TIMESTEPS), target)
    raw = np.linspace(endpoint_hi, endpoint_lo, num_steps, dtype=np.float64)
    sent = _flow_shift_transform(raw, target / _WAN2_2_T2V_A14B_FLOW_SHIFT)
    return [float(s) for s in sent]


def _wan2_2_euler_timesteps(
    num_steps: int,
    *,
    shift: float = _WAN2_2_T2V_A14B_FLOW_SHIFT,
    num_train_timesteps: int = _WAN_NUM_TRAIN_TIMESTEPS,
) -> np.ndarray:
    """Rebuild SGLang's FlowMatchEulerDiscreteScheduler timesteps for Wan2.2."""
    train_timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
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


def wan_high_window(
    args: Namespace, sample: Sample, num_steps: int, seed: int
) -> tuple[list[int] | None, list[int] | None]:
    """Sample an SDE window only from Wan2.2 high-noise steps."""
    window_size = int(args.diffusion_sde_window_size)
    if window_size <= 0:
        raise ValueError("wan_high_window requires --diffusion-sde-window-size > 0")

    boundary = _WAN2_2_T2V_A14B_BOUNDARY_RATIO * _WAN_NUM_TRAIN_TIMESTEPS
    timesteps = _wan2_2_euler_timesteps(num_steps, shift=wan_effective_shift(args))
    high_indices = [int(i) for i, timestep in enumerate(timesteps) if timestep >= boundary]

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


def wan_dual_window(
    args: Namespace, sample: Sample, num_steps: int, seed: int
) -> tuple[list[int] | None, list[int] | None]:
    """One SDE window per Wan2.2 phase (high + low noise), merged into one
    index list for dual-expert training.

    sgl-d gates SDE per step by list membership (``loop_step_index not in
    sde_step_indices`` → ODE), so the merged list being non-contiguous is fine.
    ``--diffusion-sde-window-size`` applies to each phase independently;
    ``--diffusion-sde-window-range`` (if set) restricts the high-noise phase
    only, mirroring its meaning in ``wan_high_window``."""
    window_size = int(args.diffusion_sde_window_size)
    if window_size <= 0:
        raise ValueError("wan_dual_window requires --diffusion-sde-window-size > 0")

    boundary = _WAN2_2_T2V_A14B_BOUNDARY_RATIO * _WAN_NUM_TRAIN_TIMESTEPS
    timesteps = _wan2_2_euler_timesteps(num_steps, shift=wan_effective_shift(args))
    high_indices = [int(i) for i, timestep in enumerate(timesteps) if timestep >= boundary]
    low_indices = [int(i) for i, timestep in enumerate(timesteps) if timestep < boundary]

    range_raw = getattr(args, "diffusion_sde_window_range", None)
    if range_raw:
        parts = [int(x) for x in str(range_raw).split(",")]
        lo, hi = parts[0], parts[1]
        high_indices = [i for i in high_indices if lo <= i < hi]

    rng = np.random.default_rng(seed)
    indices: list[int] = []
    for phase_name, phase_indices in (("high", high_indices), ("low", low_indices)):
        if len(phase_indices) < window_size:
            raise ValueError(
                f"Not enough Wan {phase_name}-noise steps for requested SDE window: "
                f"available={len(phase_indices)}, requested={window_size}, "
                f"num_steps={num_steps}, boundary={boundary}"
            )
        start = int(rng.integers(0, len(phase_indices) - window_size + 1))
        indices.extend(phase_indices[start : start + window_size])
    return sorted(indices), None
