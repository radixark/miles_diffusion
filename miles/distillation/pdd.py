"""Parallel Decoding Distillation (arXiv:2607.26004, equations 9--15).

The grid may increase (paper time) or decrease (diffusers sigma). The teacher
must return derivatives with respect to that same coordinate. Predictions use
shape [interval, batch, ...]; interval indices are absolute, not block-relative.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def validate_grid(grid: torch.Tensor) -> None:
    if grid.ndim != 1 or grid.numel() < 2 or not torch.isfinite(grid).all():
        raise ValueError("grid must be a finite 1D tensor with at least two points")
    delta = grid.diff()
    if not ((delta > 0).all() or (delta < 0).all()):
        raise ValueError("grid must be strictly monotonic")


class ParallelLinear(nn.Module):
    """N independent output projections initialized from one pretrained layer."""

    def __init__(self, source: nn.Linear, num_intervals: int):
        super().__init__()
        if num_intervals < 1:
            raise ValueError("num_intervals must be positive")
        self.weight = nn.Parameter(source.weight.detach().unsqueeze(0).repeat(num_intervals, 1, 1))
        self.bias = (
            nn.Parameter(source.bias.detach().unsqueeze(0).repeat(num_intervals, 1))
            if source.bias is not None
            else None
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        result = torch.einsum("...i,koi->k...o", hidden, self.weight)
        if self.bias is not None:
            result = result + self.bias.reshape(self.bias.shape[0], *([1] * (hidden.ndim - 1)), -1)
        return result


@torch.no_grad()
def teacher_mean_velocity(teacher: Callable, x: torch.Tensor, time, delta, solver: str = "midpoint") -> torch.Tensor:
    """Teacher closure includes conditioning/CFG; never build a teacher gradient graph."""
    if solver not in {"euler", "midpoint"}:
        raise ValueError("solver must be euler or midpoint")

    def call(state, t):
        velocity = teacher(state, t)
        if velocity.shape != state.shape:
            raise ValueError("teacher output must match the latent shape exactly")
        return velocity

    x = x.detach()
    velocity = call(x, time)
    if solver == "midpoint":
        velocity = call(x + 0.5 * delta * velocity, time + 0.5 * delta)
    return velocity.detach()


def block_update(x: torch.Tensor, predictions: torch.Tensor, grid: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Accumulate signed interval lengths in fp32 (or fp64 for fp64 inputs)."""
    if predictions.shape != (len(grid) - 1, *x.shape):
        raise ValueError("predictions must have shape [num_intervals, *latent.shape]")
    if not 0 <= start <= end < len(grid):
        raise ValueError("invalid block bounds")
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    h = grid.diff()[start:end].to(device=x.device, dtype=dtype)
    return x.to(dtype) + torch.einsum("k,k...->...", h, predictions[start:end].to(dtype))


def interval_loss(predictions: torch.Tensor, selected: int, target: torch.Tensor) -> torch.Tensor:
    """Per-sample squared error of the selected head against the teacher's mean velocity."""
    error = (predictions[selected].float() - target.float()).square()
    return error.flatten(1).mean(1)
