"""EMA shadow of trainable parameters for diffusion FSDP training."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


class EmaShadow:
    """EMA shadow of trainable parameters.

    ``shadow`` holds the current EMA. With ``keep_previous_ema=True``,
    ``previous_ema`` preserves the EMA from before the most recent ``update()``
    for the async trainer's prefetched-batch reference.
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        *,
        decay: float = 0.001,
        uprate: float = 0.001,
        uphold: float = 0.5,
        flat_steps: int = 0,
        keep_previous_ema: bool = False,
    ) -> None:
        self.decay = float(decay)
        self.uprate = float(uprate)
        self.uphold = float(uphold)
        self.flat_steps = int(flat_steps)
        self.step = 0
        self._swapped = False

        self.params = [p for p in parameters if p.requires_grad]
        if not self.params:
            raise ValueError("EmaShadow: model has no trainable parameters")
        self.shadow = [_local(p.detach()).clone() for p in self.params]
        self.previous_ema = [sh.clone() for sh in self.shadow] if keep_previous_ema else None

    def decay_at(self, t: int) -> float:
        if t <= self.flat_steps:
            return self.decay
        return float(min((t - self.flat_steps) * self.uprate, self.uphold))

    @torch.no_grad()
    def update(self) -> float:
        """theta_old <- delta * theta_old + (1 - delta) * theta."""
        if self._swapped:
            raise RuntimeError("EmaShadow.update called while swapped in")
        self.step += 1
        delta = self.decay_at(self.step)
        if self.previous_ema is not None:
            for previous_ema, current_ema in zip(self.previous_ema, self.shadow, strict=True):
                previous_ema.copy_(current_ema)
        for live, sh in zip(self.params, self.shadow, strict=True):
            sh.mul_(delta).add_(_local(live.detach()).to(sh.device), alpha=1.0 - delta)
        return delta

    def state_dict(self) -> dict:
        # Preserve FSDP shard metadata so DCP can restore on a different mesh.
        shadow = [
            (
                DTensor.from_local(
                    sh,
                    device_mesh=param.device_mesh,
                    placements=param.placements,
                    shape=param.shape,
                    stride=param.stride(),
                )
                if isinstance(param, DTensor)
                else sh
            )
            for param, sh in zip(self.params, self.shadow, strict=True)
        ]
        return {"shadow": shadow, "step": self.step}

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict) -> None:
        for sh, restored in zip(self.shadow, state_dict["shadow"], strict=True):
            sh.copy_(_local(restored))
        self.step = int(state_dict["step"])

    @contextmanager
    def swap_in(self, use_previous_ema: bool = False):
        """Temporarily use current EMA weights, or the snapshot before the last update."""
        buffers = self.previous_ema if use_previous_ema else self.shadow
        self._swap(buffers)
        self._swapped = True
        try:
            yield
        finally:
            self._swap(buffers)
            self._swapped = False

    @torch.no_grad()
    def _swap(self, buffers: list[torch.Tensor]) -> None:
        for live, sh in zip(self.params, buffers, strict=True):
            live_local = _local(live.data)
            tmp = live_local.clone()
            live_local.copy_(sh)
            sh.copy_(tmp)
