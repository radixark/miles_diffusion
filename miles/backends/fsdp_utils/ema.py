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
    """EMA shadow of trainable parameters."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        *,
        decay: float = 0.001,
        uprate: float = 0.001,
        uphold: float = 0.5,
        flat_steps: int = 0,
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
        for live, sh in zip(self.params, self.shadow, strict=True):
            sh.mul_(delta).add_(_local(live.detach()).to(sh.device), alpha=1.0 - delta)
        return delta

    @torch.no_grad()
    def to(self, device, *, non_blocking: bool = False, pin_memory: bool = False) -> None:
        device = torch.device(device)
        if pin_memory and device.type != "cpu":
            raise ValueError("pin_memory requires a CPU destination")
        for index, shadow in enumerate(self.shadow):
            if pin_memory:
                if shadow.device.type == "cpu" and shadow.is_pinned():
                    continue
                moved = torch.empty_like(shadow, device="cpu", pin_memory=True)
                moved.copy_(shadow, non_blocking=non_blocking and shadow.device.type == "cuda")
                self.shadow[index] = moved
            else:
                self.shadow[index] = shadow.to(device, non_blocking=non_blocking)

    @contextmanager
    def swap_in(self):
        """Temporarily expose EMA weights as the live parameters."""
        self._swap()
        self._swapped = True
        try:
            yield
        finally:
            self._swap()
            self._swapped = False

    @torch.no_grad()
    def _swap(self) -> None:
        for live, sh in zip(self.params, self.shadow, strict=True):
            live_local = _local(live.data)
            tmp = live_local.clone()
            live_local.copy_(sh)
            sh.copy_(tmp)
