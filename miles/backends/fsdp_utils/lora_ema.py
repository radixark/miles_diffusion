"""Generic LoRA EMA shadow for diffusion (and future FSDP) training.

Algorithms that need a slow-moving reference / sampling policy ``pi_old`` share
``LoraEmaShadow``: trainable LoRA weights plus EMA buffers and ``swap_in()``
for temporary in-place weight exchange.

Lifecycle (actor / weight sync)::

    ema.update()
    with ema.swap_in():
        weight_updater.update_weights()

Loss-side reference forward (via ``DiffusionLossContext.ema_shadow``)::

    with torch.no_grad(), ctx.ema_shadow.swap_in():
        old_pred = forward(...)

Works with FSDP2 DTensor shards (per-rank local swap) and colocate CPU offload.

Checkpointing (intentionally not wired yet)
-------------------------------------------
``shadow`` / ``step`` are **not** saved or restored by ``fsdp_utils.checkpoint``.
On resume the actor rebuilds EMA from the loaded LoRA weights, so ``pi_old``
cold-starts (decay schedule restarts at step 0). Fine for single-shot runs;
wrong for mid-run resume that must match UniRL's slow ``pi_old``.

Wiring it later is non-trivial: buffers are per-rank plain clones (not in the
FSDP/DCP model state), must stay aligned with the trainable-param order, and
must not be saved while ``swap_in()`` is active. Prefer a side file such as
``iter_*/lora_ema.pt`` over stuffing into the DCP model dict.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterable
from contextlib import contextmanager

import torch
import torch.nn as nn


def _local(t: torch.Tensor) -> torch.Tensor:
    """Local shard of a (possibly DTensor) tensor; EMA/swap is per-rank, no comm."""
    return t._local_tensor if hasattr(t, "_local_tensor") else t


def lora_ema_shadow_enabled(args: Namespace) -> bool:
    """True when a LoRA EMA shadow should be constructed."""
    return bool(getattr(args, "lora_ema_shadow", False))


def lora_ema_rollout_policy(args: Namespace) -> str:
    """Which LoRA weights to push to rollout engines after each rollout ('live' or 'ema')."""
    return getattr(args, "lora_ema_rollout_policy", "live")


def resolve_lora_ema_kwargs(args: Namespace) -> dict[str, float | int]:
    """Read normalized ``lora_ema_*`` fields from ``args`` (see ``miles_validate_args``)."""
    return {
        "decay": float(getattr(args, "lora_ema_decay", 0.001)),
        "uprate": float(getattr(args, "lora_ema_uprate", 0.001)),
        "uphold": float(getattr(args, "lora_ema_uphold", 0.5)),
        "flat_steps": int(getattr(args, "lora_ema_flat_steps", 0)),
    }


class LoraEmaShadow:
    """EMA shadow of trainable (LoRA) parameters.

    Not part of the FSDP checkpoint payload today (see module docstring).
    """

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
            raise ValueError("LoraEmaShadow: model has no trainable parameters")
        self.shadow = [_local(p.detach()).clone() for p in self.params]

    def decay_at(self, t: int) -> float:
        if t <= self.flat_steps:
            return self.decay
        return float(min((t - self.flat_steps) * self.uprate, self.uphold))

    @torch.no_grad()
    def update(self) -> float:
        """theta_old <- delta * theta_old + (1 - delta) * theta."""
        if self._swapped:
            raise RuntimeError("LoraEmaShadow.update called while swapped in")
        self.step += 1
        delta = self.decay_at(self.step)
        for live, sh in zip(self.params, self.shadow, strict=True):
            sh.mul_(delta).add_(_local(live.detach()).to(sh.device), alpha=1.0 - delta)
        return delta

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
