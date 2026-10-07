from __future__ import annotations

from contextlib import contextmanager

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def reshard_model(model: torch.nn.Module) -> None:
    """Drop FSDP's gathered full parameters so the registered parameters are the local shards again."""
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


class EMAOptimizer(torch.optim.Optimizer):
    """EMA of the trainable weights, kept as optimizer state.

    With ``keep_previous=True``, ``previous_ema`` also holds the EMA from before the latest ``step``:
    async training samples each prefetched batch with it, and checkpoints save it so a resumed run
    resamples its first batch with it.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        decay: float = 0.001,
        uprate: float = 0.001,
        uphold: float = 0.5,
        flat_steps: int = 0,
        keep_previous: bool = False,
    ) -> None:
        super().__init__(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            dict(decay=decay, uprate=uprate, uphold=uphold, flat_steps=flat_steps),
        )
        self.keep_previous = keep_previous
        self.previous_ema: dict[torch.nn.Parameter, torch.Tensor] | None = None
        self.reset_from_model()

    @torch.no_grad()
    def reset_from_model(self) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                self.state[parameter]["ema"] = parameter.detach().clone()
        self.reset_previous()

    @torch.no_grad()
    def reset_previous(self) -> None:
        """Set the previous EMA to the current EMA, as before the first ``step``."""
        if self.keep_previous:
            self.previous_ema = {parameter: state["ema"].clone() for parameter, state in self.state.items()}

    @torch.no_grad()
    def step(self, optimizer_step: int) -> None:
        """Average in the actor after its ``optimizer_step``-th optimizer update; the decay follows that count."""
        for group in self.param_groups:
            decay = (
                group["decay"]
                if optimizer_step <= group["flat_steps"]
                else min((optimizer_step - group["flat_steps"]) * group["uprate"], group["uphold"])
            )
            ema_tensors = [_local_tensor(self.state[parameter]["ema"]) for parameter in group["params"]]
            if self.keep_previous:
                previous_tensors = [_local_tensor(self.previous_ema[parameter]) for parameter in group["params"]]
                torch._foreach_copy_(previous_tensors, ema_tensors)
            actor_tensors = [_local_tensor(parameter.detach()) for parameter in group["params"]]
            torch._foreach_lerp_(ema_tensors, actor_tensors, 1.0 - decay)

    @contextmanager
    def use_weights(self, model: torch.nn.Module, previous: bool = False):
        """Temporarily put the EMA weights into ``model``'s shards; ``previous`` picks the EMA before the latest step.

        Gathered full parameters are dropped before swapping in and before swapping back,
        so no forward reads stale weights.
        """
        if previous and not self.keep_previous:
            raise RuntimeError("EMAOptimizer was built without keep_previous")
        reshard_model(model)
        ema_weights = {
            parameter: self.previous_ema[parameter] if previous else self.state[parameter]["ema"]
            for parameter in model.parameters()
            if parameter in self.state
        }
        self._swap_with_actor(ema_weights)
        try:
            yield
        finally:
            reshard_model(model)
            self._swap_with_actor(ema_weights)

    @torch.no_grad()
    def _swap_with_actor(self, ema_weights: dict[torch.nn.Parameter, torch.Tensor]) -> None:
        # EMA shares the actor's device, so its own storage holds the actor weights while they are swapped out.
        for parameter, ema_weight in ema_weights.items():
            actor_tensor = _local_tensor(parameter)
            ema_tensor = _local_tensor(ema_weight)
            actor_copy = actor_tensor.clone()
            actor_tensor.copy_(ema_tensor)
            ema_tensor.copy_(actor_copy)
