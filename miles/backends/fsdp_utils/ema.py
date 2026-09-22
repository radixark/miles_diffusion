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
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        decay: float = 0.001,
        uprate: float = 0.001,
        uphold: float = 0.5,
        flat_steps: int = 0,
    ) -> None:
        super().__init__(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            dict(decay=decay, uprate=uprate, uphold=uphold, flat_steps=flat_steps, update_count=0),
        )
        self.reset_from_model()

    @property
    def update_count(self) -> int:
        return self.param_groups[0]["update_count"]

    @torch.no_grad()
    def reset_from_model(self) -> None:
        for group in self.param_groups:
            group["update_count"] = 0
            for parameter in group["params"]:
                self.state[parameter]["ema"] = parameter.detach().clone()

    @torch.no_grad()
    def step(self, closure=None) -> float:
        for group in self.param_groups:
            group["update_count"] += 1
            update_count = group["update_count"]
            decay = (
                group["decay"]
                if update_count <= group["flat_steps"]
                else min((update_count - group["flat_steps"]) * group["uprate"], group["uphold"])
            )
            ema_tensors = [_local_tensor(self.state[parameter]["ema"]) for parameter in group["params"]]
            actor_tensors = [_local_tensor(parameter.detach()) for parameter in group["params"]]
            torch._foreach_lerp_(ema_tensors, actor_tensors, 1.0 - decay)
        return decay

    @contextmanager
    def use_weights(self, model: torch.nn.Module):
        """Temporarily put the EMA weights into ``model``'s shards.

        Gathered full parameters are dropped before swapping in and before swapping back,
        so no forward reads stale weights.
        """
        reshard_model(model)
        parameters = [parameter for parameter in model.parameters() if parameter in self.state]
        self._swap_with_actor(parameters)
        try:
            yield
        finally:
            reshard_model(model)
            self._swap_with_actor(parameters)

    @torch.no_grad()
    def _swap_with_actor(self, parameters: list[torch.nn.Parameter]) -> None:
        # EMA shares the actor's device, so its own storage holds the actor weights while they are swapped out.
        for parameter in parameters:
            actor_tensor = _local_tensor(parameter)
            ema_tensor = _local_tensor(self.state[parameter]["ema"])
            actor_copy = actor_tensor.clone()
            actor_tensor.copy_(ema_tensor)
            ema_tensor.copy_(actor_copy)
