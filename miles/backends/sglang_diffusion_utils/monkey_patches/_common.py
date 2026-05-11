import torch


def ensure_broadcast(mod: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if mod.dim() == ref.dim() - 1:
        return mod.unsqueeze(-2)
    return mod
