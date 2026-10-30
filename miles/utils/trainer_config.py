import copy
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainerConfig:
    role: str
    overrides: dict[str, Any] = field(default_factory=dict)


def compute_trainer_configs(args: Namespace) -> list[TrainerConfig]:
    return [TrainerConfig(role="actor")]


def compute_trainer_args(args: Namespace, trainer: TrainerConfig) -> Namespace:
    ans = copy.deepcopy(args)
    for key, value in trainer.overrides.items():
        assert key in vars(ans), f"trainer {trainer.role!r} overrides {key!r}, which the argument parser does not know"
        setattr(ans, key, value)
    return ans
