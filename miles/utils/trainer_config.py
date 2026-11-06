import copy
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainerConfig:
    role: str
    overrides: dict[str, Any] = field(default_factory=dict)


def compute_trainer_configs(args: Namespace) -> list[TrainerConfig]:
    if args.loss_type != "dmd2":
        return [TrainerConfig(role="actor")]
    critic_steps = args.dmd2_critic_steps_per_rollout
    critic_overrides = {
        "hf_checkpoint": args.critic_load,
        "lr": args.critic_lr,
        # A sibling dir, as upstream's --critic-save: sharing one would clobber the actor's iteration tracker.
        "load": None if args.load is None else args.load.rstrip("/") + "_critic",
        "save": None if args.save is None else args.save.rstrip("/") + "_critic",
        "num_steps_per_rollout": critic_steps,
        "global_batch_size": args.global_batch_size // critic_steps,
        "custom_prepare_train_batch_path": "miles.backends.fsdp_utils.loss_hub.dmd2.prepare_critic_batch",
        "custom_loss_function_path": "miles.backends.fsdp_utils.loss_hub.dmd2.critic_loss_formula",
        "custom_train_output_function_path": "miles.backends.fsdp_utils.loss_hub.dmd2.score_last_step_samples",
    }
    return [
        TrainerConfig(role="critic", overrides=critic_overrides),
        TrainerConfig(role="actor", overrides={"teacher_load": None}),
    ]


def compute_trainer_args(args: Namespace, trainer: TrainerConfig) -> Namespace:
    ans = copy.deepcopy(args)
    for key, value in trainer.overrides.items():
        assert key in vars(ans), f"trainer {trainer.role!r} overrides {key!r}, which the argument parser does not know"
        setattr(ans, key, value)
    return ans
