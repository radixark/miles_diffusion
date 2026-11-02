import copy
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainerConfig:
    role: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainPhase:
    """One step of a rollout's train schedule: ``role`` trains its optimizer steps ``optim_steps`` (all when None),
    or runs ``compute_path`` without grad; later phases read its output as ``external_data[key]``."""

    role: str
    key: str
    optim_steps: tuple[int, int] | None = None
    compute_path: str | None = None


def compute_trainer_configs(args: Namespace) -> list[TrainerConfig]:
    if args.loss_type != "dmd2":
        return [TrainerConfig(role="actor")]
    critic_steps = args.dmd2_critic_steps_per_rollout
    reuse_critic_step = args.dmd2_critic_scoring == "reuse-critic-step"
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
        "custom_train_input_function_path": None,
        "custom_train_output_function_path": (
            "miles.backends.fsdp_utils.loss_hub.dmd2.score_last_step_samples" if reuse_critic_step else None
        ),
    }
    return [
        TrainerConfig(role="critic", overrides=critic_overrides),
        TrainerConfig(role="actor", overrides={"teacher_load": None}),
    ]


def compute_train_schedule(args: Namespace) -> list[TrainPhase]:
    if args.loss_type != "dmd2" or args.dmd2_critic_scoring == "reuse-critic-step":
        return [TrainPhase(role=config.role, key=config.role) for config in compute_trainer_configs(args)]
    # tianweiy/DMD2's order: the critic scores its last step's samples before it trains on them.
    critic_steps = args.dmd2_critic_steps_per_rollout
    phases = [
        TrainPhase(
            role="critic", key="critic", compute_path="miles.backends.fsdp_utils.loss_hub.dmd2.score_held_out_samples"
        ),
        TrainPhase(role="actor", key="actor"),
        TrainPhase(role="critic", key="critic_last_step", optim_steps=(critic_steps - 1, critic_steps)),
    ]
    if critic_steps > 1:
        phases.insert(0, TrainPhase(role="critic", key="critic_earlier_steps", optim_steps=(0, critic_steps - 1)))
    return phases


def compute_trainer_args(args: Namespace, trainer: TrainerConfig) -> Namespace:
    ans = copy.deepcopy(args)
    for key, value in trainer.overrides.items():
        assert key in vars(ans), f"trainer {trainer.role!r} overrides {key!r}, which the argument parser does not know"
        setattr(ans, key, value)
    return ans
