from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.utils.trainer_config import (
    TrainerConfig,
    TrainPhase,
    compute_train_schedule,
    compute_trainer_args,
    compute_trainer_configs,
)

SCORE_HELD_OUT = "miles.backends.fsdp_utils.loss_hub.dmd2.score_held_out_samples"


def _dmd2_args(**overrides):
    values = dict(
        loss_type="dmd2",
        hf_checkpoint="student",
        teacher_load="teacher",
        critic_load="critic-init",
        lr=2e-6,
        critic_lr=8e-6,
        load="/ckpt/run/",
        save="/ckpt/run",
        global_batch_size=40,
        num_steps_per_rollout=1,
        dmd2_critic_steps_per_rollout=5,
        dmd2_critic_scoring="reuse-critic-step",
        custom_prepare_train_batch_path="my.prepare_generator",
        custom_loss_function_path="my.generator_loss",
        custom_train_input_function_path="my.select_pairs",
        custom_train_output_function_path=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_the_actor_is_the_only_trainer_by_default():
    assert compute_trainer_configs(Namespace(loss_type="nft")) == [TrainerConfig(role="actor")]
    assert compute_train_schedule(Namespace(loss_type="nft")) == [TrainPhase(role="actor", key="actor")]


def test_trainer_args_apply_overrides_to_a_deep_copy():
    args = Namespace(lr=1e-5, save="/ckpt/run", fsdp_flow_shift={"visual": 3.0})
    original = {**vars(args), "fsdp_flow_shift": dict(args.fsdp_flow_shift)}

    trainer_args = compute_trainer_args(args, TrainerConfig(role="critic", overrides={"lr": 8e-6}))
    trainer_args.fsdp_flow_shift["visual"] = 12.0

    assert (trainer_args.lr, trainer_args.save) == (8e-6, "/ckpt/run")
    assert vars(args) == original


def test_trainer_args_reject_unknown_overrides():
    with pytest.raises(AssertionError, match="overrides 'lrr'"):
        compute_trainer_args(Namespace(lr=1e-5), TrainerConfig(role="critic", overrides={"lrr": 8e-6}))


@pytest.mark.parametrize(
    ("load", "save", "critic_dir"), [("/ckpt/run/", "/ckpt/run", "/ckpt/run_critic"), (None, None, None)]
)
def test_dmd2_critic_trains_first_from_its_own_checkpoint(load, save, critic_dir):
    args = _dmd2_args(load=load, save=save)
    original = vars(args).copy()

    trainer_configs = compute_trainer_configs(args)

    assert [config.role for config in trainer_configs] == ["critic", "actor"]
    critic, actor = (compute_trainer_args(args, config) for config in trainer_configs)
    assert (critic.hf_checkpoint, critic.teacher_load, critic.lr) == ("critic-init", "teacher", 8e-6)
    assert (critic.load, critic.save) == (critic_dir, critic_dir)
    assert (critic.num_steps_per_rollout, critic.global_batch_size) == (5, 8)
    assert critic.custom_prepare_train_batch_path.endswith("loss_hub.dmd2.prepare_critic_batch")
    assert critic.custom_loss_function_path.endswith("loss_hub.dmd2.critic_loss_formula")
    assert critic.custom_train_input_function_path is None
    assert critic.custom_train_output_function_path.endswith("loss_hub.dmd2.score_last_step_samples")
    assert vars(actor) == {**original, "teacher_load": None}
    assert vars(args) == original


def test_reuse_critic_step_scoring_trains_the_critic_then_the_actor():
    assert compute_train_schedule(_dmd2_args()) == [
        TrainPhase(role="critic", key="critic"),
        TrainPhase(role="actor", key="actor"),
    ]


@pytest.mark.parametrize("critic_steps", [5, 1])
def test_original_scoring_scores_before_the_critic_trains_on_the_last_step(critic_steps):
    args = _dmd2_args(dmd2_critic_scoring="original", dmd2_critic_steps_per_rollout=critic_steps)

    schedule = compute_train_schedule(args)
    critic = compute_trainer_args(args, compute_trainer_configs(args)[0])

    expected = [
        TrainPhase(role="critic", key="critic", compute_path=SCORE_HELD_OUT),
        TrainPhase(role="actor", key="actor"),
        TrainPhase(role="critic", key="critic_last_step", optim_steps=(critic_steps - 1, critic_steps)),
    ]
    if critic_steps > 1:
        expected.insert(0, TrainPhase(role="critic", key="critic_earlier_steps", optim_steps=(0, critic_steps - 1)))
    assert schedule == expected
    assert critic.custom_train_output_function_path is None
