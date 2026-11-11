from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.utils.trainer_config import TrainerConfig, compute_trainer_args, compute_trainer_configs


def test_the_actor_is_the_only_trainer_by_default():
    assert compute_trainer_configs(Namespace(loss_type="nft")) == [TrainerConfig(role="actor")]


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
    args = Namespace(
        loss_type="dmd2",
        hf_checkpoint="student",
        teacher_load="teacher",
        critic_load="critic-init",
        lr=2e-6,
        critic_lr=8e-6,
        load=load,
        save=save,
        global_batch_size=40,
        num_steps_per_rollout=1,
        dmd2_critic_steps_per_rollout=5,
        use_ema=True,
    )
    original = vars(args).copy()

    trainer_configs = compute_trainer_configs(args)

    assert [config.role for config in trainer_configs] == ["critic", "actor"]
    critic, actor = (compute_trainer_args(args, config) for config in trainer_configs)
    # The critic differs from the run's args only in what it trains, how fast, where it checkpoints, how many
    # steps it takes and that it keeps no EMA; its hooks come from the loss hub by role, not from args.
    assert vars(critic) == {
        **original,
        "hf_checkpoint": "critic-init",
        "lr": 8e-6,
        "load": critic_dir,
        "save": critic_dir,
        "num_steps_per_rollout": 5,
        "global_batch_size": 8,
        "use_ema": False,
    }
    assert vars(actor) == {**original, "teacher_load": None}
    assert vars(args) == original
