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
