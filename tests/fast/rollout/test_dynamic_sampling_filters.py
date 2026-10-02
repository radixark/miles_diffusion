"""check_reward_nonzero_std must never drop a group made of a single sample."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

from miles.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std
from miles.utils.types import Sample


def _sample(reward: float) -> Sample:
    return Sample(reward=reward)


def test_single_sample_group_is_always_kept():
    args = Namespace(reward_key=None)

    output = check_reward_nonzero_std(args, [_sample(5.0)])

    assert output.keep is True
    assert output.reason is None


def test_zero_variance_group_is_dropped():
    args = Namespace(reward_key=None)

    output = check_reward_nonzero_std(args, [_sample(1.0), _sample(1.0), _sample(1.0)])

    assert output.keep is False
    assert output.reason == "zero_std_1.0"


def test_nonzero_variance_group_is_kept():
    args = Namespace(reward_key=None)

    output = check_reward_nonzero_std(args, [_sample(0.0), _sample(1.0), _sample(1.0)])

    assert output.keep is True
    assert output.reason is None
