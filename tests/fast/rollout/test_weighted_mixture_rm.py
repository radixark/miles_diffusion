"""weighted_mixture_rm combines built-in rewards with weights parsed from --custom-rm-args.

Mental model (--custom-rm-args "hps=0.7,pickscore=0.3", one batch of 2 samples):

    weighted_mixture_rm ─┬─ hps_rm(args, [s0, s1])       -> [0.3, 0.2]  x 0.7
                         └─ pickscore_rm(args, [s0, s1]) -> [0.8, 0.9]  x 0.3
                         = [0.45, 0.41]

Covered: each reward scores the whole batch once and the weights are applied (1); an
unknown reward name in --custom-rm-args is rejected (2).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

import miles.rollout.rm_hub.weighted_mixture_rm as weighted_mixture_rm_module
from miles.rollout.rm_hub.weighted_mixture_rm import parse_weights, weighted_mixture_rm


@pytest.mark.asyncio
async def test_weights_from_custom_rm_args_apply_to_each_reward_once(monkeypatch):
    """Fanning the batch out per sample or dropping a weight would change these sums."""
    calls = []

    def fake(scores):
        async def rm(args, samples):
            calls.append(len(samples))
            return scores

        return rm

    monkeypatch.setattr(
        weighted_mixture_rm_module, "_REWARDS", {"hps": fake([0.3, 0.2]), "pickscore": fake([0.8, 0.9])}
    )

    rewards = await weighted_mixture_rm(Namespace(custom_rm_args="hps=0.7,pickscore=0.3"), [object(), object()])

    assert rewards == pytest.approx([0.45, 0.41])
    assert calls == [2, 2]


def test_unknown_reward_name_is_rejected():
    with pytest.raises(ValueError, match="unknown reward 'clip'"):
        parse_weights("hps=0.7,clip=0.3")
