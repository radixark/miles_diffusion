"""weighted_mixture_rm combines built-in rewards with weights parsed from --custom-rm-args.

Mental model (--custom-rm-args "hps=0.7,pickscore=0.3" --reward-key weighted, one batch of 2 samples):

    weighted_mixture_rm ─┬─ hps_rm(args, [s0, s1])       -> [0.3, 0.2]  x 0.7
                         └─ pickscore_rm(args, [s0, s1]) -> [0.8, 0.9]  x 0.3
                         = [{hps: 0.3, pickscore: 0.8, weighted: 0.45}, {hps: 0.2, pickscore: 0.9, weighted: 0.41}]

Covered: each reward scores the whole batch once and every sample gets its components plus the
weighted sum (1); an unknown reward name in --custom-rm-args is rejected (2); a --reward-key
that names neither a component nor "weighted" is rejected before any reward runs (3).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

import miles.rollout.rm_hub.weighted_mixture_rm as weighted_mixture_rm_module
from miles.rollout.rm_hub.weighted_mixture_rm import parse_weights, weighted_mixture_rm


def _fake_rewards(calls):
    def fake(scores):
        async def rm(args, samples):
            calls.append(len(samples))
            return scores

        return rm

    return {"hps": fake([0.3, 0.2]), "pickscore": fake([0.8, 0.9])}


@pytest.mark.asyncio
async def test_each_sample_gets_its_components_and_the_weighted_sum(monkeypatch):
    """Fanning the batch out per sample, dropping a weight, or collapsing to a scalar would all show here."""
    calls = []
    monkeypatch.setattr(weighted_mixture_rm_module, "_REWARDS", _fake_rewards(calls))
    args = Namespace(custom_rm_args="hps=0.7,pickscore=0.3", reward_key="weighted")

    rewards = await weighted_mixture_rm(args, [object(), object()])

    assert [r["weighted"] for r in rewards] == pytest.approx([0.45, 0.41])
    assert [(r["hps"], r["pickscore"]) for r in rewards] == [(0.3, 0.8), (0.2, 0.9)]
    assert calls == [2, 2]


def test_unknown_reward_name_is_rejected():
    with pytest.raises(ValueError, match="unknown reward 'clip'"):
        parse_weights("hps=0.7,clip=0.3")


@pytest.mark.asyncio
async def test_missing_reward_key_is_rejected_before_scoring(monkeypatch):
    calls = []
    monkeypatch.setattr(weighted_mixture_rm_module, "_REWARDS", _fake_rewards(calls))

    with pytest.raises(ValueError, match="--reward-key weighted"):
        await weighted_mixture_rm(Namespace(custom_rm_args="hps=0.7,pickscore=0.3", reward_key=None), [object()])
    assert calls == []
