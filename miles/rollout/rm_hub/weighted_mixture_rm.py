"""``--custom-rm-path`` example: a weighted sum of built-in rewards, weighted by ``--custom-rm-args``.

    --custom-rm-path miles.rollout.rm_hub.weighted_mixture_rm.weighted_mixture_rm \\
    --custom-rm-args "hps=0.7,pickscore=0.3"

Each named reward scores the whole batch once and keeps its own placement flags
(``--<rm>-reward-colocate``, ``--<rm>-num-gpus-per-worker``). Weights apply to raw scores,
whose scales differ: HPSv2.1 ~0.3, PickScore/26 ~0.85.
"""

from collections.abc import Sequence

from miles.utils.types import Sample

from .hps import hps_rm
from .pickscore import pickscore_rm

_REWARDS = {"hps": hps_rm, "pickscore": pickscore_rm}


def parse_weights(custom_rm_args: str) -> list[tuple[str, float]]:
    weights = []
    # launch scripts hand the arg string to `sh`, where ";" would end the command; "," is inert
    for term in custom_rm_args.split(","):
        name, _, weight = term.strip().partition("=")
        if name not in _REWARDS:
            raise ValueError(
                f"--custom-rm-args: unknown reward {name!r} in {custom_rm_args!r}; choose from {tuple(_REWARDS)}"
            )
        weights.append((name, float(weight)))
    return weights


async def weighted_mixture_rm(args, samples: Sequence[Sample], **kwargs) -> list[float]:
    totals = [0.0] * len(samples)
    for name, weight in parse_weights(args.custom_rm_args):
        for i, score in enumerate(await _REWARDS[name](args, samples)):
            totals[i] += weight * score
    return totals
