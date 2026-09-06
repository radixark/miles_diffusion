"""``--custom-rm-path`` example: a weighted sum of built-in rewards, weighted by ``--custom-rm-args``.

    --custom-rm-path miles.rollout.rm_hub.weighted_mixture_rm.weighted_mixture_rm \\
    --custom-rm-args "hps=0.7,pickscore=0.3" --reward-key weighted

Each sample's reward is a dict holding every component plus ``"weighted"``, so each reward
gets its own ``rollout/reward/<name>_mean`` panel while ``--reward-key`` picks what GRPO trains
on. Each named reward scores the whole batch once and keeps its own placement flags
(``--<rm>-reward-colocate``, ``--<rm>-num-gpus-per-worker``). Weights apply to raw scores,
whose scales differ: HPSv2.1 ~0.3, PickScore/26 ~0.85, OCR in [0, 1].
"""

import asyncio
from collections.abc import Sequence

from miles.utils.types import Sample

from .hps import hps_rm
from .ocr import ocr_rm
from .pickscore import pickscore_rm

_REWARDS = {"hps": hps_rm, "pickscore": pickscore_rm, "ocr": ocr_rm}


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


async def weighted_mixture_rm(args, samples: Sequence[Sample], **kwargs) -> list[dict[str, float]]:
    weights = parse_weights(args.custom_rm_args)
    if args.reward_key not in {name for name, _ in weights} | {"weighted"}:
        raise ValueError(
            f"weighted_mixture_rm returns a dict per sample; pass --reward-key weighted (or one of "
            f"{[name for name, _ in weights]}), got {args.reward_key!r}"
        )
    per_reward = await asyncio.gather(*(_REWARDS[name](args, samples) for name, _ in weights))
    rewards = []
    for i in range(len(samples)):
        components = {name: scores[i] for (name, _), scores in zip(weights, per_reward, strict=True)}
        components["weighted"] = sum(weight * components[name] for name, weight in weights)
        rewards.append(components)
    return rewards
