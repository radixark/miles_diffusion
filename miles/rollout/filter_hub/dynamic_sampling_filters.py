import torch

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

__all__ = ["check_reward_nonzero_std"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    # torch.std() on a single-element tensor divides by (n - 1) == 0 and returns NaN,
    # which makes `NaN > 0.0` evaluate to False. With exactly one sample there is no
    # variance to check, so always keep it instead of dropping every group of size 1.
    if len(rewards) <= 1:
        keep = True
    else:
        keep = bool(torch.tensor(rewards, dtype=torch.float).std() > 0.0)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
