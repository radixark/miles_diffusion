"""Reward → train-signal helpers for diffusion (customization building blocks).

Default GRPO group normalization. Override with
``--custom-reward-post-process-path`` pointing at a function with the same
signature as ``grpo_normalize_rewards``.
"""

from __future__ import annotations

from argparse import Namespace

import torch

from miles.utils.types import Sample


def grpo_normalize_rewards(
    args: Namespace,
    samples: list[Sample] | list[list[Sample]],
) -> tuple[list[float], list[float]]:
    """Group-relative reward normalization used by Flow-GRPO.

    Returns ``(raw_rewards, normalized_rewards)``. Normalized values are used
    as per-sample advantages when building train pairs.

    ``--globalize-reward-mean`` / ``--globalize-reward-std`` are orthogonal.
    flow_grpo pickscore_qwenimage uses per-prompt mean + global std
    (``PerPromptStatTracker`` with ``global_std=True``), which is
    ``--globalize-reward-std`` alone.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]

    rewards_flat = torch.tensor(raw_rewards, dtype=torch.float)
    rewards = rewards_flat.view(-1, args.n_samples_per_prompt)

    if args.globalize_reward_mean:
        mean = rewards_flat.mean()
    else:
        mean = rewards.mean(dim=-1, keepdim=True)
    rewards = rewards - mean

    if args.grpo_std_normalization:
        if args.globalize_reward_std:
            std = rewards_flat.std()
        else:
            std = rewards.std(dim=-1, keepdim=True)
        # matches flow_grpo's `+ 1e-4` in both stat_tracking branches
        rewards = rewards / (std + 1e-4)

    return raw_rewards, rewards.flatten().tolist()
