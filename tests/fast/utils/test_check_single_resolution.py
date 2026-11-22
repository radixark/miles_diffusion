"""A train micro-batch holds pairs of one rollout resolution.

    pairs of one resolution / fixed-resolution pairs  ─► accepted
    672x384 next to 384x672 (same token count)        ─► rejected: qwen_image's RoPE would use
                                                         the first pair's grid for both
    unlabeled pairs while a strategy draws            ─► rejected: the check would be vacuous
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import pytest

from miles.utils.train_data_utils import check_single_resolution


def test_one_resolution_is_accepted():
    check_single_resolution([{"resolution": (672, 384)}] * 4)


def test_fixed_resolution_pairs_are_accepted():
    # without a resolution strategy pairs carry None, or no key at all (custom conversions)
    check_single_resolution([{"resolution": None}, {}])


def test_a_transposed_resolution_is_rejected():
    with pytest.raises(ValueError, match="mixes rollout resolutions"):
        check_single_resolution([{"resolution": (672, 384)}, {"resolution": (384, 672)}])


def test_unlabeled_pairs_are_rejected_while_a_strategy_draws():
    # a conversion that does not copy the sample's size onto its pairs (nft / dmd2 today) would
    # make every micro-batch look single-resolution
    with pytest.raises(ValueError, match="without one"):
        check_single_resolution([{"latent": None}, {"latent": None}], labels_required=True)


def test_unlabeled_pairs_are_fine_at_a_fixed_resolution():
    check_single_resolution([{"latent": None}, {"latent": None}], labels_required=False)


def test_a_fixed_resolution_run_writes_no_labels():
    """The two labels only exist while a strategy draws, so a fixed-resolution run does not
    carry a field per train pair that nothing reads."""
    import torch
    from types import SimpleNamespace

    from miles.ray.data_conversion_hub.flow_grpo import expand_samples_to_train_pairs

    def sample(height=None, width=None):
        traj = SimpleNamespace(
            latents=torch.zeros(4, 2), timesteps=torch.zeros(4), sigmas=torch.ones(4), latent_step_indices=None
        )
        return SimpleNamespace(
            index=0,
            prompt="p",
            height=height,
            width=width,
            dit_trajectory=traj,
            denoising_env=SimpleNamespace(tag=0),
            rollout_log_probs=torch.zeros(3),
            rollout_debug_tensors=None,
            train_metadata={"sde_step_indices": [0, 1]},
        )

    fixed = expand_samples_to_train_pairs(None, [sample()], [1.0], [1.0])["train_data"][0]
    assert "resolution" not in fixed and "log_prob_rollout" not in fixed

    drawn = expand_samples_to_train_pairs(None, [sample(672, 384)], [1.0], [1.0])["train_data"][0]
    assert drawn["resolution"] == (672, 384) and "log_prob_rollout" in drawn
