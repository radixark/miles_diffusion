"""--diffusion-score-centering is Flow-GRPO only and needs the engine's per-step means.

    off                                   ---> no checks
    on + policy_loss + --diffusion-debug-mode ---> accepted
    on + nft / sft_loss / custom_loss     ---> rejected (no SDE log-prob to center)
    on without --diffusion-debug-mode     ---> rejected (engine returns no prev-sample means)
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.utils.arguments import validate_score_centering_args


def _args(**overrides):
    values = dict(diffusion_score_centering=True, loss_type="policy_loss", diffusion_debug_mode=True)
    values.update(overrides)
    return Namespace(**values)


def test_off_skips_every_check():
    validate_score_centering_args(_args(diffusion_score_centering=False, loss_type="nft", diffusion_debug_mode=False))


def test_flow_grpo_with_debug_mode_is_accepted():
    validate_score_centering_args(_args())


@pytest.mark.parametrize("loss_type", ["nft", "sft_loss", "custom_loss"])
def test_rejects_objectives_without_an_sde_log_prob(loss_type):
    with pytest.raises(ValueError, match="Flow-GRPO"):
        validate_score_centering_args(_args(loss_type=loss_type))


def test_requires_debug_mode_for_engine_means():
    with pytest.raises(ValueError, match="--diffusion-debug-mode"):
        validate_score_centering_args(_args(diffusion_debug_mode=False))
