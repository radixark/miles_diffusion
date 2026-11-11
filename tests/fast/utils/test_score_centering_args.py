"""--diffusion-score-centering only closes the integral its Gaussian assumptions define.

    policy_loss + sde (stock backend) + debug mode + rollout log-probs + clip in (0, 1) --> allowed
    nft / cps / custom backend / no debug mode / recompute old log-prob / clip >= 1    --> rejected
    flag off                                                                          --> anything goes
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.utils.arguments import validate_score_centering_args


def _args(**overrides):
    values = dict(
        diffusion_score_centering=True,
        loss_type="policy_loss",
        diffusion_sde_type="sde",
        sde_step_backend_path=None,
        diffusion_debug_mode=True,
        diffusion_recompute_old_log_prob=False,
        diffusion_clip_range=1e-4,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(loss_type="nft"), "policy_loss"),
        (dict(loss_type="sft_loss"), "policy_loss"),
        (dict(loss_type="custom_loss"), "policy_loss"),
        (dict(diffusion_sde_type="cps"), "--diffusion-sde-type sde"),
        (dict(sde_step_backend_path="my.Backend"), "default SDE step backend"),
        (dict(diffusion_debug_mode=False), "--diffusion-debug-mode"),
        (dict(diffusion_recompute_old_log_prob=True), "--diffusion-recompute-old-log-prob"),
        (dict(diffusion_clip_range=1.0), "--diffusion-clip-range"),
    ],
)
def test_rejects_setups_outside_the_gaussian_closed_form(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_score_centering_args(_args(**overrides))


def test_accepts_the_flow_grpo_sde_setup():
    validate_score_centering_args(_args())


def test_flag_off_validates_nothing():
    validate_score_centering_args(_args(diffusion_score_centering=False, loss_type="nft", diffusion_debug_mode=False))
