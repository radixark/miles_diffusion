"""The train-side SDE backend must be able to score the rollout the engine produced.

    sde + no-const + default FlowSdeStepBackend (keeps constants) --> rejected
    sde + no-const + custom --sde-step-backend-path                --> allowed (backend owns it)
    ode + Flow-GRPO loss formula (no custom formula)               --> rejected
    ode + NFT / custom loss formula; cps; flag off                 --> allowed
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.utils.arguments import validate_train_sde_backend

NFT_FORMULA = "miles.backends.fsdp_utils.loss_hub.nft.nft_loss_formula"


def _args(**overrides):
    values = dict(
        diffusion_log_prob_no_const=True,
        diffusion_sde_type="sde",
        sde_step_backend_path=None,
        custom_loss_function_path=None,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(), "--diffusion-log-prob-no-const"),
        (dict(diffusion_sde_type="ode"), "--diffusion-sde-type ode"),
    ],
)
def test_rejects_rollouts_the_backend_cannot_score(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_train_sde_backend(_args(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(sde_step_backend_path="my_pkg.backends.NoConstFlowSde"),
        dict(diffusion_sde_type="cps"),
        dict(diffusion_sde_type="ode", custom_loss_function_path=NFT_FORMULA),
        dict(diffusion_log_prob_no_const=False),
    ],
)
def test_allows_consistent_combinations(overrides):
    validate_train_sde_backend(_args(**overrides))
