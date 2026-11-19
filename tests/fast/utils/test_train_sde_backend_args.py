"""The train-side SDE backend must be able to score the rollout the engine produced.

    sde + no-const + default FlowSdeStepBackend (keeps constants) --> rejected
    sde + no-const + custom --sde-step-backend-path                --> allowed (backend owns it)
    ode + the Flow-GRPO formula (policy_loss, or custom_loss without a formula path) --> rejected
    ode + nft / dmd2 / custom_loss with its own formula; cps; flag off --> allowed
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.utils.arguments import validate_train_sde_backend

NFT_FORMULA = "miles.backends.fsdp_utils.loss_hub.nft.nft_loss_formula"


def _args(**overrides):
    """One stream, "visual", unless the SDE types are given per stream; a single value applies to every stream."""
    values = dict(
        diffusion_log_prob_no_const=True,
        diffusion_sde_type="sde",
        diffusion_noise_level=0.7,
        sde_step_backend_path=None,
        loss_type="policy_loss",
        custom_prepare_train_batch_path=None,
        custom_loss_function_path=None,
        custom_train_input_postprocess_path=None,
        custom_train_output_function_path=None,
    )
    values.update(overrides)
    streams = list(values["diffusion_sde_type"]) if isinstance(values["diffusion_sde_type"], dict) else ["visual"]
    for name in ("diffusion_log_prob_no_const", "diffusion_sde_type", "diffusion_noise_level"):
        if not isinstance(values[name], dict):
            values[name] = dict.fromkeys(streams, values[name])
    return Namespace(**values)


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(), "--diffusion-log-prob-no-const"),
        (dict(diffusion_sde_type="ode"), "--diffusion-sde-type ode"),
        (dict(diffusion_sde_type="ode", loss_type="custom_loss"), "--diffusion-sde-type ode"),
        pytest.param(
            dict(
                diffusion_sde_type={"visual": "ode", "audio": "ode"},
                diffusion_log_prob_no_const={"visual": True, "audio": True},
            ),
            "at least one stream",
            id="every stream ode under Flow-GRPO",
        ),
        pytest.param(
            dict(
                diffusion_sde_type={"visual": "sde", "audio": "cps"},
                diffusion_log_prob_no_const={"visual": False, "audio": True},
            ),
            "with and without the Gaussian constants",
            id="a log-density added to a constant-free term",
        ),
        pytest.param(
            dict(
                diffusion_sde_type={"visual": "cps", "audio": "cps"},
                diffusion_noise_level={"visual": 0.7, "audio": 0.3},
            ),
            "one --diffusion-noise-level",
            id="constant-free streams on different noise scales",
        ),
        pytest.param(
            dict(
                sde_step_backend_path="my_pkg.backends.NoConstFlowSde",
                diffusion_sde_type={"visual": "sde", "audio": "cps"},
                diffusion_log_prob_no_const={"visual": True, "audio": True},
            ),
            "implements one dynamics",
            id="one custom backend for two dynamics",
        ),
        pytest.param(
            dict(diffusion_sde_type={"visual": "cps", "audio": "sde"}),
            "--diffusion-log-prob-no-const",
            id="a second stream's sde under the flag without a custom backend",
        ),
        pytest.param(
            dict(diffusion_log_prob_no_const=False, diffusion_noise_level=0.0),
            "must be positive",
            id="a log-density at a zero noise level",
        ),
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
        dict(diffusion_sde_type="ode", loss_type="nft"),
        dict(diffusion_sde_type="ode", loss_type="dmd2"),
        dict(diffusion_sde_type="ode", loss_type="custom_loss", custom_loss_function_path=NFT_FORMULA),
        dict(diffusion_log_prob_no_const=False),
        pytest.param(
            dict(
                diffusion_sde_type={"visual": "sde", "audio": "ode"},
                diffusion_log_prob_no_const={"visual": False, "audio": True},
            ),
            id="an ode stream next to a scored log-density",
        ),
        pytest.param(
            dict(
                diffusion_sde_type={"visual": "sde", "audio": "sde"},
                diffusion_log_prob_no_const={"visual": False, "audio": False},
                diffusion_noise_level={"visual": 0.7, "audio": 0.3},
            ),
            id="log-densities on different noise scales",
        ),
        pytest.param(
            dict(
                diffusion_sde_type={"visual": "cps", "audio": "cps"},
                diffusion_noise_level={"visual": 0.5, "audio": 0.5},
            ),
            id="constant-free streams on one noise scale",
        ),
    ],
)
def test_allows_consistent_combinations(overrides):
    validate_train_sde_backend(_args(**overrides))
