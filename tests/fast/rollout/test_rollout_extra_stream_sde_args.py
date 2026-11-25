"""An extra stream sent its own rollout settings must come back rolled out with them.

    request: rollout_extra_stream_sde_args = {"audio": {sde type, noise level, no-const}}
    engine:  extra_streams.audio.rollout_sde_kernel = the settings it applied   (echo)
    md:      check_extra_stream_sde_args, on every training response as generate_microgroup parses it

    echo equal to the request                        --> accepted
    echo equal except an ode stream's noise          --> accepted (an ode stream draws no noise)
    no echo (an engine without per-stream SDE args)  --> rejected
    echo with another type, or another noise         --> rejected
    an extra key in the echo                         --> accepted (only the three settings are compared)
    no SDE args sent                                 --> nothing to check
    generate_microgroup, training / eval             --> checked / not checked (an eval response has no rollout data)
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

from miles.rollout.sglang_diffusion_rollout import check_extra_stream_sde_args
from miles.utils.types import RolloutStreamData, Sample

SDE = {"rollout_sde_type": "sde", "rollout_noise_level": 0.3, "rollout_log_prob_no_const": False}
ODE = {"rollout_sde_type": "ode", "rollout_noise_level": 0.3, "rollout_log_prob_no_const": True}


def _check(sde_args, echoed):
    sample = Sample(
        index=3,
        extra_streams={"audio": RolloutStreamData(rollout_log_probs=torch.zeros(2), rollout_sde_kernel=echoed)},
    )
    check_extra_stream_sde_args(SimpleNamespace(rollout_extra_stream_sde_args=sde_args), [sample])


@pytest.mark.parametrize(
    ("sde_args", "echoed"),
    [
        pytest.param({"audio": SDE}, SDE, id="echo equal to the request"),
        pytest.param({"audio": ODE}, {**ODE, "rollout_noise_level": 0.0}, id="an ode stream's noise level"),
        pytest.param({"audio": SDE}, {**SDE, "rollout_steps": 4}, id="an extra key in the echo"),
        pytest.param({}, None, id="no SDE args sent"),
    ],
)
def test_a_stream_rolled_out_as_requested_passes(sde_args, echoed):
    _check(sde_args, echoed)


@pytest.mark.parametrize(
    "echoed",
    [
        pytest.param(None, id="an engine without per-stream SDE args"),
        pytest.param({**SDE, "rollout_sde_type": "cps"}, id="another type"),
        pytest.param({**SDE, "rollout_noise_level": 0.7}, id="another noise level"),
    ],
)
def test_a_stream_rolled_out_otherwise_is_rejected(echoed):
    """An engine that ignores the field rolls the stream out with the primary's settings; Flow-GRPO would score it
    with the requested ones, without any error."""
    with pytest.raises(ValueError, match="must apply and echo rollout_extra_stream_sde_args"):
        _check({"audio": SDE}, echoed)


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation", [False, True], ids=["training", "eval"])
async def test_generate_checks_every_training_response(monkeypatch, evaluation):
    """Dropping the call leaves the check unused: an engine that ignores the sde_args would go unnoticed."""
    from miles.rollout import sglang_diffusion_rollout as rollout_module

    parsed = [Sample(index=0, extra_streams={"audio": RolloutStreamData(rollout_log_probs=torch.zeros(2))})]
    parser = SimpleNamespace(apply_raw=SimpleNamespace(remote=lambda pending, raw: parsed))
    state = SimpleNamespace(
        step_strategy_fn=None, next_parser_idx=lambda: 0, parser_inflight=[0], response_parsers=[parser]
    )

    async def post(url, payload, raw):
        return b""

    monkeypatch.setattr(rollout_module, "GenerateState", lambda args: state)
    monkeypatch.setattr(rollout_module, "post", post)
    monkeypatch.setattr(rollout_module.ray, "get", lambda value: value)
    args = SimpleNamespace(
        sglang_router_ip="router",
        sglang_router_port=0,
        rollout_fetch_in_parser=False,
        rollout_extra_stream_sde_args={"audio": SDE},
    )
    generate = rollout_module.generate_microgroup(
        args, [Sample(index=0, prompt="p")], {"seed": 0, "num_inference_steps": 2}, evaluation=evaluation
    )

    if evaluation:
        assert await generate == parsed
    else:
        with pytest.raises(ValueError, match="must apply and echo rollout_extra_stream_sde_args"):
            await generate
