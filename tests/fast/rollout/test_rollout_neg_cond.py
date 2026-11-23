"""A trainer that runs CFG on unguided rollouts gets their negative cond from one request that runs CFG.

    args.rollout_neg_cond_guidance_scale   (dmd2 sets it when its teacher runs CFG)

    fetch_neg_cond       --> the run's sampling params, with CFG at that scale, stopped after step 0, no
                             trajectory and inline tensors; returns the reply's negative cond
    generate_microgroup  --> a training sample without a negative cond gets the fetched one;
                             a sample that has its own keeps it
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save

from miles.rollout import sglang_diffusion_rollout as rollout_module
from miles.utils.types import CondKwargs, DenoisingEnv, Sample


@pytest.mark.asyncio
async def test_the_fetch_runs_cfg_for_one_step_and_returns_the_replys_neg_cond(monkeypatch):
    """Sent as an unguided sample's request, the reply has no negative cond; sent through the output store, its
    tensors are deleted with the first rollout."""
    neg_embeds = torch.arange(6.0).reshape(2, 3)
    sent = {}

    async def post(url, payload):
        sent.update(payload)
        wire = {"encoder_hidden_states": [{"__tensor__": True, "data": save({"t": neg_embeds})}]}
        return [{"denoising_env": {"pos_cond_kwargs": wire, "neg_cond_kwargs": wire}}]

    monkeypatch.setattr(rollout_module, "post", post)
    args = SimpleNamespace(
        sglang_router_ip="router", sglang_router_port=0, rollout_seed=7, rollout_neg_cond_guidance_scale=4.5
    )
    sampling_params = {
        "guidance_scale": 1.0,
        "rollout_return_dit_trajectory": True,
        "rollout_return_outputs_via_store": True,
    }

    neg_cond = await rollout_module.fetch_neg_cond(args, sampling_params)

    assert torch.equal(neg_cond.encoder_hidden_states[0], neg_embeds)
    assert (sent["guidance_scale"], sent["rollout_stop_step_index"]) == (4.5, 0)
    assert not sent["rollout_return_dit_trajectory"] and not sent["rollout_return_outputs_via_store"]
    assert sampling_params["guidance_scale"] == 1.0


@pytest.mark.asyncio
async def test_a_training_sample_without_a_neg_cond_gets_the_fetched_one(monkeypatch):
    """A sample left without one stops the teacher's CFG forward; overwriting an engine's own would swap in a
    negative cond padded for another prompt."""
    fetched, own = CondKwargs(), CondKwargs()
    parsed = [
        Sample(index=0, denoising_env=DenoisingEnv()),
        Sample(index=1, denoising_env=DenoisingEnv(neg_cond_kwargs=own)),
    ]
    parser = SimpleNamespace(apply_raw=SimpleNamespace(remote=lambda pending, raw: parsed))
    state = SimpleNamespace(
        step_strategy_fn=None,
        next_parser_idx=lambda: 0,
        parser_inflight=[0],
        response_parsers=[parser],
        neg_cond=fetched,
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
        rollout_extra_stream_sde_args={},
    )

    samples = await rollout_module.generate_microgroup(
        args, [Sample(index=0), Sample(index=1)], {"seed": 0, "num_inference_steps": 2}
    )

    assert samples[0].denoising_env.neg_cond_kwargs is fetched
    assert samples[1].denoising_env.neg_cond_kwargs is own
