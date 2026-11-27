"""A trainer that runs CFG on unguided rollouts gets their negative cond from one request that runs CFG.

    args.rollout_neg_cond_guidance_scale   (dmd2 sets it when its teacher runs CFG)

    fetch_neg_cond              --> the run's sampling params, with CFG at that scale, stopped after step 0, with
                                    inline tensors; returns the reply's negative cond, and raises when the reply
                                    has none
    generate_rollout_async      --> fetches before the run's first rollout only
    generate_and_rm_microgroup  --> after either generate function, a training sample without a negative cond
                                    gets the fetched one; a sample that has its own keeps it
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import asyncio
import contextlib
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save

from miles.rollout import sglang_diffusion_rollout as rollout_module
from miles.utils.types import CondKwargs, DenoisingEnv, Sample

NEG_EMBEDS = torch.arange(6.0).reshape(2, 3)
ARGS = SimpleNamespace(
    sglang_router_ip="router", sglang_router_port=0, rollout_seed=7, rollout_neg_cond_guidance_scale=4.5
)


def _reply(*, with_neg_cond):
    wire = {"encoder_hidden_states": [{"__tensor__": True, "data": save({"t": NEG_EMBEDS})}]}
    return [{"denoising_env": {"pos_cond_kwargs": wire, "neg_cond_kwargs": wire if with_neg_cond else None}}]


@pytest.mark.asyncio
async def test_the_fetch_runs_cfg_for_one_step_on_the_runs_params_and_returns_the_replys_neg_cond(monkeypatch):
    """Sent as an unguided sample's request, the reply has no negative cond; sent without the run's negative prompt
    or size, it has the one of another prompt or another size."""
    sent = {}

    async def post(url, payload):
        sent.update(payload)
        return _reply(with_neg_cond=True)

    monkeypatch.setattr(rollout_module, "post", post)
    sampling_params = {
        "negative_prompt": "blurry",
        "width": 512,
        "guidance_scale": 1.0,
        "rollout_return_outputs_via_store": True,
    }

    neg_cond = await rollout_module.fetch_neg_cond(ARGS, sampling_params)

    assert torch.equal(neg_cond.encoder_hidden_states[0], NEG_EMBEDS)
    assert (sent["negative_prompt"], sent["width"]) == ("blurry", 512)
    assert (sent["guidance_scale"], sent["rollout_stop_step_index"]) == (4.5, 0)
    assert not sent["rollout_return_outputs_via_store"]
    assert sampling_params["guidance_scale"] == 1.0


@pytest.mark.asyncio
async def test_a_reply_without_a_neg_cond_stops_the_run_at_the_fetch(monkeypatch):
    """An engine without a negative prompt runs no CFG; accepted, the run would go on to its first scoring step."""

    async def post(url, payload):
        return _reply(with_neg_cond=False)

    monkeypatch.setattr(rollout_module, "post", post)
    with pytest.raises(ValueError, match="returned no negative cond"):
        await rollout_module.fetch_neg_cond(ARGS, {})


@pytest.mark.asyncio
async def test_the_neg_cond_is_fetched_before_the_first_rollout_only(monkeypatch):
    """Without the fetch no sample gets a negative cond; fetched per rollout, every rollout pays a CFG request."""

    class _Stop(Exception):
        pass

    class FakeState:
        media_outputs_left = None
        neg_cond = None
        sampling_params = {}

        @property
        def remaining_batch_size(self):
            raise _Stop

    state, fetched, fetches = FakeState(), CondKwargs(), []

    async def fetch_neg_cond(args, sampling_params):
        fetches.append(sampling_params)
        return fetched

    monkeypatch.setattr(rollout_module, "GenerateState", lambda args: state)
    monkeypatch.setattr(rollout_module, "fetch_neg_cond", fetch_neg_cond)
    args = SimpleNamespace(
        rollout_global_dataset=True,
        wandb_log_num_images=0,
        wandb_log_image_interval=1,
        dynamic_sampling_filter_path=None,
        rollout_batch_size=1,
        over_sampling_batch_size=1,
        n_samples_per_prompt=1,
        rollout_neg_cond_guidance_scale=4.5,
    )

    for rollout_id in (0, 1):
        with pytest.raises(_Stop):
            await rollout_module.generate_rollout_async(args, rollout_id, data_source=None)

    assert fetches == [state.sampling_params] and state.neg_cond is fetched


@pytest.mark.asyncio
@pytest.mark.parametrize("call_site", ["microgroup", "custom_generate"])
async def test_a_training_sample_without_a_neg_cond_gets_the_fetched_one(monkeypatch, call_site):
    """A sample left without one stops the teacher's CFG forward, whichever function generated it; overwriting an
    engine's own would swap in a negative cond padded for another prompt."""
    fetched, own = CondKwargs(), CondKwargs()
    state = SimpleNamespace(semaphore=asyncio.Semaphore(1), dp_rank_context=contextlib.nullcontext, neg_cond=fetched)

    async def generate(args, microgroup, sampling_params, evaluation=False):
        return [Sample(denoising_env=DenoisingEnv()), Sample(denoising_env=DenoisingEnv(neg_cond_kwargs=own))]

    monkeypatch.setattr(rollout_module, "GenerateState", lambda args: state)
    monkeypatch.setattr(rollout_module, "generate_microgroup", generate)
    monkeypatch.setattr(rollout_module, "load_function", lambda path: generate)
    args = SimpleNamespace(
        custom_generate_function_path="custom.generate" if call_site == "custom_generate" else None, group_rm=True
    )

    samples = await rollout_module.generate_and_rm_microgroup(args, [Sample(), Sample()], {})

    assert samples[0].denoising_env.neg_cond_kwargs is fetched
    assert samples[1].denoising_env.neg_cond_kwargs is own
