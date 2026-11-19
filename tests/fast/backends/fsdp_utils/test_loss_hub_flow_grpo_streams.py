"""Flow-GRPO train pairs keyed by latent stream.

The converter and the prepare hook never name a stream: they carry whatever streams the pairs hold. The streams
here are called "wide" and "narrow" (6 and 4 elements per pair) because no model has them.

    rollout sample ──expand──► pair: per-step fields keyed by stream   {"latent": {"wide": x_w, "narrow": x_n}, ...}
                                 │ prepare_flow_grpo_batch: stack each stream across the micro-batch
                                 ▼
    PreparedBatch.latents / timesteps / extras[...]   every one a {stream: tensor} dict
                                 │ flow_grpo_loss_formula, on the pairs' one stream
                                 ▼
    a recomputed old log-prob goes back into the pair under its stream's name

What each test pins:
  pairs reach prepare         the converter keys a pair by ``args.primary_stream``, and prepare reads that layout
  prepare, two streams        every stream is stacked into latents, timesteps and each extras field
  prepare, two DiTs           the primary stream's timestep picks the DiT component
  recomputed old log-prob     written back under its stream, where the next window's prepare reads it
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils.loss_hub.flow_grpo import flow_grpo_loss_formula, prepare_flow_grpo_batch
from miles.backends.fsdp_utils.metrics import new_metric_buffer
from miles.backends.fsdp_utils.sde_step_backend import FlowSdeStepBackend
from miles.ray.data_conversion_hub.flow_grpo import expand_samples_to_train_pairs
from miles.utils.types import DenoisingEnv, DiTTrajectory, RolloutDebugTensors, Sample

NOISE_LEVEL = 0.7
CLIP_RANGE = 0.2
BATCH_SIZE = 3
# stream -> (per-pair latent shape, sigmas, next sigmas, sigma max)
STREAM_SPECS = {
    "wide": ((2, 3), [0.8, 0.55, 0.3], [0.55, 0.3, 0.1], 0.8),
    "narrow": ((4,), [0.9, 0.7, 0.4], [0.7, 0.4, 0.2], 0.9),
}
ADVANTAGE = [1.0, -2.0, 0.5]


def _view(tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return tensor.view(-1, *([1] * (like.ndim - 1)))


def _transition_mean_and_std(backend, stream, pred):
    prev_mean, noise_std, std_dev_t = backend.prev_sample_mean_and_std(
        pred,
        stream.latent.to(pred.dtype),
        _view(stream.sigmas, pred).to(pred.dtype),
        _view(stream.next_sigmas, pred).to(pred.dtype),
        sigma_max=_view(stream.sigma_max, pred).to(pred.dtype),
        noise_level=NOISE_LEVEL,
    )
    return prev_mean, noise_std, std_dev_t


@pytest.fixture
def rollout():
    """Three pairs of two streams, each sampled around its own rollout mean on its own sigmas."""
    generator = torch.Generator().manual_seed(0)
    backend = FlowSdeStepBackend()
    streams, new_pred = {}, {}
    for stream_name, (shape, sigmas, next_sigmas, sigma_max) in STREAM_SPECS.items():
        stream = SimpleNamespace(
            latent=torch.randn((BATCH_SIZE, *shape), generator=generator),
            sigmas=torch.tensor(sigmas),
            next_sigmas=torch.tensor(next_sigmas),
            sigma_max=torch.full((BATCH_SIZE,), sigma_max),
            rollout_pred=torch.randn((BATCH_SIZE, *shape), generator=generator),
        )
        rollout_mean, noise_std, _ = _transition_mean_and_std(backend, stream, stream.rollout_pred)
        stream.next_latent = rollout_mean + noise_std * torch.randn((BATCH_SIZE, *shape), generator=generator)
        stream.log_prob_old = backend.log_prob(stream.next_latent, rollout_mean, noise_std)
        streams[stream_name] = stream
        new_pred[stream_name] = stream.rollout_pred + 0.05 * torch.randn((BATCH_SIZE, *shape), generator=generator)
    args = SimpleNamespace(
        diffusion_clip_range=CLIP_RANGE,
        diffusion_noise_level=NOISE_LEVEL,
        diffusion_kl_beta=0.0,
        diffusion_score_centering=False,
    )
    ctx = SimpleNamespace(args=args, sde_backend=backend, models={"dit": object()}, device=torch.device("cpu"))
    return SimpleNamespace(ctx=ctx, streams=streams, new_pred=new_pred, advantage=torch.tensor(ADVANTAGE))


def _prepared(rollout, stream_names):
    streams = {stream_name: rollout.streams[stream_name] for stream_name in stream_names}
    return SimpleNamespace(
        latents={stream_name: stream.latent for stream_name, stream in streams.items()},
        advantage=rollout.advantage,
        component_name="dit",
        extras={
            "next_latents": {stream_name: stream.next_latent for stream_name, stream in streams.items()},
            "sigmas": {stream_name: stream.sigmas for stream_name, stream in streams.items()},
            "next_sigmas": {stream_name: stream.next_sigmas for stream_name, stream in streams.items()},
            "sigma_max": {stream_name: stream.sigma_max for stream_name, stream in streams.items()},
            "log_prob_old": {stream_name: stream.log_prob_old for stream_name, stream in streams.items()},
        },
    )


def _loss(rollout, stream_names, new_pred, *, batch=None, metrics=None, ref_pred=None, **kwargs):
    return flow_grpo_loss_formula(
        rollout.ctx,
        batch if batch is not None else [{} for _ in range(BATCH_SIZE)],
        _prepared(rollout, stream_names),
        new_pred=new_pred,
        old_pred=None,
        ref_pred=ref_pred,
        metrics=metrics if metrics is not None else new_metric_buffer(None, torch.device("cpu"), ["dit"]),
        optim_step_idx=0,
        train_output_inputs=[],
        **kwargs,
    )


def _prepare_ctx():
    config = SimpleNamespace(
        cfg_batching=False,
        process_timestep_as_input=lambda timesteps: timesteps / 1000.0,
        prepare_cond_kwargs=lambda cond, device: {},
        collate_cond_for_sample_batch=lambda conds, device, pad_to_len=None: {},
    )
    args = SimpleNamespace(diffusion_adv_clip_max=5.0, diffusion_guidance_scale=1.0, diffusion_true_cfg_scale=None)
    return SimpleNamespace(
        args=args, device=torch.device("cpu"), train_pipeline_config=config, models={"dit": object()}
    )


def test_converted_pairs_are_keyed_by_the_primary_stream_and_prepare_reads_them():
    """A converter that names the stream itself, or keys a field unlike prepare, fails here and not on a GPU run."""
    num_steps, sde_steps = 4, [1, 2]
    latents = torch.randn(num_steps + 1, 2, 3)
    sample = Sample(
        index=0,
        prompt="a prompt",
        rollout_log_probs=torch.linspace(-1.0, -2.0, num_steps),
        rollout_debug_tensors=RolloutDebugTensors(rollout_model_outputs=torch.randn(num_steps, 2, 3)),
        denoising_env=DenoisingEnv(),
        dit_trajectory=DiTTrajectory(
            latents=latents,
            timesteps=torch.linspace(900.0, 0.0, num_steps + 1),
            sigmas=torch.linspace(0.9, 0.0, num_steps + 1),
        ),
        train_metadata={"sde_step_indices": sde_steps},
    )
    args = SimpleNamespace(primary_stream="main")
    pairs = expand_samples_to_train_pairs(args, [sample], [0.5], [0.5])["train_data"]

    prepared = prepare_flow_grpo_batch(_prepare_ctx(), pairs, microbatch_id=0, optim_step_idx=0)

    assert list(prepared.latents) == ["main"]
    assert torch.equal(prepared.latents["main"], latents[sde_steps])
    assert torch.equal(prepared.extras["next_latents"]["main"], latents[[step + 1 for step in sde_steps]])
    assert torch.equal(prepared.extras["log_prob_old"]["main"], sample.rollout_log_probs[sde_steps])
    assert torch.equal(
        torch.stack([pair["rollout_debug_tensors"]["main"]["rollout_step_model_output"] for pair in pairs]),
        sample.rollout_debug_tensors.rollout_model_outputs[sde_steps],
    )


def test_prepare_stacks_every_stream_of_the_pairs(rollout):
    """Reading one stream by name would drop the other's latents, sigmas and old log-probs without an error."""
    pairs = [
        {
            "advantage": ADVANTAGE[index],
            "denoising_env": DenoisingEnv(),
            "latent": {stream_name: stream.latent[index] for stream_name, stream in rollout.streams.items()},
            "next_latent": {stream_name: stream.next_latent[index] for stream_name, stream in rollout.streams.items()},
            "timestep": {
                stream_name: stream.sigmas[index] * 1000.0 for stream_name, stream in rollout.streams.items()
            },
            "sigma": {stream_name: stream.sigmas[index] for stream_name, stream in rollout.streams.items()},
            "next_sigma": {stream_name: stream.next_sigmas[index] for stream_name, stream in rollout.streams.items()},
            "sigma_max": {stream_name: stream.sigma_max[index] for stream_name, stream in rollout.streams.items()},
            "log_prob_old": {
                stream_name: stream.log_prob_old[index] for stream_name, stream in rollout.streams.items()
            },
        }
        for index in range(BATCH_SIZE)
    ]

    prepared = prepare_flow_grpo_batch(_prepare_ctx(), pairs, microbatch_id=0, optim_step_idx=0)

    for stream_name, stream in rollout.streams.items():
        assert torch.equal(prepared.latents[stream_name], stream.latent)
        assert torch.equal(prepared.timesteps[stream_name], stream.sigmas * 1000.0)
        assert torch.equal(prepared.timesteps_for_model[stream_name], prepared.timesteps[stream_name] / 1000.0)
        assert torch.equal(prepared.extras["next_latents"][stream_name], stream.next_latent)
        assert torch.equal(prepared.extras["sigmas"][stream_name], stream.sigmas)
        assert torch.equal(prepared.extras["next_sigmas"][stream_name], stream.next_sigmas)
        assert torch.equal(prepared.extras["sigma_max"][stream_name], stream.sigma_max)
        assert torch.equal(prepared.extras["log_prob_old"][stream_name], stream.log_prob_old)


def test_a_multi_dit_family_routes_by_the_primary_streams_timestep(rollout):
    """Routing by another stream's timestep would train the wrong DiT of a two-phase family."""
    ctx = _prepare_ctx()
    ctx.args.primary_stream = "narrow"
    ctx.args.diffusion_guidance_scale_2 = None
    ctx.models = {"high_noise": object(), "low_noise": object()}
    ctx.train_pipeline_config.component_for_timestep = lambda timestep: (
        "high_noise" if timestep >= 500 else "low_noise"
    )
    ctx.train_pipeline_config.select_guidance_scale = lambda timestep, guidance_scale, guidance_scale_2: guidance_scale
    # The two streams sit in different phases: 100 is low-noise, 900 is high-noise.
    timestep = {"wide": torch.tensor(100.0), "narrow": torch.tensor(900.0)}
    pair = {
        "advantage": 1.0,
        "denoising_env": DenoisingEnv(),
        "latent": {stream_name: stream.latent[0] for stream_name, stream in rollout.streams.items()},
        "next_latent": {stream_name: stream.next_latent[0] for stream_name, stream in rollout.streams.items()},
        "timestep": timestep,
        "sigma": {stream_name: stream.sigmas[0] for stream_name, stream in rollout.streams.items()},
        "next_sigma": {stream_name: stream.next_sigmas[0] for stream_name, stream in rollout.streams.items()},
        "sigma_max": {stream_name: stream.sigma_max[0] for stream_name, stream in rollout.streams.items()},
        "log_prob_old": {stream_name: stream.log_prob_old[0] for stream_name, stream in rollout.streams.items()},
    }

    prepared = prepare_flow_grpo_batch(ctx, [pair], microbatch_id=0, optim_step_idx=0)

    assert prepared.component_name == "high_noise"
    assert prepared.model is ctx.models["high_noise"]


def test_recomputed_old_log_prob_is_written_back_under_its_stream(rollout):
    """A flat tensor written back would break the next window's prepare, which stacks ``pair[key][stream]``."""
    backend, stream = rollout.ctx.sde_backend, rollout.streams["wide"]
    batch = [{} for _ in range(BATCH_SIZE)]

    assert _loss(rollout, ["wide"], {"wide": rollout.new_pred["wide"]}, batch=batch, write_old_log_prob=True) is None

    prev_mean, noise_std, _ = _transition_mean_and_std(backend, stream, rollout.new_pred["wide"])
    expected = backend.log_prob(stream.next_latent, prev_mean, noise_std)
    assert torch.equal(torch.stack([pair["log_prob_old"]["wide"] for pair in batch]), expected)
