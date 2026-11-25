"""Flow-GRPO train pairs and loss over named latent streams.

The converter, prepare hook and loss never name a stream: they carry whatever streams the pairs hold. The streams
here are called "wide" and "narrow" (6 and 4 elements per pair) because no model has them.

    rollout sample ──expand──► pair: per-step fields keyed by stream   {"latent": {"wide": x_w, "narrow": x_n}, ...}
                                 │ flow_grpo_batch: stack each stream across the micro-batch
                                 ▼
    PreparedBatch.latents / timesteps / extras[...]   every one a {stream: tensor} dict
                                 │ flow_grpo_loss_formula
                                 ▼
    per stream:  l_s = mean over the stream's D_s elements of log N(x'_s | mu_s, std_s^2)
    one action:  l   = sum_s (D_s / D) * l_s       (the mean over all D = sum_s D_s elements; here 0.6 and 0.4)
                 r   = exp(l_new - l_old),   loss = sum over pairs of max(-A r, -A clip(r))

What each test pins:
  pairs reach prepare         the converter keys a pair by ``args.primary_stream``, and prepare reads that layout;
                              a stream the run did not ask for stays out of the pair
  converter, extra stream     each requested stream is paired from its own trajectory, grid and debug tensors
  converter, missing stream   a requested stream the response lacks is an error, not a primary-only pair
  prepare, two streams        every stream is stacked into latents, timesteps and each extras field
  prepare, two DiTs           the primary stream's timestep picks the DiT component
  one stream                  loss and gradient equal the single-stream formula bit for bit
  one stream, combination     a lone stream's values pass through the weighting bit for bit
  two streams, loss           loss and both gradients equal a float64 oracle over every element of both streams,
                              unclipped and at a recipe's clip range
  two streams, recomputed     the old log-prob taken from the new prediction gives ratio 1 and the joint gradient
  two streams, KL             each stream's KL is its own per-element mean times its own beta, unweighted
  ode stream, KL              an ode stream's KL is its velocity MSE to the reference, taken in fp32
  recomputed old log-probs    written back per stream, not as one joint value
  per-stream dynamics         each stream steps with its own step backend and noise level
  two cps streams             each scored at its own noise level; the joint gradient equals a float64 oracle
  ode stream                  conditions the step, adds nothing to the score, still reports its model output
  debug diffs                 every stream's rollout model output and transition mean reach the logged diffs
  per-stream diffs            two streams also log each stream's own log-prob and model-output diff, and KL
  one stream, logged keys     a single stream logs exactly the keys it logged before streams existed, with one
                              DiT and with two
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import math
from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils.loss_hub.flow_grpo import (
    _combine_streams,
    _stream_weights,
    flow_grpo_batch,
    flow_grpo_loss_formula,
)
from miles.backends.fsdp_utils.metrics import new_metric_buffer
from miles.backends.fsdp_utils.sde_step_backend import CpsSdeStepBackend, FlowSdeStepBackend
from miles.ray.data_conversion_hub.flow_grpo import expand_samples_to_train_pairs
from miles.utils.types import DenoisingEnv, DiTTrajectory, RolloutDebugTensors, RolloutStreamData, Sample

NOISE_LEVEL = 0.7
# Wide enough that no pair is clipped; the two-stream oracle also runs at a recipe's clip range.
CLIP_RANGE = 100.0
BATCH_SIZE = 3
# stream -> (per-pair latent shape, sigmas, next sigmas, sigma max). Each stream's first pair starts at sigma == 1,
# the one step whose transition reads sigma max, and the two sigma maxes differ.
STREAM_SPECS = {
    "wide": ((2, 3), [1.0, 0.55, 0.3], [0.55, 0.3, 0.1], 0.8),
    "narrow": ((4,), [1.0, 0.7, 0.4], [0.7, 0.4, 0.2], 0.9),
}
STREAM_WEIGHTS = {"wide": 0.6, "narrow": 0.4}
ADVANTAGE = [1.0, -2.0, 0.5]


def _view(tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return tensor.view(-1, *([1] * (like.ndim - 1)))


def _transition_mean_and_std(backend, stream, pred, noise_level=NOISE_LEVEL):
    prev_mean, noise_std, std_dev_t = backend.prev_sample_mean_and_std(
        pred,
        stream.latent.to(pred.dtype),
        _view(stream.sigmas, pred).to(pred.dtype),
        _view(stream.next_sigmas, pred).to(pred.dtype),
        sigma_max=_view(stream.sigma_max, pred).to(pred.dtype),
        noise_level=noise_level,
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
        diffusion_noise_level=dict.fromkeys(STREAM_SPECS, NOISE_LEVEL),
        sde_step_backend_path=dict.fromkeys(STREAM_SPECS, "a backend"),
        diffusion_kl_beta=dict.fromkeys(STREAM_SPECS, 0.0),
        diffusion_score_centering=False,
    )
    ctx = SimpleNamespace(
        args=args,
        sde_backends=dict.fromkeys(STREAM_SPECS, backend),
        models={"dit": object()},
        device=torch.device("cpu"),
    )
    return SimpleNamespace(
        ctx=ctx, backend=backend, streams=streams, new_pred=new_pred, advantage=torch.tensor(ADVANTAGE)
    )


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
        metrics=metrics if metrics is not None else _metrics(stream_names),
        optim_step_idx=0,
        train_output_inputs=[],
        **kwargs,
    )


def _leaf(pred: dict[str, torch.Tensor], dtype=torch.float32) -> dict[str, torch.Tensor]:
    return {stream_name: value.detach().to(dtype).clone().requires_grad_(True) for stream_name, value in pred.items()}


def _clipped_ppo_loss(log_prob_new, log_prob_old, advantage, clip_range=CLIP_RANGE):
    ratio = torch.exp(log_prob_new - log_prob_old)
    return torch.maximum(-advantage * ratio, -advantage * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)).sum()


def _element_log_probs(backend, stream, pred):
    """Gaussian log-density of every element of one stream's transition, [B, D_s], in ``pred``'s dtype."""
    prev_mean, noise_std, _ = _transition_mean_and_std(backend, stream, pred)
    density = (
        -((stream.next_latent.to(pred.dtype) - prev_mean) ** 2) / (2 * noise_std**2)
        - torch.log(noise_std)
        - 0.5 * math.log(2 * math.pi)
    )
    return density.flatten(1)


def _metrics(stream_names, components=("dit",)):
    return new_metric_buffer(None, torch.device("cpu"), components, streams=stream_names)


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
        # Rolled out, but not asked for by this run: it must stay out of the pairs.
        extra_streams={"echo": RolloutStreamData(rollout_log_probs=torch.zeros(num_steps))},
        train_metadata={"sde_step_indices": sde_steps},
    )
    args = SimpleNamespace(primary_stream="main", rollout_extra_streams=[])
    pairs = expand_samples_to_train_pairs(args, [sample], [0.5], [0.5])["train_data"]

    prepared = flow_grpo_batch(_prepare_ctx(), pairs, microbatch_id=0, optim_step_idx=0)

    assert list(prepared.latents) == ["main"]
    assert torch.equal(prepared.latents["main"], latents[sde_steps])
    assert torch.equal(prepared.extras["next_latents"]["main"], latents[[step + 1 for step in sde_steps]])
    assert torch.equal(prepared.extras["log_prob_old"]["main"], sample.rollout_log_probs[sde_steps])
    assert torch.equal(
        torch.stack([pair["rollout_debug_tensors"]["main"]["rollout_step_model_output"] for pair in pairs]),
        sample.rollout_debug_tensors.rollout_model_outputs[sde_steps],
    )


def _sample_with_an_extra_stream(extra_streams: dict) -> Sample:
    num_steps = 4
    return Sample(
        index=7,
        prompt="a prompt",
        rollout_log_probs=torch.linspace(-1.0, -2.0, num_steps),
        denoising_env=DenoisingEnv(),
        dit_trajectory=DiTTrajectory(
            latents=torch.randn(num_steps + 1, 2, 3),
            timesteps=torch.linspace(900.0, 0.0, num_steps + 1),
            sigmas=torch.linspace(0.9, 0.0, num_steps + 1),
        ),
        extra_streams=extra_streams,
        train_metadata={"sde_step_indices": [1, 2]},
    )


def test_converter_pairs_every_requested_stream_from_its_own_trajectory():
    """Indexing an extra stream by the primary's grid or latent window would pair it with the wrong step."""
    num_steps = 4
    # The extra stream returns only the latents of steps 1..3, on its own grid, and is the one with debug tensors.
    echo = RolloutStreamData(
        rollout_log_probs=torch.linspace(-3.0, -4.0, num_steps),
        rollout_debug_tensors=RolloutDebugTensors(rollout_model_outputs=torch.randn(num_steps, 5)),
        dit_trajectory=DiTTrajectory(
            latents=torch.randn(3, 5),
            timesteps=torch.linspace(800.0, 0.0, num_steps + 1),
            sigmas=torch.linspace(0.8, 0.0, num_steps + 1),
            latent_step_indices=torch.tensor([1, 2, 3]),
        ),
    )
    sample = _sample_with_an_extra_stream({"echo": echo})
    args = SimpleNamespace(primary_stream="main", rollout_extra_streams=["echo"])

    pairs = expand_samples_to_train_pairs(args, [sample], [0.5], [0.5])["train_data"]

    assert len(pairs) == 2
    for pair, step in zip(pairs, [1, 2], strict=True):
        assert list(pair["latent"]) == ["main", "echo"]
        assert torch.equal(pair["latent"]["main"], sample.dit_trajectory.latents[step])
        assert torch.equal(pair["next_latent"]["main"], sample.dit_trajectory.latents[step + 1])
        assert torch.equal(pair["latent"]["echo"], echo.dit_trajectory.latents[step - 1])
        assert torch.equal(pair["next_latent"]["echo"], echo.dit_trajectory.latents[step])
        assert pair["timestep"]["echo"] == echo.dit_trajectory.timesteps[step]
        assert pair["sigma"]["echo"] == echo.dit_trajectory.sigmas[step]
        assert pair["next_sigma"]["echo"] == echo.dit_trajectory.sigmas[step + 1]
        assert pair["sigma_max"]["echo"] == echo.dit_trajectory.sigmas[1]
        assert pair["log_prob_old"]["echo"] == echo.rollout_log_probs[step]
        assert pair["sigma"]["main"] == sample.dit_trajectory.sigmas[step]
        assert list(pair["rollout_debug_tensors"]) == ["echo"]
        assert torch.equal(
            pair["rollout_debug_tensors"]["echo"]["rollout_step_model_output"],
            echo.rollout_debug_tensors.rollout_model_outputs[step],
        )


def test_a_requested_stream_the_response_lacks_is_an_error():
    """An engine without extra rollout streams ignores the request field; training would go on primary-only."""
    args = SimpleNamespace(primary_stream="main", rollout_extra_streams=["echo"])

    with pytest.raises(ValueError, match="no rollout data for --rollout-extra-streams"):
        expand_samples_to_train_pairs(args, [_sample_with_an_extra_stream({})], [0.5], [0.5])


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

    prepared = flow_grpo_batch(_prepare_ctx(), pairs, microbatch_id=0, optim_step_idx=0)

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

    prepared = flow_grpo_batch(ctx, [pair], microbatch_id=0, optim_step_idx=0)

    assert prepared.component_name == "high_noise"
    assert prepared.model is ctx.models["high_noise"]


@pytest.mark.parametrize(
    ("kl_beta", "old_log_prob_from_new"),
    [
        pytest.param(0.0, False, id="rollout log-probs"),
        pytest.param(0.5, False, id="with the KL term"),
        pytest.param(0.0, True, id="old log-prob recomputed from the new prediction"),
    ],
)
def test_one_stream_reproduces_the_single_stream_formula_bit_for_bit(rollout, kl_beta, old_log_prob_from_new):
    """Writing the ratio or the KL sum in another arithmetic form moves fp32 values the strict e2e standards compare."""
    backend, stream = rollout.backend, rollout.streams["wide"]
    rollout.ctx.args.diffusion_kl_beta = {"wide": kl_beta}
    # Far from the new prediction: a KL term of the policy loss's size, so one ulp of it shows in the sum.
    ref_pred = {"wide": stream.rollout_pred + 3.0}
    production_pred = _leaf({"wide": rollout.new_pred["wide"]})
    production_loss = _loss(
        rollout, ["wide"], production_pred, ref_pred=ref_pred, old_log_prob_from_new=old_log_prob_from_new
    )
    (production_grad,) = torch.autograd.grad(production_loss, production_pred["wide"])

    def single_stream_step(pred):
        return backend.sde_step_logprob(
            pred,
            stream.latent,
            prev_sample=stream.next_latent,
            noise_level=NOISE_LEVEL,
            sigmas=stream.sigmas,
            next_sigmas=stream.next_sigmas,
            sigma_max=stream.sigma_max,
        )

    reference_pred = _leaf({"wide": rollout.new_pred["wide"]})["wide"]
    step_out = single_stream_step(reference_pred)
    log_prob_old = step_out.log_prob.detach() if old_log_prob_from_new else stream.log_prob_old
    reference_loss = _clipped_ppo_loss(step_out.log_prob, log_prob_old, rollout.advantage)
    if kl_beta > 0:
        kl_per_pair = ((step_out.prev_mean - single_stream_step(ref_pred["wide"]).prev_mean) ** 2).mean(
            dim=tuple(range(1, step_out.prev_mean.ndim)), keepdim=True
        ) / (2 * step_out.std_dev_t**2)
        reference_loss = reference_loss + kl_beta * kl_per_pair.sum()
    (reference_grad,) = torch.autograd.grad(reference_loss, reference_pred)

    assert torch.equal(production_loss, reference_loss)
    assert torch.equal(production_grad, reference_grad)


def test_a_lone_streams_values_pass_through_the_combination_bit_for_bit():
    """Combining as (D * l) / D instead of weight * l moves one fp32 value in seven, for instance a rollout's log-prob."""
    values = torch.randn(1000, generator=torch.Generator().manual_seed(0))
    stream_weights = _stream_weights({"wide": torch.zeros(1, 2, 3)})

    assert torch.equal(_combine_streams({"wide": values}, stream_weights), values)


@pytest.mark.parametrize(
    "clip_range", [pytest.param(CLIP_RANGE, id="no pair clipped"), pytest.param(1e-4, id="a recipe's clip range")]
)
def test_two_streams_match_the_mean_over_every_element_of_both(rollout, clip_range):
    """An unweighted mean of the stream means, swapped weights, or a clip per stream changes the loss or a gradient."""
    backend = rollout.backend
    rollout.ctx.args.diffusion_clip_range = clip_range
    production_pred = _leaf(rollout.new_pred)
    production_loss = _loss(rollout, list(STREAM_SPECS), production_pred)
    production_grads = torch.autograd.grad(production_loss, list(production_pred.values()))

    oracle_pred = _leaf(rollout.new_pred, torch.float64)
    new_elements = [
        _element_log_probs(backend, rollout.streams[stream_name], oracle_pred[stream_name])
        for stream_name in oracle_pred
    ]
    old_elements = [
        _element_log_probs(backend, rollout.streams[stream_name], rollout.streams[stream_name].rollout_pred.double())
        for stream_name in oracle_pred
    ]
    oracle_loss = _clipped_ppo_loss(
        torch.cat(new_elements, dim=1).mean(dim=1),
        torch.cat(old_elements, dim=1).mean(dim=1),
        rollout.advantage,
        clip_range,
    )
    oracle_grads = torch.autograd.grad(oracle_loss, list(oracle_pred.values()))

    torch.testing.assert_close(production_loss.double(), oracle_loss, rtol=1e-5, atol=1e-6)
    for production_grad, oracle_grad in zip(production_grads, oracle_grads, strict=True):
        assert production_grad.abs().sum().item() > 0.0
        torch.testing.assert_close(production_grad.double(), oracle_grad, rtol=1e-4, atol=1e-6)


def test_two_streams_with_the_old_log_prob_taken_from_the_new_prediction(rollout):
    """Without the detach the first window has no gradient; reading the rollout's log-probs here changes its ratio."""
    backend = rollout.backend
    for stream in rollout.streams.values():
        stream.log_prob_old = torch.full_like(stream.log_prob_old, float("nan"))
    production_pred = _leaf(rollout.new_pred)
    production_loss = _loss(rollout, list(STREAM_SPECS), production_pred, old_log_prob_from_new=True)
    production_grads = torch.autograd.grad(production_loss, list(production_pred.values()))

    oracle_pred = _leaf(rollout.new_pred, torch.float64)
    elements = [
        _element_log_probs(backend, rollout.streams[stream_name], oracle_pred[stream_name])
        for stream_name in oracle_pred
    ]
    # The ratio is exactly 1: the loss is -sum(A) and its gradient is that of -sum(A * l_new).
    oracle_grads = torch.autograd.grad(
        -(rollout.advantage * torch.cat(elements, dim=1).mean(dim=1)).sum(), list(oracle_pred.values())
    )

    assert torch.equal(production_loss, -rollout.advantage.sum())
    for production_grad, oracle_grad in zip(production_grads, oracle_grads, strict=True):
        assert production_grad.abs().sum().item() > 0.0
        torch.testing.assert_close(production_grad.double(), oracle_grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "kl_betas",
    [
        pytest.param({"wide": 0.5, "narrow": 2.0}, id="a beta per stream"),
        pytest.param({"wide": 0.0, "narrow": 2.0}, id="only the narrow stream"),
    ],
)
def test_each_streams_kl_takes_its_own_beta(rollout, kl_betas):
    """Weighting the KL by element share would shrink a small stream's beta; one beta would leak into a zero one."""
    backend = rollout.backend
    # Far from the new prediction, so the KL term is of the policy loss's size.
    ref_pred = {stream_name: stream.rollout_pred + 3.0 for stream_name, stream in rollout.streams.items()}
    policy_loss = _loss(rollout, list(STREAM_SPECS), rollout.new_pred)
    rollout.ctx.args.diffusion_kl_beta = kl_betas
    loss = _loss(rollout, list(STREAM_SPECS), rollout.new_pred, ref_pred=ref_pred)

    expected_kl_term = 0.0
    for stream_name, beta in kl_betas.items():
        stream = rollout.streams[stream_name]
        prev_mean, _, std_dev_t = _transition_mean_and_std(backend, stream, rollout.new_pred[stream_name])
        ref_mean, _, _ = _transition_mean_and_std(backend, stream, ref_pred[stream_name])
        per_pair = (prev_mean - ref_mean).square().flatten(1).mean(dim=1) / (2 * std_dev_t.flatten() ** 2)
        expected_kl_term = expected_kl_term + beta * per_pair.sum()
    torch.testing.assert_close(loss - policy_loss, expected_kl_term, rtol=1e-4, atol=1e-8)


def test_an_ode_streams_kl_is_its_velocity_mse(rollout):
    """Taking KL only over the scored streams would leave an ode stream's beta silently without effect."""
    rollout.ctx.args.sde_step_backend_path = {"wide": "a backend"}
    rollout.ctx.sde_backends = {"wide": rollout.backend}
    rollout.ctx.args.diffusion_kl_beta = {"wide": 0.0, "narrow": 2.0}
    ref_pred = {stream_name: stream.rollout_pred + 3.0 for stream_name, stream in rollout.streams.items()}
    pred = _leaf(rollout.new_pred)
    loss = _loss(rollout, list(STREAM_SPECS), pred, ref_pred=ref_pred)
    (narrow_grad,) = torch.autograd.grad(loss, pred["narrow"])

    velocity_diff = rollout.new_pred["narrow"] - ref_pred["narrow"]
    # d/dv of 2.0 * sum over pairs of mean((v - v_ref)^2) over the stream's 4 elements.
    torch.testing.assert_close(narrow_grad, 2.0 * 2 * velocity_diff / velocity_diff[0].numel())


def test_an_ode_streams_kl_subtracts_in_fp32(rollout):
    """Subtracting bf16 predictions first rounds each gap to 8 bits before it is squared."""
    rollout.ctx.args.sde_step_backend_path = {"wide": "a backend"}
    rollout.ctx.sde_backends = {"wide": rollout.backend}
    pred = dict(rollout.new_pred, narrow=rollout.new_pred["narrow"].bfloat16())
    ref_pred = dict(pred, narrow=(rollout.streams["narrow"].rollout_pred + 0.3).bfloat16())
    policy_loss = _loss(rollout, list(STREAM_SPECS), pred)
    rollout.ctx.args.diffusion_kl_beta = {"wide": 0.0, "narrow": 1.0}
    loss = _loss(rollout, list(STREAM_SPECS), pred, ref_pred=ref_pred)

    velocity_diff = pred["narrow"].float() - ref_pred["narrow"].float()
    torch.testing.assert_close(loss - policy_loss, velocity_diff.square().mean(dim=1).sum(), rtol=1e-6, atol=0)


def test_recomputed_old_log_probs_are_written_per_stream(rollout):
    """Writing the joint value back would feed the next window one number where it stacks one per stream."""
    backend = rollout.backend
    batch = [{} for _ in range(BATCH_SIZE)]

    assert _loss(rollout, list(STREAM_SPECS), rollout.new_pred, batch=batch, write_old_log_prob=True) is None

    for stream_name, stream in rollout.streams.items():
        prev_mean, noise_std, _ = _transition_mean_and_std(backend, stream, rollout.new_pred[stream_name])
        expected = backend.log_prob(stream.next_latent, prev_mean, noise_std)
        written = torch.stack([pair["log_prob_old"][stream_name] for pair in batch])
        assert torch.equal(written, expected)


def test_an_ode_stream_conditions_the_step_but_is_not_scored(rollout):
    """Scoring an ode stream adds a zero-noise Gaussian: NaN log-probs and a KL of 0/0."""
    # An ode stream gets no step backend.
    rollout.ctx.args.sde_step_backend_path = {"wide": "a backend"}
    rollout.ctx.sde_backends = {"wide": rollout.backend}
    pred = _leaf(rollout.new_pred)
    metrics = _metrics(list(STREAM_SPECS))
    loss = _loss(
        rollout, list(STREAM_SPECS), pred, batch=_model_output_batch(rollout, list(STREAM_SPECS)), metrics=metrics
    )
    (wide_grad,) = torch.autograd.grad(loss, pred["wide"])

    reference_pred = _leaf({"wide": rollout.new_pred["wide"]})
    reference_loss = _loss(rollout, ["wide"], reference_pred)
    (reference_grad,) = torch.autograd.grad(reference_loss, reference_pred["wide"])

    assert torch.equal(loss, reference_loss)
    assert torch.equal(wide_grad, reference_grad)
    logged = metrics.reduce()
    assert "log_prob_mean_abs_diff_narrow" not in logged
    assert logged["model_output_mean_abs_diff_narrow"] == pytest.approx(
        (rollout.new_pred["narrow"] - rollout.streams["narrow"].rollout_pred).abs().mean().item()
    )


def test_each_stream_steps_with_its_own_backend_and_noise_level(rollout):
    """Reading one stream's backend or noise level for every stream would score the other with the wrong dynamics."""
    noise_levels = {"wide": 0.7, "narrow": 0.3}
    narrow_backend = CpsSdeStepBackend()
    rollout.ctx.args.diffusion_noise_level = noise_levels
    rollout.ctx.sde_backends = {"wide": rollout.backend, "narrow": narrow_backend}
    batch = [{} for _ in range(BATCH_SIZE)]

    _loss(rollout, list(STREAM_SPECS), rollout.new_pred, batch=batch, write_old_log_prob=True)

    for stream_name, backend in rollout.ctx.sde_backends.items():
        stream = rollout.streams[stream_name]
        prev_mean, noise_std, _ = _transition_mean_and_std(
            backend, stream, rollout.new_pred[stream_name], noise_levels[stream_name]
        )
        expected = backend.log_prob(stream.next_latent, prev_mean, noise_std)
        assert torch.equal(torch.stack([pair["log_prob_old"][stream_name] for pair in batch]), expected)


def test_constant_free_streams_are_scored_at_their_own_noise_levels(rollout):
    """Scoring both streams at one noise level moves the other stream's transition mean, and the joint ratio with it."""
    backend, noise_levels = CpsSdeStepBackend(), {"wide": 0.8, "narrow": 0.3}
    rollout.ctx.sde_backends = dict.fromkeys(STREAM_SPECS, backend)
    rollout.ctx.args.diffusion_noise_level = noise_levels
    production_pred = _leaf(rollout.new_pred)
    production_loss = _loss(rollout, list(STREAM_SPECS), production_pred, old_log_prob_from_new=True)
    production_grads = torch.autograd.grad(production_loss, list(production_pred.values()))

    oracle_pred = _leaf(rollout.new_pred, torch.float64)
    log_prob = 0.0
    for stream_name, weight in STREAM_WEIGHTS.items():
        stream = rollout.streams[stream_name]
        prev_mean, _, _ = _transition_mean_and_std(
            backend, stream, oracle_pred[stream_name], noise_levels[stream_name]
        )
        log_prob = log_prob + weight * (-((stream.next_latent.double() - prev_mean) ** 2)).flatten(1).mean(dim=1)
    oracle_loss = _clipped_ppo_loss(log_prob, log_prob.detach(), rollout.advantage.double())
    oracle_grads = torch.autograd.grad(oracle_loss, list(oracle_pred.values()))

    for production_grad, oracle_grad in zip(production_grads, oracle_grads, strict=True):
        assert production_grad.abs().sum().item() > 0.0
        torch.testing.assert_close(production_grad.double(), oracle_grad, rtol=1e-4, atol=1e-6)


def test_every_streams_rollout_model_output_reaches_the_logged_diff(rollout):
    """A debug lookup that misses a stream returns None and the train-vs-rollout diff silently stops being logged."""
    backend = rollout.backend
    rollout_means = {
        stream_name: _transition_mean_and_std(backend, stream, stream.rollout_pred)[0]
        for stream_name, stream in rollout.streams.items()
    }
    batch = [
        {
            "rollout_debug_tensors": {
                stream_name: {
                    "rollout_step_model_output": stream.rollout_pred[index],
                    "rollout_step_prev_sample_mean": rollout_means[stream_name][index],
                }
                for stream_name, stream in rollout.streams.items()
            }
        }
        for index in range(BATCH_SIZE)
    ]
    metrics = _metrics(list(STREAM_SPECS))

    _loss(rollout, list(STREAM_SPECS), rollout.new_pred, batch=batch, metrics=metrics)

    diffs = [
        (rollout.new_pred[stream_name] - stream.rollout_pred).abs() for stream_name, stream in rollout.streams.items()
    ]
    every_element = torch.cat([diff.flatten() for diff in diffs])
    logged = metrics.reduce()
    assert logged["model_output_mean_abs_diff"] == pytest.approx(every_element.mean().item())
    assert logged["model_output_max_abs_diff"] == pytest.approx(every_element.max().item())
    # One value per pair of each stream: the transition-mean gap's RMS, in units of that stream's noise std.
    mean_gaps = []
    for stream_name, stream in rollout.streams.items():
        prev_mean, noise_std, _ = _transition_mean_and_std(backend, stream, rollout.new_pred[stream_name])
        mean_gaps.append(((rollout_means[stream_name] - prev_mean) / noise_std).flatten(1).square().mean(dim=1).sqrt())
    assert logged["prev_mean_diff_over_noise_std"] == pytest.approx(torch.cat(mean_gaps).mean().item())


def _model_output_batch(rollout, stream_names):
    return [
        {
            "rollout_debug_tensors": {
                stream_name: {"rollout_step_model_output": rollout.streams[stream_name].rollout_pred[index]}
                for stream_name in stream_names
            }
        }
        for index in range(BATCH_SIZE)
    ]


def test_two_streams_also_log_each_streams_own_diffs(rollout):
    """The joint diff is almost all the larger stream; a stream replayed wrongly only shows in its own key."""
    backend, stream_names = rollout.backend, list(STREAM_SPECS)
    metrics = _metrics(stream_names)
    rollout.ctx.args.diffusion_kl_beta = {"wide": 0.0, "narrow": 2.0}
    ref_pred = {stream_name: stream.rollout_pred + 3.0 for stream_name, stream in rollout.streams.items()}

    _loss(
        rollout,
        stream_names,
        rollout.new_pred,
        batch=_model_output_batch(rollout, stream_names),
        metrics=metrics,
        ref_pred=ref_pred,
    )

    logged = metrics.reduce()
    for stream_name, stream in rollout.streams.items():
        prev_mean, noise_std, _ = _transition_mean_and_std(backend, stream, rollout.new_pred[stream_name])
        log_prob_new = backend.log_prob(stream.next_latent, prev_mean, noise_std)
        assert logged[f"log_prob_mean_abs_diff_{stream_name}"] == pytest.approx(
            (log_prob_new - stream.log_prob_old).abs().mean().item()
        )
        assert logged[f"model_output_mean_abs_diff_{stream_name}"] == pytest.approx(
            (rollout.new_pred[stream_name] - stream.rollout_pred).abs().mean().item()
        )
    narrow = rollout.streams["narrow"]
    prev_mean, _, std_dev_t = _transition_mean_and_std(backend, narrow, rollout.new_pred["narrow"])
    ref_mean, _, _ = _transition_mean_and_std(backend, narrow, ref_pred["narrow"])
    narrow_kl = (prev_mean - ref_mean).square().flatten(1).mean(dim=1) / (2 * std_dev_t.flatten() ** 2)
    assert logged["kl_loss_narrow"] == pytest.approx(narrow_kl.mean().item(), rel=1e-5)
    # A zero-beta stream adds nothing to the total either.
    assert logged["kl_loss"] == pytest.approx(narrow_kl.mean().item(), rel=1e-5)
    assert "kl_loss_wide" not in logged


@pytest.mark.parametrize(
    ("components", "component_keys"),
    [
        pytest.param(("dit",), [], id="one DiT"),
        pytest.param(
            ("dit", "dit_2"), ["log_prob_mean_abs_diff_dit", "model_output_mean_abs_diff_dit"], id="two DiTs"
        ),
    ],
)
def test_one_stream_logs_the_keys_it_logged_before_streams_existed(rollout, components, component_keys):
    """A per-stream key leaking into a single-stream run, or a lost per-DiT key, changes every existing run's logs."""
    rollout.ctx.models = dict.fromkeys(components)
    metrics = _metrics(["wide"], components)
    batch = _model_output_batch(rollout, ["wide"])

    _loss(rollout, ["wide"], {"wide": rollout.new_pred["wide"]}, batch=batch, metrics=metrics)

    assert sorted(metrics.reduce()) == sorted(
        [
            *component_keys,
            "adv_abs_mean",
            "approx_kl",
            "clipfrac",
            "kl_loss",
            "log_prob_mean_abs_diff",
            "log_prob_new_idx_0",
            "log_prob_old_idx_0",
            "loss",
            "loss_abs_mean",
            "model_output_max_abs_diff",
            "model_output_mean_abs_diff",
            "model_output_rel_max",
            "policy_loss",
            "ratio_abs_minus_1",
        ]
    )
