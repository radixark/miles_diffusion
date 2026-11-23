"""Parsing one sample's rollout response into a ``Sample``.

    engine response                                   Sample
      generated_output                             ──►  normalized to [C, F, H, W]
      {"__store__": field} markers                 ──►  StoredTensor handles of the response's ref
      generated_audio, audio_sample_rate           ──►  the same fields
      rollout_log_probs / dit_trajectory / ...     ──►  the same fields            (primary stream)
      extra_streams:
        audio: rolled out (log-probs, trajectory)  ──►  extra_streams["audio"]
        audio: only its decoded waveform           ──►  left out
      extra_streams: None (no audio pipeline)      ──►  extra_streams == {}

What each test pins:
  output layout               CHW, HWC, CFHW and FHWC all become [C, F, H, W]
  ambiguous layout            first and last dimension both look like channels: rejected
  no channel dimension        rejected
  store markers               the primary stream's markers become StoredTensors of the response's ref
  response audio              the audio fields fill the sample
  extra stream, engine-built  a stream the engine rolled out reaches the sample, tensor by tensor
  stream not rolled out       an entry without rollout data is not a sample stream
  extra stream, stored        an extra stream's markers become StoredTensors of the response's ref
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.utils.diffusion_rollout_response import _normalize_generated_output, apply_rollout_image_response
from miles.utils.types import Sample


@pytest.mark.parametrize(
    ("shape", "permutation"),
    [
        ((3, 8, 12), None),
        ((8, 12, 3), (2, 0, 1)),
        ((3, 5, 8, 12), None),
        ((5, 8, 12, 3), (3, 0, 1, 2)),
    ],
)
def test_normalize_generated_output_to_cfhw(shape, permutation):
    output = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    expected = output if permutation is None else output.permute(permutation)
    if expected.ndim == 3:
        expected = expected.unsqueeze(1)

    actual = _normalize_generated_output(output)

    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected)


def test_normalize_generated_output_rejects_ambiguous_layout():
    with pytest.raises(ValueError, match="layout is ambiguous"):
        _normalize_generated_output(torch.zeros(3, 8, 4))


def test_normalize_generated_output_rejects_missing_channel_dimension():
    with pytest.raises(ValueError, match="no recognizable channel dimension"):
        _normalize_generated_output(torch.zeros(5, 8, 12))


def test_store_markers_become_stored_tensors_of_the_response_ref():
    from miles.utils.diffusion_rollout_response import apply_rollout_image_response
    from miles.utils.output_store import StoredTensor
    from miles.utils.types import Sample

    ref = {
        "backend": "mooncake",
        "key": "prefix/uuid",
        "fields": [
            *(
                {
                    "name": f"dit_trajectory.latents.1.{step}",
                    "dtype": "bfloat16",
                    "shape": [4, 3],
                    "codec": "identity",
                    "nbytes": 24,
                }
                for step in (2, 3)
            ),
            {
                "name": "denoising_env.pos_cond_kwargs.pooled",
                "dtype": "float32",
                "shape": [3],
                "codec": "identity",
                "nbytes": 12,
            },
        ],
    }
    body = {
        "request_id": "r",
        "output_store_ref": ref,
        "dit_trajectory": {
            "latents": [{"__store__": "dit_trajectory.latents.1.2"}, {"__store__": "dit_trajectory.latents.1.3"}],
            "timesteps": {"__tensor__": True, "data": _tensor_bytes(torch.tensor([1.0, 0.0]))},
            "sigmas": None,
            "latent_step_indices": None,
        },
        "denoising_env": {
            "pos_cond_kwargs": {
                "encoder_hidden_states": [{"__store__": "denoising_env.pos_cond_kwargs.encoder_hidden_states.0.1"}],
                "pooled_projections": {"__store__": "denoising_env.pos_cond_kwargs.pooled"},
                "h3_packed_layout": {"ids": {"__store__": "denoising_env.pos_cond_kwargs.h3_packed_layout.ids.1"}},
                "txt_seq_lens": [7],
            },
        },
    }
    sample = apply_rollout_image_response(Sample(index=0), body)
    assert sample.rollout_store_ref.key == "prefix/uuid"
    latents = sample.dit_trajectory.latents
    assert all(isinstance(latent, StoredTensor) for latent in latents)
    assert [latent.field for latent in latents] == ["dit_trajectory.latents.1.2", "dit_trajectory.latents.1.3"]
    assert torch.equal(sample.dit_trajectory.timesteps, torch.tensor([1.0, 0.0]))
    cond = sample.denoising_env.pos_cond_kwargs
    assert cond.encoder_hidden_states[0].field == "denoising_env.pos_cond_kwargs.encoder_hidden_states.0.1"
    assert cond.pooled_projections[0].field == "denoising_env.pos_cond_kwargs.pooled"
    assert cond.h3_packed_layout["ids"].field == "denoising_env.pos_cond_kwargs.h3_packed_layout.ids.1"
    assert cond.txt_seq_lens == [7]


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    from safetensors.torch import save

    return save({"t": tensor})


def test_response_audio_fills_the_sample():
    audio = torch.zeros(2, 480)
    body = {"generated_audio": audio, "audio_sample_rate": 48000}

    sample = apply_rollout_image_response(Sample(), body, deserialize_func=lambda value: value)

    assert sample.generated_audio is audio
    assert sample.audio_sample_rate == 48000
    silent = apply_rollout_image_response(Sample(), {}, deserialize_func=lambda value: value)
    assert silent.generated_audio is None and silent.audio_sample_rate is None


def _engine_response_bodies(*, extra_stream_rolled_out: bool) -> tuple[list[dict], dict]:
    """Two samples' response bodies as the engine builds them, and the audio stream's batched rollout tensors."""
    from sglang.multimodal_gen.configs.post_training.rl_rollout import AUDIO_STREAM, PRIMARY_STREAM
    from sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api import _build_response
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
    from sglang.multimodal_gen.runtime.post_training import rl_dataclasses as engine

    generator = torch.Generator().manual_seed(0)
    batch_size, num_steps = 2, 3

    def stream(*latent_shape):
        return engine.RolloutStreamData(
            rollout_log_probs=torch.randn(batch_size, num_steps, generator=generator),
            rollout_debug_tensors=engine.RolloutDebugTensors(
                rollout_model_outputs=torch.randn(batch_size, num_steps, *latent_shape, generator=generator),
                rollout_noise_std_devs=torch.rand(batch_size, num_steps, 1, generator=generator),
            ),
            dit_trajectory=engine.RolloutDitTrajectory(
                latents=torch.randn(batch_size, 2, *latent_shape, generator=generator),
                timesteps=torch.linspace(1000.0, 0.0, num_steps + 1),
                sigmas=torch.rand(num_steps + 1, generator=generator),
                latent_step_indices=torch.tensor([1, 2]),
            ),
        )

    streams = {PRIMARY_STREAM: stream(4, 6)}
    if extra_stream_rolled_out:
        streams[AUDIO_STREAM] = stream(8)
    result = OutputBatch(
        output=torch.rand(batch_size, 3, 1, 8, 8, generator=generator),
        audio=torch.rand(batch_size, 2, 16, generator=generator),
        audio_sample_rate=48000,
        rollout_trajectory_data=engine.RolloutTrajectoryData(denoising_env=None, streams=streams),
    )
    bodies = [response.model_dump() for response in _build_response("request", "a prompt", 0, True, result)]
    return bodies, streams.get(AUDIO_STREAM)


def test_an_extra_stream_the_engine_rolled_out_reaches_the_sample():
    """A parser that reads a stream's field from the wrong key fails here and not as a wrong ratio on a GPU run."""
    bodies, audio = _engine_response_bodies(extra_stream_rolled_out=True)

    samples = [apply_rollout_image_response(Sample(index=index), body) for index, body in enumerate(bodies)]

    for index, sample in enumerate(samples):
        (stream_name,) = sample.extra_streams
        parsed = sample.extra_streams[stream_name]
        assert stream_name == "audio"
        assert torch.equal(parsed.rollout_log_probs, audio.rollout_log_probs[index])
        assert torch.equal(parsed.dit_trajectory.latents, audio.dit_trajectory.latents[index])
        assert torch.equal(parsed.dit_trajectory.timesteps, audio.dit_trajectory.timesteps)
        assert torch.equal(parsed.dit_trajectory.sigmas, audio.dit_trajectory.sigmas)
        assert torch.equal(parsed.dit_trajectory.latent_step_indices, audio.dit_trajectory.latent_step_indices)
        debug_tensors = parsed.rollout_debug_tensors
        assert torch.equal(
            debug_tensors.rollout_model_outputs, audio.rollout_debug_tensors.rollout_model_outputs[index]
        )
        assert torch.equal(
            debug_tensors.rollout_noise_std_devs, audio.rollout_debug_tensors.rollout_noise_std_devs[index]
        )
        assert debug_tensors.rollout_prev_sample_means is None
        # The primary stream stays at the top level, with its own shapes.
        assert sample.dit_trajectory.latents.shape == (2, 4, 6)


def _audio_output_only(generated_output) -> dict:
    return {
        "audio": {
            "generated_output": generated_output,
            "sample_rate": 48000,
            "rollout_log_probs": None,
            "rollout_debug_tensors": None,
            "dit_trajectory": None,
        }
    }


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"extra_streams": None}, id="a pipeline without extra streams"),
        pytest.param(_engine_response_bodies, id="the engine's video-only rollout, which still decodes audio"),
        pytest.param(
            {"extra_streams": _audio_output_only({"__tensor__": True, "data": b""})}, id="a waveform-only entry"
        ),
    ],
)
def test_a_stream_that_was_not_rolled_out_is_not_a_sample_stream(body):
    """Keeping a waveform-only entry would hand every video-only run a second, empty stream to train on."""
    if callable(body):
        body = body(extra_stream_rolled_out=False)[0][0]
        assert body["extra_streams"]["audio"]["generated_output"] is not None

    sample = apply_rollout_image_response(Sample(index=0), body)

    assert sample.extra_streams == {}


def test_an_extra_streams_stored_tensors_belong_to_the_response_ref():
    """With the output store on, an extra stream's tensors must stay handles the trainer reads, like the primary's."""
    from miles.utils.output_store import StoredTensor

    fields = ["extra_streams.audio.rollout_log_probs.0", "extra_streams.audio.dit_trajectory.latents.0.1"]
    body = {
        "output_store_ref": {
            "backend": "mooncake",
            "key": "prefix/uuid",
            "fields": [
                {"name": name, "dtype": "float32", "shape": [3], "codec": "identity", "nbytes": 12} for name in fields
            ],
        },
        "extra_streams": {
            "audio": {
                "generated_output": None,
                "rollout_log_probs": {"__store__": fields[0]},
                "rollout_debug_tensors": None,
                "dit_trajectory": {
                    "latents": [{"__store__": fields[1]}],
                    "timesteps": {"__tensor__": True, "data": _tensor_bytes(torch.tensor([1.0, 0.0]))},
                    "sigmas": None,
                    "latent_step_indices": None,
                },
            }
        },
    }

    sample = apply_rollout_image_response(Sample(index=0), body)

    stream = sample.extra_streams["audio"]
    assert stream.rollout_log_probs == StoredTensor(sample.rollout_store_ref, fields[0])
    assert stream.dit_trajectory.latents == [StoredTensor(sample.rollout_store_ref, fields[1])]
    assert torch.equal(stream.dit_trajectory.timesteps, torch.tensor([1.0, 0.0]))
