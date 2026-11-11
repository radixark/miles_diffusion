from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.utils.diffusion_rollout_response import _normalize_generated_output


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
