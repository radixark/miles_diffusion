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


def test_response_audio_fills_the_sample():
    audio = torch.zeros(2, 480)
    body = {"generated_audio": audio, "audio_sample_rate": 48000}

    sample = apply_rollout_image_response(Sample(), body, deserialize_func=lambda value: value)

    assert sample.generated_audio is audio
    assert sample.audio_sample_rate == 48000
    silent = apply_rollout_image_response(Sample(), {}, deserialize_func=lambda value: value)
    assert silent.generated_audio is None and silent.audio_sample_rate is None
