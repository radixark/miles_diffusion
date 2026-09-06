from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.utils.processing_utils import cfhw_to_fhwc, fhwc_to_cfhw, image_or_video_to_uint8


def test_cfhw_fhwc_round_trip():
    cfhw = torch.arange(3 * 5 * 8 * 12).reshape(3, 5, 8, 12)

    fhwc = cfhw_to_fhwc(cfhw)

    assert fhwc.shape == (5, 8, 12, 3)
    assert fhwc.is_contiguous()
    torch.testing.assert_close(fhwc_to_cfhw(fhwc), cfhw)


@pytest.mark.parametrize("converter", [cfhw_to_fhwc, fhwc_to_cfhw])
def test_layout_conversion_requires_four_dimensions(converter):
    with pytest.raises(ValueError, match="expected a 4D"):
        converter(torch.zeros(3, 8, 12))


def test_image_or_video_to_uint8_truncates_by_default():
    actual = image_or_video_to_uint8(torch.tensor([0.0, 0.5, 1.0]))

    torch.testing.assert_close(actual, torch.tensor([0, 127, 255], dtype=torch.uint8))


def test_image_or_video_to_uint8_can_round_normalized_values():
    actual = image_or_video_to_uint8(torch.tensor([0.0, 0.5, 1.0]), round_normalized=True)

    torch.testing.assert_close(actual, torch.tensor([0, 128, 255], dtype=torch.uint8))


def test_image_or_video_to_uint8_does_not_round_pixel_values():
    actual = image_or_video_to_uint8(torch.tensor([0.0, 127.9, 255.0]), round_normalized=True)

    torch.testing.assert_close(actual, torch.tensor([0, 127, 255], dtype=torch.uint8))
