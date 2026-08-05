"""Image branch of the SFT media reader."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import numpy as np
import pytest
import torch
from PIL import Image

from miles.rollout.sft_rollout import read_media_clip


def _write_png(path, height, width):
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(path)


def test_image_reads_as_single_frame(tmp_path):
    path = tmp_path / "a.png"
    _write_png(path, 120, 200)
    clip = read_media_clip(str(path), height=64, width=64, num_frames=1, frame_stride=1)
    assert clip.shape == (3, 1, 64, 64)
    assert clip.min() >= -1.0 and clip.max() <= 1.0


def test_image_rejects_multi_frame(tmp_path):
    path = tmp_path / "a.png"
    _write_png(path, 64, 64)
    with pytest.raises(ValueError, match="requires --diffusion-output-num-frames 1"):
        read_media_clip(str(path), height=64, width=64, num_frames=5, frame_stride=1)


def test_image_no_resize_when_exact(tmp_path):
    path = tmp_path / "a.png"
    _write_png(path, 64, 64)
    clip = read_media_clip(str(path), height=64, width=64, num_frames=1, frame_stride=1)
    original = torch.from_numpy(np.asarray(Image.open(path))).permute(2, 0, 1).float() / 127.5 - 1.0
    assert torch.allclose(clip[:, 0], original)
