"""Per-sample content addressing for the SFT cache."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import os
from argparse import Namespace

from miles.rollout.sft_rollout import sft_sample_key


def _args(**overrides):
    base = dict(
        sft_encoder_checkpoint="ckpt",
        diffusion_height=480,
        diffusion_width=832,
        diffusion_output_num_frames=81,
        sft_frame_stride=2,
    )
    base.update(overrides)
    return Namespace(**base)


def test_key_is_deterministic_with_seed(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x" * 100)
    item = {"media": str(video), "prompt": "p"}

    name, seed = sft_sample_key(_args(), item)
    assert (name, seed) == sft_sample_key(_args(), item)
    assert name.endswith(".pt")
    assert 0 <= seed < 2**63


def test_key_invalidates_per_axis(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x" * 100)
    item = {"media": str(video), "prompt": "p"}
    base_name, base_seed = sft_sample_key(_args(), item)

    assert sft_sample_key(_args(diffusion_height=512), item)[0] != base_name
    assert sft_sample_key(_args(sft_frame_stride=1), item)[0] != base_name
    assert sft_sample_key(_args(sft_encoder_checkpoint="other"), item)[0] != base_name
    assert sft_sample_key(_args(), {"media": str(video), "prompt": "q"})[0] != base_name

    video.write_bytes(b"y" * 101)
    assert sft_sample_key(_args(), item)[0] != base_name

    video.write_bytes(b"x" * 100)
    os.utime(video, ns=(1, 1))
    replaced_name, replaced_seed = sft_sample_key(_args(), item)
    assert replaced_name != base_name
    assert replaced_seed != base_seed


def test_key_is_per_sample(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"x" * 100)
    b.write_bytes(b"x" * 100)
    name_a, _ = sft_sample_key(_args(), {"media": str(a), "prompt": "p"})
    name_b, _ = sft_sample_key(_args(), {"media": str(b), "prompt": "p"})
    assert name_a != name_b
