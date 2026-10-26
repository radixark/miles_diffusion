"""Logged and saved media: videos play at --diffusion-fps and carry the engine's audio; eval media can be kept on disk."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import json
import subprocess
from argparse import Namespace
from types import SimpleNamespace

import imageio_ffmpeg
import pytest
import torch

from miles.ray import rollout
from miles.utils.types import Sample


def _streams(path):
    probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", path], capture_output=True, text=True)
    return {kind for kind in ("Video", "Audio") if f": {kind}:" in probe.stderr}


def _video_sample(**fields):
    return Sample(prompt="a boat", seed=7, generated_output=torch.rand(3, 12, 16, 16), **fields)


def test_video_plays_at_the_given_fps(tmp_path):
    path = rollout.write_sample_media(_video_sample(), str(tmp_path / "clip"), fps=24)

    frames, seconds = imageio_ffmpeg.count_frames_and_secs(path)
    assert path.endswith(".mp4") and frames == 12
    assert seconds == pytest.approx(0.5, abs=0.05)
    assert _streams(path) == {"Video"}


def test_audio_is_muxed_into_the_video(tmp_path):
    sample = _video_sample(generated_audio=torch.rand(2, 24000) * 2 - 1, audio_sample_rate=48000)

    path = rollout.write_sample_media(sample, str(tmp_path / "clip"), fps=24)

    assert _streams(path) == {"Video", "Audio"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4"]


def test_a_single_frame_is_saved_as_an_image(tmp_path):
    sample = Sample(generated_output=torch.rand(3, 1, 16, 16))

    assert rollout.write_sample_media(sample, str(tmp_path / "image"), fps=8).endswith(".png")


def test_saved_eval_media_keeps_every_sample_without_wandb(tmp_path):
    manager = SimpleNamespace(args=Namespace(diffusion_fps=24))
    samples = [_video_sample(), Sample(prompt="no media")]

    rollout.RolloutManager.__ray_metadata__.modified_class._log_images(
        manager,
        {"eval_media/demo_images": samples},
        max_images=0,
        step_key="eval/step",
        step_value=0,
        reward_key=None,
        save_dir=str(tmp_path),
    )

    assert sorted(p.name for p in tmp_path.iterdir()) == ["eval_media_demo_images_0.mp4", "index.jsonl"]
    rows = [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert rows == [{"file": "eval_media_demo_images_0.mp4", "prompt": "a boat", "seed": 7, "reward": None}]
