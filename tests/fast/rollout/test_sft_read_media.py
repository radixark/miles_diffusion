"""Image and video branches of the SFT media reader."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import os
import shutil
import subprocess

import numpy as np
import pytest
import torch
from PIL import Image

from miles.rollout.sft_rollout import read_media_clip


def _require_ffmpeg(*binaries: str) -> None:
    """Skip locally when ffmpeg is absent, but fail on CI.

    A silent skip would quietly delete this file's only real coverage of the
    decoder if a runner image ever stopped shipping ffmpeg.
    """
    missing = [name for name in binaries if shutil.which(name) is None]
    if not missing:
        return
    message = f"{', '.join(missing)} not installed"
    if os.environ.get("CI"):
        pytest.fail(f"{message}; CI must decode with the real binaries")
    pytest.skip(message)


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


def _patch_decoder(monkeypatch, num_frames=100):
    import miles.rollout.sft_rollout as sft_rollout

    video = torch.arange(num_frames, dtype=torch.uint8).reshape(num_frames, 1, 1, 1).expand(num_frames, 3, 8, 8)
    monkeypatch.setattr(sft_rollout, "_decode_video", lambda path: (video.contiguous(), 24.0))
    return video


def test_video_window_is_centered_and_strided(tmp_path, monkeypatch):
    _patch_decoder(monkeypatch)
    # span = (21-1)*2+1 = 41 frames, centered in 100 -> starts at 29
    clip = read_media_clip(str(tmp_path / "a.mp4"), height=8, width=8, num_frames=21, frame_stride=2)
    assert clip.shape == (3, 21, 8, 8)
    # frame values encode their source index, so the window is verifiable
    kept = [int(round(float(v) * 127.5 + 127.5)) for v in clip[0, :, 0, 0]]
    assert kept == list(range(29, 70, 2))


def test_video_rejects_too_short_a_clip(tmp_path, monkeypatch):
    _patch_decoder(monkeypatch, num_frames=10)
    with pytest.raises(ValueError, match="need 41"):
        read_media_clip(str(tmp_path / "a.mp4"), height=8, width=8, num_frames=21, frame_stride=2)


def test_decode_video_is_lossless_roundtrip(tmp_path):
    """Known pixels survive encode->_decode_video bit-exactly.

    Uses libx264rgb at crf 0 because the default H.264 + yuv420p path is doubly
    lossy (DCT quantization and chroma subsampling) and cannot prove anything
    about the decoder. Bit-exactness is what catches a swapped channel order or
    a reversed frame order, which a shape-and-range assertion would let through.
    """
    _require_ffmpeg("ffmpeg", "ffprobe")
    from miles.rollout.sft_rollout import _decode_video

    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (8, 48, 64, 3), dtype=np.uint8)
    path = tmp_path / "lossless.mp4"
    encode = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "64x48",
            "-r",
            "24",
            "-i",
            "-",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-pix_fmt",
            "rgb24",
            str(path),
        ],
        input=frames.tobytes(),
        capture_output=True,
    )
    if encode.returncode != 0:
        pytest.skip(f"no lossless rgb encoder: {encode.stderr.decode()[:120]}")

    decoded, fps = _decode_video(str(path))
    assert decoded.shape == (8, 3, 48, 64)
    assert fps == pytest.approx(24.0)
    assert torch.equal(decoded.permute(0, 2, 3, 1), torch.from_numpy(frames))


def test_read_media_clip_on_a_real_file(tmp_path):
    """The full reader (probe + decode + window + resize) on a real mp4."""
    _require_ffmpeg("ffmpeg", "ffprobe")
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=24:duration=1.25",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    clip = read_media_clip(str(path), height=32, width=32, num_frames=9, frame_stride=2)
    assert clip.shape == (3, 9, 32, 32)
    assert clip.min() >= -1.0 and clip.max() <= 1.0
    # testsrc animates, so the kept frames must differ from one another
    assert not torch.equal(clip[:, 0], clip[:, -1])


def test_ffmpeg_failure_is_reported(tmp_path):
    _require_ffmpeg("ffprobe")
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="ffprobe could not read"):
        read_media_clip(str(broken), height=32, width=32, num_frames=1, frame_stride=1)
