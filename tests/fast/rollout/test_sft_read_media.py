"""read_media_clip: target file -> media clip, built on the MediaSource load layer.

    visual_source(path) -> ImageSource | VideoSource      load (miles.utils.media), chosen by extension
      .info         ffprobe frame list     num_frames/fps/frame times, upright size, sample aspect ratio
      .decode       ffmpeg trim            frames [start, start + length) only, upright stored pixels
           |
    process: exact frame count, canvas size and square pixels, else an error; every frame_stride-th frame
           |
    uint8 {"video": [C,T,H,W], "fps", "frame_times_seconds"}

What each test pins:
  image branch      single frame, byte-exact passthrough, EXIF orientation applied, wrong size rejected
  frame selection   which source frames survive num_frames x frame_stride, and any other length is
                    rejected (mocked load)
  decoder           bit-exact rgb24 round trip through real ffmpeg, frame-exact windows, probed fps
  display geometry  a 90-degree rotated video decodes bit-exactly as the rotated frames, its size and pixel
                    aspect ratio turned with it (stored 2:1 -> upright 1:2); 2:1 pixels are
                    rejected, a scale-then-crop residue such as 4096:4095 is not
  edit list         preroll packets an edit list discards are not counted as frames
  variable rate     a variable-frame-rate window returns each selected frame exactly once, timed by its own
                    presentation timestamp
  missing pts       packets stored without timestamps (AVI with B-frames) still get frame times
  raw stream        a raw elementary stream, which has no container timeline, is rejected
  stream offset     frame times sit on the file timeline, so a video starting 1 s after its audio starts at 1 s
  subtitle track    a subtitle track adds no frames
  resolution change a mid-stream resolution change is rejected, since no single size describes the frames
  cover art         an audio file's attached picture is not a video stream, so the file is rejected
  probe failure     unreadable media names ffprobe

Real media (public Hugging Face fixtures at pinned revisions, and ffmpeg FATE samples pinned by sha256):
  cat-rotated.jpg    EXIF orientation 8 photo   stored 500x333 --rotate 90 CCW--> upright 333x500 frame
  sample_demo_1.mp4  H.264 + AAC, 640x360       243 frames at 25 fps; a mid-clip window equals an independent
                                                ffmpeg select-filter decode of the same frames
  displaymatrix.mov  H.264, -90 degree rotation stored 160x240 of 1:2 pixels --turn clockwise--> upright 240x160
                     (FATE, cut to 33 frames)   of 2:1 pixels; a window equals the stored frames read with
                                                -noautorotate and turned by hand
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import hashlib
import os
import shutil
import subprocess
from fractions import Fraction

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


def test_image_reads_as_its_single_frame(tmp_path):
    path = tmp_path / "a.png"
    _write_png(path, 48, 64)
    media_clip = read_media_clip(str(path), height=48, width=64, num_frames=1, frame_stride=1)
    original = torch.from_numpy(np.asarray(Image.open(path))).permute(2, 0, 1)
    assert media_clip["video"].dtype == torch.uint8
    assert torch.equal(media_clip["video"][:, 0], original)


def test_image_of_another_size_is_rejected(tmp_path):
    path = tmp_path / "a.png"
    _write_png(path, 120, 200)
    with pytest.raises(ValueError, match="is 200x120, not the 64x64 canvas"):
        read_media_clip(str(path), height=64, width=64, num_frames=1, frame_stride=1)


def test_image_exif_orientation_is_applied(tmp_path):
    # EXIF orientation 6 stores a portrait photo as landscape pixels, to be turned 90 degrees clockwise.
    from miles.utils.media import ImageSource

    pixels = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif[0x0112] = 6
    path = tmp_path / "portrait.jpg"
    Image.fromarray(pixels).save(path, exif=exif, quality=100)
    source = ImageSource(str(path))
    assert (source.info.width, source.info.height) == (48, 64)
    assert source.decode(0, 1).shape == (1, 3, 64, 48)


def _patch_source(monkeypatch, num_frames=41):
    """A fake 8x8 source whose pixel value is the frame index."""
    from miles.utils.media import MediaInfo, VideoSource

    video = torch.arange(num_frames, dtype=torch.uint8).reshape(num_frames, 1, 1, 1).expand(num_frames, 3, 8, 8)
    frame_times = tuple(index / 24 for index in range(num_frames))
    info = MediaInfo(num_frames=num_frames, fps=24.0, frame_times_seconds=frame_times, width=8, height=8)
    monkeypatch.setattr(VideoSource, "info", property(lambda self: info))
    monkeypatch.setattr(VideoSource, "decode", lambda self, start, length: video[start : start + length])


def test_video_keeps_every_stride_th_frame(tmp_path, monkeypatch):
    _patch_source(monkeypatch)
    # span = (21-1)*2+1 = 41 frames: the whole file, every other frame
    media_clip = read_media_clip(str(tmp_path / "a.mp4"), height=8, width=8, num_frames=21, frame_stride=2)
    assert media_clip["video"].shape == (3, 21, 8, 8)
    assert media_clip["fps"] == 24.0
    # frame values encode their source index, so the selection is verifiable
    kept = [int(v) for v in media_clip["video"][0, :, 0, 0]]
    assert kept == list(range(0, 41, 2))
    assert media_clip["frame_times_seconds"] == pytest.approx([index / 24 for index in kept])


@pytest.mark.parametrize("num_frames", [40, 42])
def test_video_of_another_length_is_rejected(tmp_path, monkeypatch, num_frames):
    _patch_source(monkeypatch, num_frames=num_frames)
    with pytest.raises(ValueError, match="needs exactly 41"):
        read_media_clip(str(tmp_path / "a.mp4"), height=8, width=8, num_frames=21, frame_stride=2)


def _encode_lossless_rgb(path, frames, *output_args):
    """Encode uint8 ``[T, H, W, 3]`` frames with libx264rgb at crf 0.

    The default H.264 + yuv420p path is doubly lossy (DCT quantization and chroma
    subsampling) and cannot prove anything about the decoder.
    """
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
            f"{frames.shape[2]}x{frames.shape[1]}",
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
            *output_args,
            str(path),
        ],
        input=frames.tobytes(),
        capture_output=True,
    )
    if encode.returncode != 0:
        pytest.skip(f"no lossless rgb encoder: {encode.stderr.decode()[:120]}")


def test_decode_is_lossless_and_frame_exact(tmp_path):
    """Known pixels survive encode->decode bit-exactly, for the whole clip and a window.

    Bit-exactness is what catches a swapped channel order or a reversed frame order,
    which a shape-and-range assertion would let through.
    """
    _require_ffmpeg("ffmpeg", "ffprobe")
    from miles.utils.media import VideoSource

    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (8, 48, 64, 3), dtype=np.uint8)
    path = tmp_path / "lossless.mp4"
    _encode_lossless_rgb(path, frames)

    source = VideoSource(str(path))
    assert source.info.num_frames == 8
    assert source.info.fps == pytest.approx(24.0)
    assert (source.info.width, source.info.height) == (64, 48)
    decoded = source.decode(0, 8)
    assert torch.equal(decoded.permute(0, 2, 3, 1), torch.from_numpy(frames))
    # source frames 3..5 only: a seek that lands on a neighbouring frame changes the pixels
    decoded = source.decode(3, 3)
    assert torch.equal(decoded.permute(0, 2, 3, 1), torch.from_numpy(frames[3:6]))


def test_rotated_video_decodes_upright(tmp_path):
    # A 64x48 stream of 2:1 pixels tagged with a 90-degree display rotation plays as 48x64, turned
    # counterclockwise, and each turned pixel is twice as tall as wide:
    #   decode == np.rot90(frames), and info reports the upright 48x64 with 1:2 pixels
    _require_ffmpeg("ffmpeg", "ffprobe")
    from miles.utils.media import VideoSource

    frames = np.random.default_rng(0).integers(0, 256, (2, 48, 64, 3), dtype=np.uint8)
    stored_path, rotated_path = tmp_path / "stored.mp4", tmp_path / "rotated.mp4"
    _encode_lossless_rgb(stored_path, frames, "-vf", "setsar=2")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(stored_path), "-c", "copy"]
        + [str(rotated_path)],
        check=True,
    )
    source = VideoSource(str(rotated_path))
    assert (source.info.width, source.info.height, source.info.sample_aspect_ratio) == (48, 64, Fraction(1, 2))
    decoded = source.decode(0, 2)
    assert torch.equal(decoded.permute(0, 2, 3, 1), torch.from_numpy(np.rot90(frames, axes=(1, 2)).copy()))


@pytest.mark.parametrize(("sar", "square"), [("2", False), ("4096/4095", True)])
def test_only_non_square_pixels_are_rejected(tmp_path, sar, square):
    # 2:1 pixels display a 64-wide frame 128 wide; 4096:4095, what scale-then-crop leaves behind, displays it
    # 64.02 wide.
    _require_ffmpeg("ffmpeg", "ffprobe")
    path = tmp_path / "anamorphic.mp4"
    _encode_lossless_rgb(path, np.zeros((1, 48, 64, 3), dtype=np.uint8), "-vf", f"setsar={sar}")
    if square:
        assert read_media_clip(str(path), height=48, width=64, num_frames=1, frame_stride=1)["video"].shape[1] == 1
    else:
        with pytest.raises(ValueError, match="non-square pixels"):
            read_media_clip(str(path), height=48, width=64, num_frames=1, frame_stride=1)


def _encode_testsrc(path, *output_args):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=24:duration=2",
            *output_args,
            path,
        ],
        check=True,
    )


def test_edit_list_preroll_is_not_counted(tmp_path):
    # A stream copy cut mid-GOP keeps the preroll packets but flags them discard:
    #   source          48 frames, one keyframe
    #   -ss 1 -c copy   packets [24 preroll, flagged D | 24 presented]   -> num_frames 24
    _require_ffmpeg("ffmpeg", "ffprobe")
    from miles.utils.media import VideoSource

    source_path, cut_path = tmp_path / "source.mp4", tmp_path / "cut.mp4"
    _encode_testsrc(source_path, "-g", "1000")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", str(source_path), "-c", "copy", str(cut_path)], check=True
    )
    cut, source = VideoSource(str(cut_path)), VideoSource(str(source_path))
    assert cut.info.num_frames == 24
    # The cut's first presented frame is source frame 24, decoded from the same packets.
    assert torch.equal(cut.decode(0, 4), source.decode(24, 4))


def test_variable_frame_rate_window_returns_each_frame_once(tmp_path):
    # Dropping every third frame leaves 32 frames on an uneven timeline:
    #   kept source frames 0 1 3 4 6 7 ...   so kept frame 2 is source frame 3, shown at 3 / 24 s
    #   decode(2, 5) -> exactly frames 2..6; ffmpeg's default output sync would duplicate some
    _require_ffmpeg("ffmpeg", "ffprobe")
    from miles.utils.media import VideoSource

    path = tmp_path / "vfr.mp4"
    _encode_testsrc(path, "-vf", "select=not(eq(mod(n\\,3)\\,2))", "-fps_mode", "vfr")
    source = VideoSource(str(path))
    assert source.info.num_frames == 32
    assert source.info.frame_times_seconds[2] == pytest.approx(3 / 24)
    assert torch.equal(source.decode(2, 5), source.decode(0, 32)[2:7])


def test_packets_without_timestamps_still_get_frame_times(tmp_path):
    # MPEG-4 AVI with B-frames stores no pts on reordered packets; the probe generates them from dts.
    # The B-frame reorder delay puts the first frame at 1/24 s, where the decoder also places it.
    _require_ffmpeg("ffmpeg", "ffprobe")
    from miles.utils.media import VideoSource

    path = tmp_path / "clip.avi"
    _encode_testsrc(path, "-c:v", "mpeg4", "-bf", "2")
    frame_times = VideoSource(str(path)).info.frame_times_seconds
    assert list(frame_times) == pytest.approx([(index + 1) / 24 for index in range(48)], abs=1e-3)


def test_raw_stream_is_rejected(tmp_path):
    # Raw H.264 carries no timestamps, and ffprobe reports this 24 fps stream as 25 fps.
    _require_ffmpeg("ffmpeg", "ffprobe")
    path = tmp_path / "clip.h264"
    _encode_testsrc(path, "-f", "h264")
    with pytest.raises(ValueError, match="no container timeline"):
        read_media_clip(str(path), height=48, width=64, num_frames=1, frame_stride=1)


def test_frame_times_keep_the_video_stream_offset(tmp_path):
    # Audio starts at 0 s, video at 1 s: an audio crop that seeks the file timeline must start at 1 s.
    _require_ffmpeg("ffmpeg", "ffprobe")
    from miles.utils.media import VideoSource

    source_path, offset_path = tmp_path / "source.mkv", tmp_path / "offset.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=2"]
        + ["-f", "lavfi", "-i", "sine=duration=3", "-c:v", "mpeg4", "-c:a", "pcm_s16le", str(source_path)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-itsoffset", "1", "-i", str(source_path), "-i", str(source_path)]
        + ["-map", "0:v", "-map", "1:a", "-c", "copy", str(offset_path)],
        check=True,
    )
    assert VideoSource(str(offset_path)).info.frame_times_seconds[0] == pytest.approx(1.0, abs=1e-3)


def test_subtitle_track_adds_no_frames(tmp_path):
    _require_ffmpeg("ffmpeg", "ffprobe")
    from miles.utils.media import VideoSource

    subtitles_path, path = tmp_path / "subtitles.srt", tmp_path / "clip.mkv"
    subtitles_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")
    _encode_testsrc(path, "-i", str(subtitles_path), "-c:v", "mpeg4", "-c:s", "srt")
    assert VideoSource(str(path)).info.num_frames == 48


def test_resolution_change_is_rejected(tmp_path):
    # 2 s at 64x48, then 1 s at 32x24: no single size describes the decoded frames.
    _require_ffmpeg("ffmpeg", "ffprobe")
    first_path, second_path, path = tmp_path / "first.mkv", tmp_path / "second.mkv", tmp_path / "switch.mkv"
    _encode_testsrc(first_path, "-c:v", "mpeg2video")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=32x24:rate=24:duration=1"]
        + ["-c:v", "mpeg2video", str(second_path)],
        check=True,
    )
    playlist = tmp_path / "playlist.txt"
    playlist.write_text(f"file '{first_path}'\nfile '{second_path}'\n")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(playlist), "-c", "copy", str(path)],
        check=True,
    )
    with pytest.raises(ValueError, match="changes resolution mid-stream"):
        read_media_clip(str(path), height=48, width=64, num_frames=71, frame_stride=1)


def test_cover_art_is_not_a_video_stream(tmp_path):
    _require_ffmpeg("ffmpeg", "ffprobe")
    cover_path, path = tmp_path / "cover.png", tmp_path / "song.flac"
    _write_png(cover_path, 48, 64)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=1:sample_rate=16000",
            "-i",
            str(cover_path),
        ]
        + ["-map", "0:a", "-map", "1:v", "-c:v", "png", "-disposition:v", "attached_pic", str(path)],
        check=True,
    )
    with pytest.raises(ValueError, match="has no video stream"):
        read_media_clip(str(path), height=48, width=64, num_frames=1, frame_stride=1)


def test_ffmpeg_failure_is_reported(tmp_path):
    _require_ffmpeg("ffprobe")
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="ffprobe could not read"):
        read_media_clip(str(broken), height=32, width=32, num_frames=1, frame_stride=1)


# ---------------------------------------------------------------------------
# Real media: public Hugging Face fixtures, pinned to a revision
# ---------------------------------------------------------------------------


def _hub_media(repo_id: str, revision: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision)


def _real_photo() -> str:
    return _hub_media(
        "hf-internal-testing/fixtures_image_utils", "4539dd3e9faa39d54e606b4c875f85c7561f5423", "cat-rotated.jpg"
    )


def _real_video() -> str:
    return _hub_media(
        "raushan-testing-hf/videos-test", "4cba700bd771f44d72b549253da025c32e944d42", "sample_demo_1.mp4"
    )


def _real_rotated_video(tmp_path) -> str:
    from urllib.request import urlopen

    path = tmp_path / "displaymatrix.mov"
    path.write_bytes(urlopen("https://fate-suite.ffmpeg.org/mov/displaymatrix.mov").read())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "02ad3fcdc28a1c8d9d81384829f60c5c28cd23a5a8570cbb56d0f90647ba92b4"
    )
    return str(path)


def test_image_source_turns_a_real_exif_rotated_photo_upright():
    from miles.utils.media import ImageSource

    # The camera stored the cat sideways, 500 wide x 333 tall, and EXIF orientation 8 says "rotate 90 degrees
    # counter-clockwise to display", so the upright frame is the stored pixels rotated: 333 wide x 500 tall.
    path = _real_photo()
    source = ImageSource(path)
    assert (source.info.num_frames, source.info.width, source.info.height) == (1, 333, 500)
    stored = np.asarray(Image.open(path).convert("RGB"))
    upright = torch.from_numpy(np.rot90(stored).copy()).permute(2, 0, 1)
    assert torch.equal(source.decode(0, 1), upright[None])


def test_video_source_probes_and_windows_a_real_h264_clip():
    _require_ffmpeg("ffmpeg", "ffprobe")
    from miles.utils.media import VideoSource

    path = _real_video()
    source = VideoSource(path)
    info = source.info
    assert (info.num_frames, info.fps, info.width, info.height) == (243, 25.0, 640, 360)
    assert info.frame_times_seconds[100:103] == pytest.approx((4.0, 4.04, 4.08))
    # frames 100..102, decoded by select instead of trim:
    #   decode(100, 3)  ==  ffmpeg -vf "select=between(n,100,102)"
    selected = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-an", "-vf", "select=between(n\\,100\\,102)"]
        + ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    ).stdout
    reference = torch.from_numpy(np.frombuffer(selected, dtype=np.uint8).reshape(3, 360, 640, 3).copy())
    assert torch.equal(source.decode(100, 3), reference.permute(0, 3, 1, 2))


def test_video_source_turns_a_real_rotated_clip_upright(tmp_path):
    _require_ffmpeg("ffmpeg", "ffprobe")
    from miles.utils.media import VideoSource

    # The camera stored 160x240 frames of 1:2 pixels and tagged them -90 degrees, a quarter turn clockwise, so
    # the frames decode turned, 240x160, and each turned pixel is twice as wide as tall. FATE cut the file short,
    # so only its first 33 frames decode:
    #   decode(10, 3)  ==  np.rot90(stored frames 10..12, clockwise), the stored frames read with -noautorotate
    path = _real_rotated_video(tmp_path)
    source = VideoSource(path)
    info = source.info
    assert (info.num_frames, info.width, info.height, info.sample_aspect_ratio) == (33, 240, 160, Fraction(2))
    selected = subprocess.run(
        ["ffmpeg", "-v", "error", "-noautorotate", "-i", path, "-an", "-vf", "select=between(n\\,10\\,12)"]
        + ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    ).stdout
    stored = np.frombuffer(selected, dtype=np.uint8).reshape(3, 240, 160, 3)
    upright = torch.from_numpy(np.rot90(stored, k=-1, axes=(1, 2)).copy()).permute(0, 3, 1, 2)
    assert torch.equal(source.decode(10, 3), upright)
