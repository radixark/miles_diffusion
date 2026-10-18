"""Media load: locate media files, then probe and decode them lazily."""

import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import torch

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# Locate: dataset uri -> local file
# ---------------------------------------------------------------------------


def resolve_media_uri(uri: str, prompt_data: str) -> str:
    """Anchor relative paths at the dataset jsonl's directory; URLs pass through."""
    scheme = urlparse(uri).scheme
    if scheme in {"http", "https"} or os.path.isabs(uri):
        return uri
    if scheme:
        raise ValueError(f"Unsupported media URI scheme: {scheme!r}")
    return str(Path(prompt_data).absolute().parent / uri)


def maybe_download_media(uri: str, cache_dir: str | Path) -> str:
    """Download an http(s) URL once into cache_dir and return the local copy; local paths pass through."""
    parsed = urlparse(uri)
    if parsed.scheme not in {"http", "https"}:
        return uri
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(uri.encode()).hexdigest()
    existing = [path for path in directory.glob(f"{key}.*") if path.suffix != ".tmp"]
    if existing:
        return str(existing[0])
    # Some hosts, Wikimedia among them, reject urllib's default User-Agent.
    request = Request(uri, headers={"User-Agent": "miles-diffusion/1.0"})
    temporary = None
    try:
        with urlopen(request, timeout=60) as response:
            # Only a media extension names the type; download.php names a script, not its PNG response.
            url_media_type = mimetypes.guess_type(unquote(parsed.path))[0] or ""
            if url_media_type.split("/")[0] in {"image", "video", "audio"}:
                suffix = Path(unquote(parsed.path)).suffix.lower()
            else:
                suffix = mimetypes.guess_extension(response.headers.get_content_type()) or ".media"
            destination = directory / f"{key}{suffix}"
            with tempfile.NamedTemporaryFile(dir=directory, prefix=f"{key}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                shutil.copyfileobj(response, stream)
            size = temporary.stat().st_size
            declared_size = response.headers.get("Content-Length")
            if declared_size is not None and size != int(declared_size):
                raise ValueError(f"The media URL returned {size} of its {declared_size} declared bytes")
            os.replace(temporary, destination)
            return str(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def media_fingerprint(path: str) -> dict:
    """Identify a local media revision for an encoded-sample cache key."""
    stat = Path(path).stat()
    return {"path": path, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


# ---------------------------------------------------------------------------
# Load: local file -> probed facts -> decoded window
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MediaInfo:
    """Probed facts of the file, display rotation applied; a fact the kind lacks stays ``None``."""

    num_frames: int | None = None
    fps: float | None = None
    # On the file timeline, the one ffmpeg's input -ss seeks.
    frame_times_seconds: tuple[float, ...] | None = None
    width: int | None = None
    height: int | None = None
    # Unknown counts as square.
    sample_aspect_ratio: Fraction = Fraction(1)
    # Of the first audio stream, the one ffmpeg's -map 0:a:0 decodes.
    audio_sample_rate: int | None = None


class MediaSource(ABC):
    """One kind of media in a local file, probed on first use and decoded one window at a time.

    Loading only: checking or transforming what is decoded is the caller's job.
    """

    def __init__(self, path: str):
        self.path = path

    @property
    @abstractmethod
    def info(self) -> MediaInfo:
        """Probed facts of the file; subclasses cache them with ``cached_property``."""

    @abstractmethod
    def decode(self, start: int, length: int) -> torch.Tensor:
        """The window ``[start, start + length)`` in the media's own unit: frames for images and video."""

    def _ffprobe(self, *arguments: str) -> dict:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", *arguments, "-of", "json", self.path], capture_output=True, text=True
        )
        if probe.returncode != 0:
            raise ValueError(f"ffprobe could not read {self.path}: {probe.stderr.strip()[:200]}")
        return json.loads(probe.stdout)

    def _ffmpeg(self, *arguments: str) -> bytes:
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", self.path, *arguments, "-"], capture_output=True)
        if decoded.returncode != 0:
            raise ValueError(f"ffmpeg failed on {self.path}: {decoded.stderr.decode()[:200]}")
        return decoded.stdout


class ImageSource(MediaSource):
    """A still image: one upright frame."""

    @cached_property
    def info(self) -> MediaInfo:
        from PIL import Image, ImageOps

        with Image.open(self.path) as image:
            width, height = ImageOps.exif_transpose(image).size
        return MediaInfo(num_frames=1, width=width, height=height)

    def decode(self, start: int, length: int) -> torch.Tensor:
        """Frames ``[start, start + length)`` of the one-frame clip as uint8 ``[T, C, H, W]`` RGB."""
        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(self.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return torch.from_numpy(np.array(image)).permute(2, 0, 1)[None][start : start + length]


class VideoSource(MediaSource):
    """The first video stream of a container; ``info`` probes every frame and the soundtrack once."""

    @cached_property
    def info(self) -> MediaInfo:
        probed = self._probe_streams_and_frames()
        video = self._main_video_stream(probed)
        file_start = self._file_start_seconds(probed)
        frames = self._frames_of(video, probed)
        width, height = self._stored_size(frames)
        sample_aspect_ratio = self._stored_sample_aspect_ratio(video)
        # ffmpeg rotates while decoding; a quarter turn swaps the axes and inverts the pixel aspect ratio.
        if self._display_rotation(video) % 180 == 90:
            width, height, sample_aspect_ratio = height, width, 1 / sample_aspect_ratio
        return MediaInfo(
            num_frames=len(frames),
            fps=float(Fraction(video["avg_frame_rate"])),
            frame_times_seconds=tuple(float(frame["best_effort_timestamp_time"]) - file_start for frame in frames),
            width=width,
            height=height,
            sample_aspect_ratio=sample_aspect_ratio,
            audio_sample_rate=self._soundtrack_sample_rate(probed),
        )

    def decode(self, start: int, length: int) -> torch.Tensor:
        """Frames ``[start, start + length)`` as upright uint8 ``[T, C, H, W]`` RGB.

        ``trim`` counts decoded frames, so the window is frame-exact; ``-fps_mode passthrough`` stops a
        variable-rate source from duplicating frames. ffmpeg because torchvision 0.26 has no video API and
        torchcodec no Linux ARM build.
        """
        import numpy as np

        width, height = self.info.width, self.info.height
        window = f"trim=start_frame={start}:end_frame={start + length},setpts=PTS-STARTPTS"
        raw = self._ffmpeg(
            "-map", "0:V:0", "-an", "-vf", window, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24"
        )
        if len(raw) != length * height * width * 3:
            raise ValueError(
                f"{self.path}: ffmpeg returned {len(raw)} bytes, not {length} {width}x{height} rgb24 frames"
            )
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(length, height, width, 3)
        return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)

    def _probe_streams_and_frames(self) -> dict:
        return self._ffprobe(
            # Containers such as AVI store no pts on reordered packets; derive them from dts.
            "-fflags",
            "+genpts",
            "-show_entries",
            "format=start_time"
            ":stream=index,codec_type,avg_frame_rate,sample_aspect_ratio,sample_rate"
            ":stream_disposition=attached_pic"
            ":stream_side_data=rotation"
            ":frame=stream_index,best_effort_timestamp_time,width,height",
        )

    def _main_video_stream(self, probed: dict) -> dict:
        """The first video stream that is not cover art (an attached picture)."""
        for stream in probed["streams"]:
            if stream["codec_type"] == "video" and not stream["disposition"]["attached_pic"]:
                return stream
        raise ValueError(f"{self.path} has no video stream")

    @staticmethod
    def _soundtrack_sample_rate(probed: dict) -> int | None:
        return next(
            (int(stream["sample_rate"]) for stream in probed["streams"] if stream["codec_type"] == "audio"), None
        )

    def _file_start_seconds(self, probed: dict) -> float:
        # Raw elementary streams (.h264, .m1v), whose rate ffprobe misreports, and mislabeled images have none.
        if "start_time" not in probed["format"]:
            raise ValueError(
                f"{self.path} has no container timeline; remux a raw video stream into a container such as mp4, "
                "or give an image an image extension"
            )
        return float(probed["format"]["start_time"])

    @staticmethod
    def _frames_of(video: dict, probed: dict) -> list[dict]:
        """Decoded frames in presentation order, the sequence ``trim`` counts in ``decode``."""
        # Subtitle frames carry no stream_index.
        return [frame for frame in probed["frames"] if frame.get("stream_index") == video["index"]]

    def _stored_size(self, frames: list[dict]) -> tuple[int, int]:
        """(width, height) before the display rotation."""
        sizes = {(frame["width"], frame["height"]) for frame in frames}
        if len(sizes) > 1:
            raise ValueError(f"{self.path} changes resolution mid-stream; re-encode it at one size")
        return sizes.pop()

    @staticmethod
    def _display_rotation(video: dict) -> int:
        return next((side["rotation"] for side in video.get("side_data_list", []) if "rotation" in side), 0)

    @staticmethod
    def _stored_sample_aspect_ratio(video: dict) -> Fraction:
        # "0:1" and "N/A" mean unknown.
        numerator, _, denominator = video.get("sample_aspect_ratio", "1:1").partition(":")
        if numerator.isdigit() and denominator.isdigit() and int(numerator) > 0:
            return Fraction(int(numerator), int(denominator))
        return Fraction(1)


def visual_source(path: str) -> ImageSource | VideoSource:
    """The image or video source for a visual file, chosen by its extension."""
    return ImageSource(path) if Path(path).suffix.lower() in IMAGE_EXTENSIONS else VideoSource(path)
