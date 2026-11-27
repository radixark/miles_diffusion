"""Compare bf16 VideoAlign inputs and scores against a commit-pinned upstream SDPA reference.

The reference runs in a subprocess with its original Transformers/PEFT versions.
Individual and padded mixed-resolution batches must match exactly.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=600,
    suite="stage-b-3-gpu-h200",
    labels=[],
)

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from miles.rollout.rm_hub.videoalign import VideoAlignScorer

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_OFFICIAL_SCRIPT = _FIXTURES / "videoalign_official_scores.py"
_spec = importlib.util.spec_from_file_location("videoalign_official_scores", _OFFICIAL_SCRIPT)
_official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_official)

# the official pins live in a per-machine overlay (transformers/peft/trl for the 4.45-era checkpoint)
OFFICIAL_PINS = [
    "transformers==4.45.2",
    "tokenizers==0.20.3",
    "huggingface_hub==0.25.2",
    "trl==0.12.2",
    "peft==0.10.0",
]
_OVERLAY_KEY = hashlib.sha256(
    " ".join([*OFFICIAL_PINS, sys.implementation.cache_tag, platform.machine()]).encode()
).hexdigest()[:12]
_OVERLAY = Path.home() / ".cache" / "miles" / f"videoalign_official_{_OVERLAY_KEY}"


def _paths():
    # machine-specific locations; skip unless the checkpoint/repo/base model are present
    checkpoint = os.environ.get("VIDEOALIGN_CHECKPOINT_DIR")
    model = os.environ.get("VIDEOALIGN_MODEL_PATH")
    repo = os.environ.get("VIDEOALIGN_REPO_DIR")
    if not (checkpoint and model and repo):
        pytest.skip("set VIDEOALIGN_CHECKPOINT_DIR / VIDEOALIGN_MODEL_PATH / VIDEOALIGN_REPO_DIR to run the alignment")
    return checkpoint, model, repo


def _install_overlay() -> Path:
    if not _OVERLAY.is_dir():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-deps",
                "--target",
                str(_OVERLAY),
                *OFFICIAL_PINS,
            ],
            check=True,
        )
    return _OVERLAY


@pytest.fixture(scope="module")
def official_scores(tmp_path_factory):
    checkpoint, model, repo = _paths()
    overlay = _install_overlay()
    out = tmp_path_factory.mktemp("videoalign_official") / "scores.json"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(overlay), os.environ.get("PYTHONPATH")]))}
    subprocess.run(
        [
            sys.executable,
            str(_OFFICIAL_SCRIPT),
            "--repo-dir",
            repo,
            "--checkpoint-dir",
            checkpoint,
            "--model-path",
            model,
            "--output",
            str(out),
        ],
        env=env,
        check=True,
    )
    return json.loads(out.read_text())


def _frames(sample):
    return _official.make_video(sample["width"], sample["height"], sample["frames"], sample["offset"])


@pytest.fixture(scope="module")
def scorer(official_scores):
    checkpoint, model, _ = _paths()
    return VideoAlignScorer(device="cuda", model_path=model, checkpoint_path=checkpoint, normalize=False)


def test_inputs_match_official(official_scores, scorer):
    for sample in official_scores["samples"]:
        inputs = scorer.build_inputs([sample["prompt"]], [_frames(sample)])
        assert inputs["input_ids"][0].tolist() == sample["input_ids"]
        assert inputs["video_grid_thw"][0].tolist() == sample["video_grid_thw"]
        assert list(inputs["pixel_values_videos"].shape) == sample["pixel_values_videos_shape"]
        assert _official._sha256(inputs["pixel_values_videos"]) == sample["pixel_values_videos_sha256"]


def test_scores_match_official(official_scores, scorer):
    for sample in official_scores["samples"]:
        (result,) = scorer([sample["prompt"]], [_frames(sample)])
        assert [result[key] for key in ("vq", "mq", "ta")] == [sample[key] for key in ("VQ", "MQ", "TA")]


def test_padded_batch_scores_match_official(official_scores, scorer):
    samples = official_scores["samples"]
    results = scorer([s["prompt"] for s in samples], [_frames(s) for s in samples])
    assert [[r[k] for k in ("vq", "mq", "ta")] for r in results] == official_scores["batch_scores"]


def test_normalization_and_overall(official_scores, scorer):
    from miles.rollout.rm_hub.videoalign import _NORM

    samples = official_scores["samples"]
    scorer.normalize = True
    try:
        results = scorer([s["prompt"] for s in samples], [_frames(s) for s in samples])
    finally:
        scorer.normalize = False
    for result, raw in zip(results, official_scores["batch_scores"], strict=True):
        expected = {
            key: (score - _NORM[key][0]) / _NORM[key][1] for key, score in zip(("vq", "mq", "ta"), raw, strict=True)
        }
        assert result == {**expected, "overall": sum(expected.values())}
