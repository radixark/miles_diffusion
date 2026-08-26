"""Relative media paths anchor at the dataset jsonl directory; absolute paths pass through."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from miles.rollout.sft_rollout import resolve_media_path


def test_absolute_path_passes_through():
    assert resolve_media_path("/data/clips/a.mp4", "/elsewhere/train.jsonl") == "/data/clips/a.mp4"


def test_relative_path_anchors_at_jsonl_dir():
    assert resolve_media_path("clips/a.mp4", "/data/set/train.jsonl") == "/data/set/clips/a.mp4"
