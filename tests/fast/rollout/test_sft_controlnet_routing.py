"""Adding pose conditioning must preserve existing RGB-only family semantics."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest

from miles.rollout.sft_rollout import build_sft_encode_item


def args(tmp_path, family):
    return SimpleNamespace(
        prompt_data=str(tmp_path / "train.jsonl"),
        diffusion_model_family=family,
        sft_encoder_checkpoint="base",
        diffusion_height=480,
        diffusion_width=832,
        diffusion_output_num_frames=49,
        sft_frame_stride=1,
    )


@pytest.mark.parametrize("family", ["wan2_2", "h3"])
def test_rgb_family_ignores_preexisting_control_metadata(tmp_path, family):
    (tmp_path / "rgb.mp4").write_bytes(b"rgb")
    plain = SimpleNamespace(index=0, prompt="dancing", metadata={"video": "rgb.mp4"})
    with_unused_control = SimpleNamespace(
        index=0,
        prompt="dancing",
        metadata={"video": "rgb.mp4", "control_video": "does-not-exist.mp4"},
    )
    original = build_sft_encode_item(args(tmp_path, family), plain)
    actual = build_sft_encode_item(args(tmp_path, family), with_unused_control)
    assert actual == original  # Includes the cache filename and latent sampling seed.
    assert "control_media" not in actual


def test_control_family_requires_and_tracks_pose(tmp_path):
    (tmp_path / "rgb.mp4").write_bytes(b"rgb")
    sample = SimpleNamespace(index=0, prompt="dancing", metadata={"video": "rgb.mp4"})
    config = args(tmp_path, "wan_controlnet")
    with pytest.raises(ValueError, match="requires metadata.control_video"):
        build_sft_encode_item(config, sample)
    sample.metadata["control_video"] = "pose.mp4"
    pose = tmp_path / "pose.mp4"
    pose.write_bytes(b"first pose")
    before = build_sft_encode_item(config, sample)
    assert before["control_media"] == str(pose)
    pose.write_bytes(b"changed pose")
    after = build_sft_encode_item(config, sample)
    assert before["cache_name"] != after["cache_name"]
