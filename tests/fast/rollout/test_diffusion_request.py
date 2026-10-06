from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

# One SFT request flows from a jsonl row to the family encoder:
#
#   jsonl row {"text", "conditions", "target"}
#        │
#     Dataset
#        │  every relative uri anchored at the jsonl, condition order kept
#        ▼
#   Sample(prompt: str, conditions: [...], target: {...})
#        │
#   localize_sft_media(sample)
#        │  every url in target / conditions ─► local copy; local paths unchanged
#        ▼
#   sft_sample_key(sample): order + every file revision
#   family encode_sample(sample, target clip)
#     Wan: conditions / target audio rejected

import dataclasses
import json
from argparse import Namespace

import pytest

from miles.rollout.encoder_hub import wan2_2
from miles.rollout.sft_rollout import localize_sft_media, sft_sample_key
from miles.utils.diffusion_data import Dataset
from miles.utils.types import Sample


def _args(tmp_path, **kwargs):
    base = dict(
        prompt_data=str(tmp_path / "dataset.jsonl"),
        diffusion_model_family="h3",
        sft_encoder_checkpoint="test-checkpoint",
        diffusion_height=32,
        diffusion_width=32,
        diffusion_output_num_frames=5,
        sft_frame_stride=1,
        diffusion_task="t2va",
    )
    return Namespace(**(base | kwargs))


def _write_dataset(tmp_path, *rows) -> str:
    path = tmp_path / "dataset.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


# ---------------------------------------------------------------------------
# Dataset: jsonl row -> Sample
# ---------------------------------------------------------------------------


def test_dataset_row_becomes_prompt_conditions_and_target(tmp_path):
    row = {
        "text": " A speaker ",
        "conditions": [
            {"type": "audio", "uri": "voice.wav", "role": "reference"},
            {"type": "image", "uri": "https://media.test/face.png", "role": "reference"},
        ],
        "target": {"visual": "target.mp4", "audio": "target.wav"},
    }
    sample = Dataset(_write_dataset(tmp_path, row))[0]
    assert sample.prompt == "A speaker"
    # order is semantic (<Audio 1>, <Picture 1>); relative uris anchor at the jsonl, urls stay
    assert sample.conditions == [
        {"type": "audio", "uri": str(tmp_path / "voice.wav"), "role": "reference"},
        {"type": "image", "uri": "https://media.test/face.png", "role": "reference"},
    ]
    assert sample.target == {"visual": str(tmp_path / "target.mp4"), "audio": str(tmp_path / "target.wav")}


# ---------------------------------------------------------------------------
# SFT: Sample -> local media -> cache key -> family encoder
# ---------------------------------------------------------------------------


def test_sft_cache_key_covers_order_and_file_revisions(tmp_path):
    for name in ("target.mp4", "target.wav", "voice.wav", "face.png"):
        (tmp_path / name).write_bytes(b"media")
    sample = Sample(
        prompt="A speaker",
        conditions=[
            {"type": "audio", "uri": str(tmp_path / "voice.wav"), "role": "reference"},
            {"type": "image", "uri": str(tmp_path / "face.png"), "role": "reference"},
        ],
        target={"visual": str(tmp_path / "target.mp4"), "audio": str(tmp_path / "target.wav")},
    )
    args = _args(tmp_path)
    local = dataclasses.replace(sample)
    localize_sft_media(local, tmp_path / "cache" / "media")
    assert (local.conditions, local.target) == (sample.conditions, sample.target)

    key = sft_sample_key(args, sample)[0]
    reordered = dataclasses.replace(sample, conditions=list(reversed(sample.conditions)))
    assert sft_sample_key(args, reordered)[0] != key
    (tmp_path / "voice.wav").write_bytes(b"edited reference")
    assert sft_sample_key(args, sample)[0] != key
    key = sft_sample_key(args, sample)[0]
    (tmp_path / "target.wav").write_bytes(b"edited target audio")
    assert sft_sample_key(args, sample)[0] != key


@pytest.mark.parametrize(
    "sample",
    [
        Sample(prompt="p", conditions=[{"type": "audio", "uri": "voice.wav", "role": "reference"}]),
        Sample(prompt="p", target={"visual": "target.mp4", "audio": "target.wav"}),
    ],
)
def test_wan_rejects_conditions_and_target_audio(tmp_path, sample):
    with pytest.raises(ValueError, match="supports neither conditions nor target audio"):
        wan2_2.encode_sample({}, sample, {}, None, _args(tmp_path))
