"""Per-sample content addressing for the SFT cache.

    Sample (prompt, conditions, target) + media file fingerprints + SFT_CACHE_KEY_ARGS
        ──sha256──► (cache file name, latent seed)

What each test pins: keys are deterministic; each encode arg, the prompt and the file revision change the key;
args no encoder reads leave the name and seed alone; keys differ per sample and per model family.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import os
from argparse import Namespace

from miles.rollout.sft_rollout import sft_sample_key
from miles.utils.types import Sample


def _args(**overrides):
    base = dict(
        diffusion_model_family="wan2_2",
        sft_encoder_checkpoint="ckpt",
        diffusion_height=480,
        diffusion_width=832,
        diffusion_output_num_frames=81,
        sft_frame_stride=2,
    )
    base.update(overrides)
    return Namespace(**base)


def _sample(video, prompt="p"):
    return Sample(prompt=prompt, target={"visual": str(video)})


def test_key_is_deterministic_with_seed(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x" * 100)
    sample = _sample(video)

    name, seed = sft_sample_key(_args(), sample)
    assert (name, seed) == sft_sample_key(_args(), sample)
    assert name.endswith(".pt")
    assert 0 <= seed < 2**63


def test_key_invalidates_per_axis(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x" * 100)
    sample = _sample(video)
    base_name, base_seed = sft_sample_key(_args(), sample)

    assert sft_sample_key(_args(diffusion_height=512), sample)[0] != base_name
    assert sft_sample_key(_args(sft_frame_stride=1), sample)[0] != base_name
    assert sft_sample_key(_args(sft_encoder_checkpoint="other"), sample)[0] != base_name
    assert sft_sample_key(_args(), _sample(video, prompt="q"))[0] != base_name

    video.write_bytes(b"y" * 101)
    assert sft_sample_key(_args(), sample)[0] != base_name

    video.write_bytes(b"x" * 100)
    os.utime(video, ns=(1, 1))
    replaced_name, replaced_seed = sft_sample_key(_args(), sample)
    assert replaced_name != base_name
    assert replaced_seed != base_seed


def test_key_ignores_args_no_encoder_reads(tmp_path):
    # Offloading moves the encoder between devices; the encoded latents and their seed are unchanged.
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x" * 100)
    sample = _sample(video)
    assert sft_sample_key(_args(sft_offload_encoder=True), sample) == sft_sample_key(_args(), sample)


def test_key_is_per_sample(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"x" * 100)
    b.write_bytes(b"x" * 100)
    name_a, _ = sft_sample_key(_args(), _sample(a))
    name_b, _ = sft_sample_key(_args(), _sample(b))
    assert name_a != name_b


def test_key_changes_with_model_family(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x" * 100)
    sample = _sample(video)
    name_wan, _ = sft_sample_key(_args(), sample)
    name_other, _ = sft_sample_key(_args(diffusion_model_family="other"), sample)
    assert name_wan != name_other
