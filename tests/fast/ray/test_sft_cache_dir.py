"""Cache addressing for the SFT data manager."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace

from miles.ray.sft_data_manager import sft_cache_dir


def _args(**overrides):
    base = dict(hf_checkpoint="ckpt", sft_height=480, sft_width=832, sft_num_frames=81, sft_frame_stride=2)
    base.update(overrides)
    return Namespace(**base)


def test_cache_dir_is_stable_and_content_addressed(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"video": "a.mp4", "prompt": "p"}\n')

    base = sft_cache_dir(_args(), data)
    assert base == sft_cache_dir(_args(), data)
    assert base.parent == tmp_path / ".sft_cache"

    assert sft_cache_dir(_args(sft_height=512), data) != base
    assert sft_cache_dir(_args(sft_frame_stride=1), data) != base
    assert sft_cache_dir(_args(hf_checkpoint="other"), data) != base

    data.write_text('{"video": "b.mp4", "prompt": "p"}\n')
    assert sft_cache_dir(_args(), data) != base
