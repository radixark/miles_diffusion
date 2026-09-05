from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import json
from argparse import Namespace

import pytest
import torch

from miles.rollout.data_source import RolloutDataSourceWithBuffer


def _args(tmp_path, **overrides):
    prompt_path = tmp_path / "prompts.jsonl"
    with open(prompt_path, "w") as f:
        for i in range(16):
            f.write(json.dumps({"input": f"prompt {i}"}) + "\n")
    values = dict(
        rollout_global_dataset=True,
        prompt_data=str(prompt_path),
        input_key="input",
        metadata_key="metadata",
        rollout_seed=42,
        n_samples_per_prompt=2,
        save=str(tmp_path / "ckpt"),
        load=str(tmp_path / "ckpt"),
        buffer_filter_path=None,
    )
    values.update(overrides)
    return Namespace(**values)


def _cursor(source):
    return (source.sample_offset, source.epoch_id, source.sample_group_index, source.sample_index)


class TestCursorSnapshot:
    def test_save_uses_the_snapshot_not_the_live_cursor(self, tmp_path):
        args = _args(tmp_path)
        source = RolloutDataSourceWithBuffer(args)

        source.get_samples(4)
        source.snapshot(0)
        cursor_after_rollout_0 = _cursor(source)

        # A prefetched rollout advances the live cursor past the saved rollout.
        source.get_samples(4)
        source.snapshot(1)
        assert _cursor(source) != cursor_after_rollout_0

        source.save(0)

        restored = RolloutDataSourceWithBuffer(_args(tmp_path))
        restored.load(0)
        assert _cursor(restored) == cursor_after_rollout_0

    def test_save_without_snapshot_rejects(self, tmp_path):
        source = RolloutDataSourceWithBuffer(_args(tmp_path))
        with pytest.raises(ValueError, match="no cursor snapshot"):
            source.save(0)

    def test_snapshot_prunes_older_entries(self, tmp_path):
        source = RolloutDataSourceWithBuffer(_args(tmp_path))
        for rollout_id in range(5):
            source.get_samples(2)
            source.snapshot(rollout_id)
        assert sorted(source._cursor_snapshots) == [3, 4]

    def test_saved_state_matches_legacy_format(self, tmp_path):
        args = _args(tmp_path)
        source = RolloutDataSourceWithBuffer(args)
        source.get_samples(4)
        source.snapshot(0)
        source.save(0)

        state = torch.load(f"{args.save}/rollout/global_dataset_state_dict_0.pt")
        assert set(state) == {"sample_offset", "epoch_id", "sample_group_index", "sample_index", "metadata"}
