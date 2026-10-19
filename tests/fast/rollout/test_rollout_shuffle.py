"""--rollout-shuffle permutes the prompt order per epoch, deterministically from --rollout-seed.

Mental model (5 prompts p0..p4, 3 per rollout):

    off : p0 p1 p2 | p3 p4 p0 | p1 p2 p3 ...          file order, wraps around
    on  : perm(seed, epoch 0) | ... wrap -> perm(seed, epoch 1) | ...
          save() at the wrap and load() elsewhere resumes at the same offset of the same permutation

Covered: off keeps file order (1); on yields a permutation of every prompt that differs from the
file order and repeats for the same seed (2); the epoch wrap re-permutes and load() reproduces
the order after the wrap (3); a save() mid-permutation resumes on the exact next prompt and
carries through the following wrap (4).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import json
from argparse import Namespace

from miles.rollout.data_source import RolloutDataSourceWithBuffer

PROMPTS = [f"p{i}" for i in range(5)]


def _args(tmp_path, **overrides):
    path = tmp_path / "train.jsonl"
    if not path.exists():
        path.write_text("".join(json.dumps({"input": p}) + "\n" for p in PROMPTS))
    base = dict(
        rollout_global_dataset=True,
        prompt_data=str(path),
        input_key="input",
        metadata_key="metadata",
        rollout_seed=7,
        rollout_shuffle=True,
        n_samples_per_prompt=1,
        save=str(tmp_path / "ckpt"),
        load=None,
        buffer_filter_path=None,
    )
    return Namespace(**{**base, **overrides})


def _prompts(groups):
    return [group[0].prompt for group in groups]


def test_off_reads_the_file_order(tmp_path):
    source = RolloutDataSourceWithBuffer(_args(tmp_path, rollout_shuffle=False))

    assert _prompts(source.get_samples(3)) == ["p0", "p1", "p2"]
    assert _prompts(source.get_samples(3)) == ["p3", "p4", "p0"]


def test_on_permutes_every_prompt_deterministically(tmp_path):
    first = _prompts(RolloutDataSourceWithBuffer(_args(tmp_path)).get_samples(5))
    again = _prompts(RolloutDataSourceWithBuffer(_args(tmp_path)).get_samples(5))

    assert sorted(first) == PROMPTS
    assert first != PROMPTS
    assert again == first


def test_epoch_wrap_repermutes_and_load_resumes_it(tmp_path):
    source = RolloutDataSourceWithBuffer(_args(tmp_path))
    epoch0 = _prompts(source.get_samples(5))
    source.save(rollout_id=1)
    epoch1_start = _prompts(source.get_samples(3))

    resumed = RolloutDataSourceWithBuffer(_args(tmp_path, load=str(tmp_path / "ckpt")))
    resumed.load(rollout_id=1)

    assert source.dataset.epoch_id == 1
    assert epoch1_start != epoch0[:3]
    assert _prompts(resumed.get_samples(3)) == epoch1_start


def test_resume_mid_epoch_continues_the_same_permutation(tmp_path):
    """A resumed run must not restart the permutation or re-serve prompts it already used."""
    source = RolloutDataSourceWithBuffer(_args(tmp_path))
    source.get_samples(2)
    source.save(rollout_id=2)
    continued = [_prompts(source.get_samples(2)) for _ in range(3)]

    resumed = RolloutDataSourceWithBuffer(_args(tmp_path, load=str(tmp_path / "ckpt")))
    resumed.load(rollout_id=2)

    assert [_prompts(resumed.get_samples(2)) for _ in range(3)] == continued
    assert resumed.dataset.epoch_id == 1


def test_request_larger_than_dataset_wraps_multiple_epochs(tmp_path):
    """A rollout asking for more prompts than the dataset holds still gets exactly that many."""
    source = RolloutDataSourceWithBuffer(_args(tmp_path, rollout_shuffle=False))

    assert _prompts(source.get_samples(12)) == PROMPTS + PROMPTS + ["p0", "p1"]
    assert source.sample_offset == 2
    assert source.epoch_id == 2
    assert _prompts(source.get_samples(3)) == ["p2", "p3", "p4"]


def test_shuffled_request_larger_than_dataset_covers_each_epoch(tmp_path):
    source = RolloutDataSourceWithBuffer(_args(tmp_path))
    drawn = _prompts(source.get_samples(11))

    assert len(drawn) == 11
    assert sorted(drawn[:5]) == PROMPTS
    assert sorted(drawn[5:10]) == PROMPTS
    assert source.sample_offset == 1
