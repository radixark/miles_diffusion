"""Replaying a saved mixed-resolution rollout needs its resolution arguments too.

Driven through RolloutManager._get_rollout_data, the entry point that loads a dump, so
deleting the check from it fails these rather than leaving them green.

    dump of one resolution       ─► replays, strategy or not
    dump of several, no strategy ─► refused: the train side would not check that a
                                    micro-batch holds one of them
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.ray.rollout import RolloutManager
from miles.utils.types import Sample

# the class behind @ray.remote, as test_generated_output_store reaches it
_RolloutManager = RolloutManager.__ray_metadata__.modified_class


def _dump(tmp_path, samples) -> str:
    path = tmp_path / "rollout_{rollout_id}.pt"
    torch.save({"samples": [sample.to_dict() for sample in samples]}, str(path).format(rollout_id=0))
    return str(path)


def _load(path, *, strategy=None):
    """The real loader, on a self holding only what its load branch reads."""
    manager = SimpleNamespace(
        args=Namespace(
            load_debug_rollout_data=path,
            load_debug_rollout_data_subsample=None,
            diffusion_resolution_strategy_path=strategy,
        )
    )
    return _RolloutManager._get_rollout_data(manager, rollout_id=0)


MIXED = [Sample(index=0, height=672, width=384), Sample(index=1, height=384, width=672)]


def test_a_mixed_dump_without_a_strategy_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no --diffusion-resolution-strategy-path"):
        _load(_dump(tmp_path, MIXED))


def test_a_mixed_dump_replays_with_the_strategy_set(tmp_path):
    data, metrics = _load(_dump(tmp_path, MIXED), strategy="hub.weighted_buckets")

    assert [(s.height, s.width) for s in data] == [(672, 384), (384, 672)]
    assert metrics is None


@pytest.mark.parametrize(
    "samples",
    [
        [Sample(index=0), Sample(index=1)],  # a fixed-resolution run saves no sizes
        [Sample(index=0, height=512, width=512), Sample(index=1, height=512, width=512)],
    ],
)
def test_one_resolution_replays_either_way(tmp_path, samples):
    data, _ = _load(_dump(tmp_path, samples))

    assert len(data) == 2
