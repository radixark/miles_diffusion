"""Replaying a saved mixed-resolution rollout needs its resolution arguments too.

    saved samples of one resolution      ─► replays, strategy or not
    saved samples of several, no strategy ─► refused: the train side would not check a
                                             micro-batch holds one of them
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import ast
import pathlib
from argparse import Namespace

import pytest

from miles.utils.types import Sample


def _check_replay_resolutions():
    """The function out of rollout.py, which imports ray and sglang to be imported whole."""
    path = pathlib.Path(__file__).resolve().parents[3] / "miles/ray/rollout.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_check_replay_resolutions")
    namespace: dict = {"Sample": Sample}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_check_replay_resolutions"]


MIXED = [Sample(index=0, height=672, width=384), Sample(index=1, height=384, width=672)]


def test_mixed_samples_without_a_strategy_are_refused():
    with pytest.raises(ValueError, match="no --diffusion-resolution-strategy-path"):
        _check_replay_resolutions()(Namespace(diffusion_resolution_strategy_path=None), MIXED)


def test_mixed_samples_replay_with_the_strategy_set():
    _check_replay_resolutions()(Namespace(diffusion_resolution_strategy_path="hub.weighted_buckets"), MIXED)


@pytest.mark.parametrize(
    "samples",
    [
        [Sample(index=0), Sample(index=1)],  # a fixed-resolution run saves no sizes
        [Sample(index=0, height=512, width=512), Sample(index=1, height=512, width=512)],
    ],
)
def test_one_resolution_replays_either_way(samples):
    _check_replay_resolutions()(Namespace(diffusion_resolution_strategy_path=None), samples)
