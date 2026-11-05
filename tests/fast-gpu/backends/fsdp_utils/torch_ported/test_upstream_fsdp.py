"""Selected PyTorch v2.11 FSDP regression tests with the Miles patch installed."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=240,
    suite="stage-b-5-gpu-h200",
    labels=["torch"],
)

import os
import subprocess
import sys
from pathlib import Path

import pytest


_DIR = Path(__file__).parent


def _run_ported_test(filename, *selectors):
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONPATH"] = os.pathsep.join((str(_DIR), env.get("PYTHONPATH", "")))
    env["PYTHONUNBUFFERED"] = "1"
    # CI=true makes torch's run_tests write XML reports, which needs the absent xmlrunner
    env.pop("CI", None)
    result = subprocess.run(
        [sys.executable, str(_DIR / filename), *selectors],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # site.py swallows sitecustomize errors, which would run the ported tests unpatched
    assert "Error in sitecustomize" not in result.stderr, result.stderr


def test_fully_shard_mixed_precision():
    _run_ported_test(
        "ported_fully_shard_mixed_precision.py",
        "TestFullyShardMixedPrecisionTraining.test_compute_dtype",
        "TestFullyShardMixedPrecisionTraining.test_reduce_dtype",
        "TestFullyShardMixedPrecisionTraining.test_grad_acc_with_reduce_dtype",
    )


def test_fully_shard_collective_ops():
    _run_ported_test(
        "ported_fully_shard_comm.py",
        "TestFullyShardCollectiveOps",
    )


def test_fully_shard_frozen():
    _run_ported_test("ported_fully_shard_frozen.py")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
