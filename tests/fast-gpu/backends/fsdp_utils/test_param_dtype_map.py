"""FSDP2 per-parameter mixed precision."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=480,
    suite="stage-b-5-gpu-h200",
    labels=["fsdp"],
)

import os
import subprocess
import sys
from pathlib import Path

import pytest


_E2E_WORKER = Path(__file__).with_name("_param_dtype_map_worker.py")
_ADVERSARIAL_WORKER = Path(__file__).with_name("_param_dtype_map_adversarial_worker.py")
_INTEGRATION_WORKER = Path(__file__).with_name("_param_dtype_map_integration_worker.py")
_VALIDATION_WORKER = Path(__file__).with_name("_param_dtype_map_validation_worker.py")
_EQUIVALENCE_WORKER = Path(__file__).with_name("_param_dtype_map_equivalence_worker.py")


def _run_worker(worker, *args):
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=4",
            str(worker),
            *args,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_param_dtype_map_full_size_blocks():
    _run_worker(_E2E_WORKER)


def test_param_dtype_map_apply_fsdp2_integration():
    _run_worker(_INTEGRATION_WORKER)


@pytest.mark.parametrize(
    "case",
    [
        "fully-shard-prime-dtype-zoo",
        "hybrid-grouped-prime-dtype-zoo",
    ],
)
def test_param_dtype_map_adversarial(case):
    _run_worker(_ADVERSARIAL_WORKER, case)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate-multi-module-fqn",
        "same-fqn-separate-wraps",
        "same-fqn-shared-parameter",
        "shared-parameter-aliases-same-dtype",
        "shared-parameter-alias-conflict",
        "grouped-layer-norm-wrap",
        "unknown-fqn",
        "mixed-requires-reduce-dtype",
        "frozen-override-no-reduce-dtype",
        "mixed-forward-backward",
        "empty-map-delegation",
        "reduce-scatter-empty-grad",
    ],
)
def test_param_dtype_map_validation(case):
    _run_worker(_VALIDATION_WORKER, case)


@pytest.mark.parametrize("topology", ["1x4", "2x2", "4x1"])
def test_param_dtype_map_uniform_map_matches_stock(topology):
    """A map assigning every param one dtype must be bitwise-equal to plain param_dtype."""
    _run_worker(_EQUIVALENCE_WORKER, topology)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
