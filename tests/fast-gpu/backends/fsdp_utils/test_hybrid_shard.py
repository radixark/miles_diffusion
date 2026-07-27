"""Hybrid sharding with FSDP2: topology, gradients, weight sync, and DCP."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=90,
    suite="stage-c-5-gpu-h200",
    labels=["fsdp"],
)

import os
import subprocess
import sys
from pathlib import Path

import pytest

_WORKER = Path(__file__).with_name("_hybrid_shard_worker.py")


def _run_worker(*worker_args):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=4",
            str(_WORKER),
            *worker_args,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


@pytest.mark.parametrize(
    "worker_args",
    [
        ["--dp-replicate-size", "1"],
        ["--dp-replicate-size", "2"],
        ["--dp-replicate-size", "2", "--sequence-parallel-size", "2"],
    ],
    ids=["flat-dp4", "hybrid-shard-r2s2", "hybrid-shard-r2s1-sp2"],
)
def test_hybrid_shard(worker_args):
    _run_worker(*worker_args)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
