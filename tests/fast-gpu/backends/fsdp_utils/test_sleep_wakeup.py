"""FSDP2 sleep/wakeup through pinned host memory."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    suite="stage-b-5-gpu-h200",
    labels=["fsdp"],
)

import os
import subprocess
import sys
from pathlib import Path

_WORKER = Path(__file__).with_name("_sleep_wakeup_worker.py")


def test_fsdp_sleep_wakeup_uses_pinned_memory():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(_WORKER),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
