"""Nested precision wrap units under real FSDP2 on 2 gloo ranks; assertions live in the worker."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import os
import subprocess
import sys
from pathlib import Path

import pytest

_WORKER = Path(__file__).with_name("_precision_wrap_worker.py")


def test_precision_wrap_units():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(_WORKER),
        ],
        check=True,
        env=env,
        timeout=300,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
