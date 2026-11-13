"""PDD FSDP parity through the generic loader hooks."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="stage-b-5-gpu-h200", labels=["fsdp"])

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("world_size", [1, 2])
def test_pdd_fsdp_parity(tmp_path, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA devices")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            str(Path(__file__).with_name("_pdd_worker.py")),
            "--output",
            str(tmp_path),
        ],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "verification.json").read_text())
    assert report["state"] == "passed" and report["world_size"] == world_size


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
