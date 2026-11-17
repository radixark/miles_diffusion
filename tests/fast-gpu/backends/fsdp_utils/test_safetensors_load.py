"""Two ranks read their own shards of a safetensors checkpoint, or fall back to the broadcast together.

    files hold the state dict --> rows [0:2] on rank 0, rows [2:4] on rank 1, GPU-resident and CPU-offloaded
    another name in the files --> the value rank 0 loaded, on both ranks
    LoRA, saved or new        --> rank 0's adapter on both ranks: the shards and the generator the broadcast leaves
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-5-gpu-h200", labels=["fsdp"])

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_ranks_read_their_own_shards(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("_safetensors_load_worker.py")),
            str(tmp_path),
        ],
        # The worker imports the CPU test's harness from the `tests` package at the repository root.
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(Path(__file__).resolve().parents[4]), os.environ.get("PYTHONPATH")))
            ),
            "PYTHONUNBUFFERED": "1",
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
