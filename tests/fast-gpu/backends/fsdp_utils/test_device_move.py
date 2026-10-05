"""Two-rank sleep/wake: FSDP shards + buffers + AdamW -> pinned CPU -> CUDA.

    trainable / frozen  x  reshard / no-reshard   (native CPU offload: never sleeps)
                                    |
                   uneven shards + tied/nonpersistent buffers
                                    |
                sleep --> pinned storage --> wake --> same outputs/updates

    GPU/CPU actor + EMA on the same device --> average --> inference --> actor forward/backward
                                     |
                      pinned sleep / wake + DCP round trip
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
def test_fsdp_sleep_wake():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("_device_move_worker.py")),
        ],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
