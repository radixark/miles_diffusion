"""H3 Image full-weight training under FSDP2: ranks whose pairs come from different denoising steps.

    rank 0, pair of step 0:  ... pdd_proj_out.0      FSDP reduces the gradients of one wrap as a single
    rank 1, pair of step 2:  ... pdd_proj_out.2      buffer of the parameters that have one, in order

The heads have one shape, so the two buffers line up: without a gradient for every head on every rank, head 0
leaves the reduce with the mean of its own gradient and head 2's, and so does head 2.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="stage-b-5-gpu-h200",
)

import os
import subprocess
import sys
from pathlib import Path

import pytest

_WORKER = Path(__file__).with_name("_h3_image_head_gradients_worker.py")


def test_each_head_reduces_to_the_mean_of_its_own_gradients():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2", _WORKER],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
