"""Two-rank FSDP role loading and reference forwards match dense models.

    actor: saved LoRA --> FSDP --> forward/backward/AdamW --> dense gradient/weight oracle
    ref: full / LoRA --> FSDP + GPU / CPUOffloadPolicy --> frozen reference output
    teacher: LoRA   --> FSDP + GPU / CPUOffloadPolicy --> frozen teacher output
    LoRA base       --> adapter-disabled reference --> unchanged actor buffers and flags

Every role has two components; rank 0 loads checkpoints and rank 1 starts on meta.
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
def test_independent_fsdp_references(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("_reference_models_worker.py")),
            str(tmp_path),
        ],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
