from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=240,
    suite="stage-c-5-gpu-h200",
    labels=["fsdp"],
)

import os
import subprocess
import sys
from pathlib import Path

import pytest


_WORKER = Path(__file__).with_name("_attention_parity_worker.py")


def _run_worker(ulysses_degree, *, deterministic=False, ring_backend=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=4",
        str(_WORKER),
        "--ulysses-degree",
        str(ulysses_degree),
    ]
    if ring_backend is not None:
        command += ["--ring-backend", ring_backend]
    if deterministic:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        command.append("--deterministic")
    subprocess.run(
        command,
        check=True,
        env=env,
    )


@pytest.mark.parametrize(
    ("ulysses_degree", "ring_backend"),
    [(4, None), (2, None), (1, None), (2, "_native_cudnn"), (1, "_native_cudnn")],
    ids=["sp4-u4r1", "sp4-u2r2", "sp4-u1r4", "sp4-u2r2-cudnn", "sp4-u1r4-cudnn"],
)
def test_usp_forward_backward_matches_full_sequence_sdpa(ulysses_degree, ring_backend):
    _run_worker(ulysses_degree, ring_backend=ring_backend)


@pytest.mark.parametrize(
    "ulysses_degree",
    [4, 2, 1],
    ids=["sp4-u4r1", "sp4-u2r2", "sp4-u1r4"],
)
def test_usp_forward_backward_is_bitwise_repeatable_in_deterministic_mode(ulysses_degree):
    _run_worker(ulysses_degree, deterministic=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
