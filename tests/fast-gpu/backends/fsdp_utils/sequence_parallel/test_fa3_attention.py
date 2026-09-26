from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="stage-c-5-gpu-h200", labels=["fsdp"])

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch


_WORKER = Path(__file__).with_name("_fa3_attention_worker.py")
_REPEATS = 20


def _run_worker(command, *, env, timeout):
    # torchrun owns several CUDA workers. Killing only its parent on timeout
    # can leave those descendants using the rented GPUs after pytest exits.
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


@pytest.mark.parametrize(
    ("world_size", "dtype"),
    [(1, "bfloat16"), (2, "bfloat16"), (4, "bfloat16"), (1, "float16")],
    ids=["bf16-sp1", "bf16-sp2", "bf16-sp4", "fp16-sp1"],
)
def test_fa3_dispatch_ulysses_forward_backward_and_repeatability(world_size, dtype, tmp_path):
    if not torch.cuda.is_available():
        if os.environ.get("CI"):
            pytest.fail("CUDA CI has no working CUDA device; this is not a passing FA3 regression")
        pytest.skip("CUDA unavailable; FA3 GPU regression requires a GPU supported by the installed FA3 build")
    assert (
        torch.cuda.device_count() >= world_size
    ), f"This case needs {world_size} visible GPUs; select a smaller case explicitly instead of skipping coverage"
    evidence = tmp_path / f"fa3-{dtype}-sp{world_size}.json"
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(_WORKER.parents[5]), env.get("PYTHONPATH"))))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={world_size}",
        str(_WORKER),
        "--dtype",
        dtype,
        "--seq-len",
        "1024",
        "--repeats",
        str(_REPEATS),
        "--output-json",
        str(evidence),
    ]
    _run_worker(command, env=env, timeout=240)
    report = json.loads(evidence.read_text())
    assert report["status"] == "passed"
    assert report["world_size"] == world_size
    # 8 reference + 4*(repeats-1) repeated + 3 projection + 4 mode-off comparisons.
    assert len(report["checks"]) == 4 * _REPEATS + 11
    assert len(report["kernel_calls_per_rank"]) == _REPEATS + 2
    assert [call["deterministic"] for call in report["kernel_calls_per_rank"]] == [True] * (_REPEATS + 1) + [False]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
