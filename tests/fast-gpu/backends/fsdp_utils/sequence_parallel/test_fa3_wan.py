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


_WORKER = Path(__file__).with_name("_fa3_wan_worker.py")


def test_fa3_wan_fsdp_ulysses_training(tmp_path):
    if not torch.cuda.is_available():
        if os.environ.get("CI"):
            pytest.fail("CUDA CI has no working CUDA device")
        pytest.skip("CUDA unavailable; real Wan FA3 integration requires two compatible GPUs")
    assert torch.cuda.device_count() >= 2, "This regression requires two visible GPUs"
    evidence = tmp_path / "wan-fa3-sp2.json"
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(_WORKER.parents[5]), env.get("PYTHONPATH"))))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        str(_WORKER),
        "--seq-len",
        "1024",
        "--repeats",
        "2",
        "--output-json",
        str(evidence),
    ]
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        returncode = process.wait(timeout=360)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    assert returncode == 0, f"Wan FA3 worker exited {returncode}; see {evidence}"
    for path in (evidence, evidence.with_suffix(".rank1.json")):
        report = json.loads(path.read_text())
        assert report["status"] == "passed"
        assert report["mode"] == "real_fa3_bf16"
        assert len(report["steps"]) == 16
        assert len(report["kernel_calls"]) == 96
        assert all(call["completed"] for call in report["kernel_calls"] if call["phase"] == "forward")
        assert len(report["norm2_calls"]) == 48
        assert len(report["checks"]) == 2304 and all(check["passed"] for check in report["checks"])
        assert all(control["rejected"] for control in report["negative_controls"].values())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
