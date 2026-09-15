"""`_cvd_export` turns the recipe's device list into a `ray start` prefix:

    unset        -> ""                                     inherit the environment
    "4,5,2", 3   -> "export CUDA_VISIBLE_DEVICES=4,5,2 && " pin the raylet
    "0,1",   5   -> AssertionError                          ray would hand out unknown ids

Explicit runtime environment values are submitted through a private file, not logged commands.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import json
import shlex
from pathlib import Path

import pytest

from miles.utils.external_utils import command_utils as commands
from miles.utils.external_utils.command_utils import ExecuteTrainConfig, _cvd_export


def test_unset_inherits_the_environment():
    assert _cvd_export(ExecuteTrainConfig(), num_gpus_per_node=4) == ""


def test_set_exports_for_ray_start():
    config = ExecuteTrainConfig(cuda_visible_devices="4,5,2")
    assert _cvd_export(config, num_gpus_per_node=3) == "export CUDA_VISIBLE_DEVICES=4,5,2 && "


def test_count_mismatch_is_rejected():
    config = ExecuteTrainConfig(cuda_visible_devices="0,1")
    with pytest.raises(AssertionError, match="lists 2 GPU"):
        _cvd_export(config, num_gpus_per_node=5)


def test_submit_passes_explicit_env_in_private_runtime_env_file(monkeypatch):
    monkeypatch.delenv("TEST_RM_KEY", raising=False)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setattr(commands, "check_has_nvlink", lambda: False)
    submissions = []

    def execute(command, **kwargs):
        assert "test-secret-not-for-logs" not in command
        if "ray job submit" not in command:
            return ""
        tokens = shlex.split(command)
        runtime_path = Path(next(t.split("=", 1)[1] for t in tokens if t.startswith("--runtime-env=")))
        assert runtime_path.stat().st_mode & 0o777 == 0o600
        env = json.loads(runtime_path.read_text())["env_vars"]
        assert env["TEST_RM_KEY"] == "test-secret-not-for-logs"
        submissions.append(runtime_path)
        return ""

    monkeypatch.setattr(commands, "exec_command", execute)
    commands.execute_train(
        "--api-rm-config unused.yaml --rm-type api",
        1,
        extra_env_vars={"TEST_RM_KEY": "test-secret-not-for-logs"},
    )
    assert len(submissions) == 1
    assert not submissions[0].exists()
