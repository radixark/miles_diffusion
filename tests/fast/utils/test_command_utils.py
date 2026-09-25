"""`_cvd_export` turns the recipe's device list into a `ray start` prefix:

    unset        -> ""                                     inherit the environment
    "4,5,2", 3   -> "export CUDA_VISIBLE_DEVICES=4,5,2 && " pin the raylet
    "0,1",   5   -> AssertionError                          ray would hand out unknown ids
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import json
import shlex
import subprocess
import traceback

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


@pytest.mark.parametrize("redact_env_vars", [(), ("TEST_RM_KEY",)])
@pytest.mark.parametrize("submit_fails", [False, True])
def test_submit_logs_only_selected_env_values_redacted(monkeypatch, capsys, redact_env_vars, submit_fails):
    monkeypatch.delenv("TEST_RM_KEY", raising=False)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setattr(commands, "check_has_nvlink", lambda: False)
    submitted_envs = []

    def read_env(command):
        tokens = shlex.split(command)
        return json.loads(next(t.split("=", 1)[1] for t in tokens if t.startswith("--runtime-env-json=")))["env_vars"]

    def execute(argv, **kwargs):
        command = argv[2]
        if "ray job submit" not in command:
            return subprocess.CompletedProcess(argv, 0)
        env = read_env(command)
        assert env["TEST_RM_KEY"] == "test-secret-not-for-logs"
        submitted_envs.append(env)
        if submit_fails:
            raise subprocess.CalledProcessError(7, argv, output="job output", stderr="job error")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", execute)
    kwargs = dict(
        train_args="--custom-api-rm-config unused.yaml --rm-type custom_api",
        num_gpus_per_node=1,
        config=ExecuteTrainConfig(extra_env_vars='{"TEST_RM_KEY": "test-secret-not-for-logs"}'),
        extra_env_vars={"TEST_RM_KEY": "overridden-secret", "OTHER_API_KEY": "unselected-test-value"},
        redact_env_vars=redact_env_vars,
    )
    if submit_fails:
        with pytest.raises(subprocess.CalledProcessError) as error:
            commands.execute_train(**kwargs)
        assert error.value.returncode == 7
        assert (error.value.stdout, error.value.stderr) == ("job output", "job error")
    else:
        commands.execute_train(**kwargs)
    assert len(submitted_envs) == 1
    output = capsys.readouterr().out
    log_lines = [line for line in output.splitlines() if line.startswith("EXEC:") and "ray job submit" in line]
    assert len(log_lines) == 1
    log_cmd = log_lines[0].removeprefix("EXEC: ")
    logged_env = read_env(log_cmd)
    expected_env = dict(submitted_envs[0])
    if redact_env_vars:
        expected_env["TEST_RM_KEY"] = "***"
        assert "test-secret-not-for-logs" not in output
        if submit_fails:
            assert "test-secret-not-for-logs" not in repr(error.value)
            assert "test-secret-not-for-logs" not in "".join(traceback.format_exception(error.value))
    if submit_fails:
        assert error.value.cmd == ["bash", "-c", log_cmd]
    assert logged_env == expected_env
    assert logged_env["NCCL_DEBUG"] == "INFO"
    assert logged_env["OTHER_API_KEY"] == "unselected-test-value"
    assert "overridden-secret" not in output
