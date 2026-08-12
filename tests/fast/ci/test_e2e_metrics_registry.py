from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import json
import subprocess
import sys

import pytest

from tests.ci import ci_register
from tests.ci import e2e_metrics_registry as reg

_MINI_TEST = """\
from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=100,
    suite="stage-c-5-gpu-h200",
    script="scripts/example.py",
    args=["--num-rollout", "2"],
    metrics=["train/loss", "rollout/reward"],
)
"""


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps({"step_key": "step", **r}) + "\n" for r in rows))


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "STANDARDS_DIR", tmp_path / "standards")
    monkeypatch.delenv("MILES_E2E_METRICS_UPDATE", raising=False)
    return tmp_path


def test_record_then_check_roundtrip(sandbox, monkeypatch):
    jsonl = sandbox / "run.jsonl"
    _write_jsonl(jsonl, [{"step": 1, "m": 0.5}, {"step": 2, "m": 0.25}])
    monkeypatch.setenv("MILES_E2E_METRICS_UPDATE", "1")
    reg.check_or_update("test_foo.py", jsonl, ["m"])
    assert reg.standard_path_for("test_foo.py").exists()
    monkeypatch.delenv("MILES_E2E_METRICS_UPDATE")
    reg.check_or_update("test_foo.py", jsonl, ["m"])


def test_strict_mismatch_fails_but_tolerance_passes(sandbox, monkeypatch):
    good = sandbox / "good.jsonl"
    drifted = sandbox / "drifted.jsonl"
    _write_jsonl(good, [{"step": 1, "m": 0.5}])
    _write_jsonl(drifted, [{"step": 1, "m": 0.5005}])
    monkeypatch.setenv("MILES_E2E_METRICS_UPDATE", "1")
    reg.check_or_update("test_foo.py", good, ["m"])
    monkeypatch.delenv("MILES_E2E_METRICS_UPDATE")
    with pytest.raises(AssertionError, match="strict"):
        reg.check_or_update("test_foo.py", drifted, ["m"])
    reg.check_or_update("test_foo.py", drifted, ["m"], tolerances={"m": {"atol": 1e-3}})


def test_recorded_nan_matches_itself_but_not_a_number(sandbox, monkeypatch):
    # fp16 runs emit grad_norm=nan on the step the scaler overflows its init scale.
    nan_run = sandbox / "nan.jsonl"
    finite_run = sandbox / "finite.jsonl"
    _write_jsonl(nan_run, [{"step": 1, "m": float("nan")}, {"step": 2, "m": 0.25}])
    _write_jsonl(finite_run, [{"step": 1, "m": 0.5}, {"step": 2, "m": 0.25}])
    monkeypatch.setenv("MILES_E2E_METRICS_UPDATE", "1")
    reg.check_or_update("test_foo.py", nan_run, ["m"])
    monkeypatch.delenv("MILES_E2E_METRICS_UPDATE")
    reg.check_or_update("test_foo.py", nan_run, ["m"])
    # A run that stops overflowing is a real change, not a match.
    with pytest.raises(AssertionError, match="strict"):
        reg.check_or_update("test_foo.py", finite_run, ["m"])
    # Nor does a tolerance let a number pass against a recorded NaN.
    with pytest.raises(AssertionError, match="tol"):
        reg.check_or_update("test_foo.py", finite_run, ["m"], tolerances={"m": {"atol": 1e9}})


def test_series_shape_mismatches_fail(sandbox, monkeypatch):
    std = sandbox / "std.jsonl"
    _write_jsonl(std, [{"step": 1, "m": 0.5}, {"step": 2, "m": 0.25}])
    monkeypatch.setenv("MILES_E2E_METRICS_UPDATE", "1")
    reg.check_or_update("test_foo.py", std, ["m"])
    monkeypatch.delenv("MILES_E2E_METRICS_UPDATE")
    short = sandbox / "short.jsonl"
    _write_jsonl(short, [{"step": 1, "m": 0.5}])
    with pytest.raises(AssertionError, match="points"):
        reg.check_or_update("test_foo.py", short, ["m"])
    reindexed = sandbox / "reindexed.jsonl"
    _write_jsonl(reindexed, [{"step": 1, "m": 0.5}, {"step": 3, "m": 0.25}])
    with pytest.raises(AssertionError, match="step"):
        reg.check_or_update("test_foo.py", reindexed, ["m"])


def test_missing_metric_in_recording_fails(sandbox):
    jsonl = sandbox / "run.jsonl"
    _write_jsonl(jsonl, [{"step": 1, "m": 0.5}])
    with pytest.raises(AssertionError, match="never emitted"):
        reg.load_series(jsonl, ["m", "absent"])


def test_metrics_spec_extracted_via_ast(sandbox):
    mini = sandbox / "test_mini.py"
    mini.write_text(_MINI_TEST)
    assert reg._test_metrics_spec(str(mini)) == ["train/loss", "rollout/reward"]


def test_recipe_runs_under_this_interpreter_with_its_args(sandbox, monkeypatch):
    """The registry launches a Python recipe directly — no shell in between."""
    launched = []
    monkeypatch.setattr(reg, "RECORDINGS_DIR", sandbox / "recordings")
    monkeypatch.setattr(reg, "_cleanup_gpu_state", lambda: None)
    monkeypatch.setattr(reg, "check_or_update", lambda *a, **k: None)
    monkeypatch.setattr(
        reg.subprocess,
        "run",
        lambda cmd, **kw: launched.append((cmd, kw)) or subprocess.CompletedProcess(cmd, 0),
    )

    caller = {"__name__": "__main__", "__file__": str(sandbox / "test_mini.py")}
    exec("reg.register_e2e_ci(**kwargs)", {"reg": reg, **caller}, {"reg": reg, "kwargs": _E2E_CALL})

    ((cmd, kwargs),) = launched
    assert cmd[0] == sys.executable and "bash" not in cmd
    assert cmd[1:] == ["-u", str(reg.REPO_ROOT / "scripts/example.py"), "--num-rollout", "2"]
    assert kwargs["env"]["MILES_METRICS_JSONL"].endswith("test_mini.jsonl")


_E2E_CALL = dict(
    est_time=100,
    suite="stage-c-5-gpu-h200",
    script="scripts/example.py",
    args=["--num-rollout", "2"],
    metrics=["m"],
)


def test_register_e2e_ci_parses_like_other_registers(sandbox):
    mini = sandbox / "test_mini.py"
    mini.write_text(_MINI_TEST)
    (entry,) = ci_register.ut_parse_one_file(str(mini))
    assert entry.backend is ci_register.HWBackend.CUDA
    assert entry.suite == "stage-c-5-gpu-h200"
    assert entry.est_time == 100.0
    bad = sandbox / "test_bad.py"
    bad.write_text(_MINI_TEST.replace("args=", "typo_kwarg="))
    with pytest.raises(ValueError, match="unknown argument"):
        ci_register.ut_parse_one_file(str(bad))
