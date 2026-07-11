"""E2E metrics registry: per-step metric regression against a committed standard.

A test is one register_e2e_ci(...) call: suite placement is parsed via AST by
ci_register.py; executing the file runs `script` and checks `metrics` against
tests/ci/fixtures/e2e_standards/<test_stem>.json. Standards are updated by the
PR author, never by CI:

    python tests/ci/e2e_metrics_registry.py record --test <test-file>
    python tests/ci/e2e_metrics_registry.py register --test <test-file> [--metrics <jsonl>]
"""

import argparse
import ast
import json
import math
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STANDARDS_DIR = Path(__file__).resolve().parent / "fixtures" / "e2e_standards"
RECORDINGS_DIR = REPO_ROOT / "logs" / "e2e_metrics"


def standard_path_for(test_file: str | Path) -> Path:
    return STANDARDS_DIR / (Path(test_file).stem + ".json")


def metrics_path_for(test_file: str | Path) -> Path:
    return RECORDINGS_DIR / (Path(test_file).stem + ".jsonl")


def register_e2e_ci(
    est_time: float,
    suite: str,
    *,
    script: str,
    metrics: list[str],
    env: dict[str, str] | None = None,
    tolerances: dict[str, dict] | None = None,
    labels: list[str] | None = None,
    nightly: bool = False,
    disabled: str | None = None,
) -> None:
    """Dual-role marker: AST-parsed for suite scheduling (est_time/suite/...)
    AND executed when the test file runs (script/metrics/env/tolerances)."""
    del est_time, suite, labels, nightly, disabled
    caller = sys._getframe(1).f_globals
    if caller.get("__name__") != "__main__":
        return
    _cleanup_gpu_state()
    test_file = caller["__file__"]
    recording = metrics_path_for(test_file)
    recording.parent.mkdir(parents=True, exist_ok=True)
    recording.unlink(missing_ok=True)
    run_env = dict(os.environ) | (env or {}) | {"MILES_METRICS_JSONL": str(recording)}
    subprocess.run(["bash", str(REPO_ROOT / script)], check=True, cwd=REPO_ROOT, env=run_env)
    check_or_update(test_file, recording, metrics, tolerances)


def _cleanup_gpu_state() -> None:
    """Kill ray/sglang leftovers from a previous test in this job and wait for
    GPU memory to drain (engines take minutes to die after their driver exits)."""
    subprocess.run(
        "pkill -9 -f 'ray::'; pkill -9 -f raylet; pkill -9 -f gcs_server; pkill -9 -f sgl_diffusion; ray stop --force; true",
        shell=True,
        capture_output=True,
    )
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    query = (
        ["nvidia-smi"] + (["-i", gpus] if gpus else []) + ["--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    for _ in range(60):
        out = subprocess.run(query, capture_output=True, text=True)
        if out.returncode != 0 or all(int(x) < 2048 for x in out.stdout.split()):
            return
        time.sleep(5)
    print("[e2e-metrics] WARNING: GPU memory still occupied after cleanup wait", flush=True)


def load_series(jsonl_path: str | Path, metrics: list[str]) -> dict[str, list[list[float]]]:
    """Extract [step, value] series (in emission order) for each metric key."""
    series: dict[str, list[list[float]]] = {m: [] for m in metrics}
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            step = rec.get(rec.get("step_key"))
            for m in metrics:
                if m in rec:
                    series[m].append([step, rec[m]])
    empty = [m for m in metrics if not series[m]]
    assert not empty, f"metrics never emitted by {jsonl_path} (check key names): {empty}"
    return series


def _values_match(got: float, want: float, tol: dict | None) -> bool:
    if tol is None:
        return got == want
    return math.isclose(got, want, rel_tol=tol.get("rtol", 0.0), abs_tol=tol.get("atol", 0.0))


def _write_standard(test_file: str | Path, series: dict, source: str) -> None:
    path = standard_path_for(test_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"commit": os.environ.get("GITHUB_SHA", "local"), "source": source}, "metrics": series}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[e2e-metrics] RECORDED {path} ({', '.join(series)}) — review + commit it in your PR")


def check_or_update(
    test_file: str,
    jsonl_path: str | Path,
    metrics: list[str],
    tolerances: dict[str, dict] | None = None,
) -> None:
    """Check the run's series against the standard (strict unless a tolerance
    is declared); with MILES_E2E_METRICS_UPDATE=1, record it instead."""
    tolerances = tolerances or {}
    standard_path = standard_path_for(test_file)
    series = load_series(jsonl_path, metrics)

    if os.environ.get("MILES_E2E_METRICS_UPDATE"):
        _write_standard(test_file, series, Path(test_file).name)
        return

    assert standard_path.exists(), (
        f"no standard at {standard_path} — record one on a GPU machine:\n"
        f"    python tests/ci/e2e_metrics_registry.py record --test {test_file}"
    )
    standard = json.loads(standard_path.read_text())["metrics"]

    failures: list[str] = []
    for m in metrics:
        want_series = standard.get(m)
        if want_series is None:
            failures.append(f"{m}: missing from standard (re-record)")
            continue
        got_series = series[m]
        if len(got_series) != len(want_series):
            failures.append(f"{m}: {len(got_series)} points vs standard {len(want_series)}")
            continue
        tol = tolerances.get(m)
        for i, ((gs, gv), (ws, wv)) in enumerate(zip(got_series, want_series, strict=True)):
            if gs != ws:
                failures.append(f"{m}[{i}]: step {gs} vs standard step {ws}")
            elif not _values_match(gv, wv, tol):
                failures.append(
                    f"{m}[{i}] @step {gs}: {gv!r} vs standard {wv!r}" + (f" (tol={tol})" if tol else " (strict)")
                )

    if failures:
        report = "\n".join(f"  - {f}" for f in failures)
        raise AssertionError(
            f"e2e metrics mismatch vs standard {standard_path.name} ({len(failures)} failure(s)):\n{report}\n"
            f"If intentional, record a new standard and commit it in your PR:\n"
            f"    python tests/ci/e2e_metrics_registry.py record --test {test_file}"
        )
    print(f"[e2e-metrics] PASSED: {len(metrics)} metric series match {standard_path.name}")


def _test_metrics_spec(test_path: str) -> list[str]:
    """Read the `metrics` kwarg of register_e2e_ci() via AST (never executes)."""
    tree = ast.parse(Path(test_path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "register_e2e_ci":
            for kw in node.keywords:
                if kw.arg == "metrics":
                    return ast.literal_eval(kw.value)
    raise ValueError(f"{test_path}: no register_e2e_ci(metrics=[...]) call found")


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E metrics registry tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="Rerun a test's example and record the result as its standard")
    rec.add_argument("--test", required=True)
    rec.add_argument("--list-only", action="store_true", help="Print what would be recorded and exit")
    reg = sub.add_parser("register", help="Register a past run's metrics as a test's standard")
    reg.add_argument("--test", required=True)
    reg.add_argument("--metrics", help="metrics.jsonl to register (default: the test's own last recording)")
    args = parser.parse_args()

    if args.cmd == "record":
        if args.list_only:
            print(f"[e2e-metrics] would record {standard_path_for(args.test)}")
            print(f"[e2e-metrics] metrics: {', '.join(_test_metrics_spec(args.test))}")
            return
        os.environ["MILES_E2E_METRICS_UPDATE"] = "1"
        runpy.run_path(args.test, run_name="__main__")
    else:
        metrics_path = args.metrics or metrics_path_for(args.test)
        series = load_series(metrics_path, _test_metrics_spec(args.test))
        _write_standard(args.test, series, f"registered from {metrics_path}")


if __name__ == "__main__":
    main()
