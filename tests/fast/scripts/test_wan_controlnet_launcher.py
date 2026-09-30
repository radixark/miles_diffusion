"""The launcher must keep local-path and resume provenance semantics intact."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "run_diffusion_sft_wan_controlnet.py"
FUNCTIONS = runpy.run_path(str(SCRIPT))


def test_local_base_path_resolves_before_launcher_changes_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "base").mkdir()
    (tmp_path / "train.jsonl").write_text("{}\n")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = FUNCTIONS["parser"]().parse_args(
        [
            "--base-model",
            "base",
            "--data-jsonl",
            "train.jsonl",
            "--output-dir",
            "run",
        ]
    )
    values = FUNCTIONS["build_train_args"](args)
    assert values[values.index("--hf-checkpoint") + 1] == str(tmp_path / "base")
    assert values[values.index("--sft-encoder-checkpoint") + 1] == str(tmp_path / "base")


def test_resume_keeps_initial_launch_manifest(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path / "run", resume_checkpoint=None)
    initial = FUNCTIONS["write_launch_manifest"](args, ["--lr", "0.001"])
    original = initial.read_bytes()
    with pytest.raises(FileExistsError):
        FUNCTIONS["write_launch_manifest"](args, ["--lr", "0.002"])
    args.resume_checkpoint = tmp_path / "run" / "checkpoints"
    resumed = FUNCTIONS["write_launch_manifest"](args, ["--load", str(args.resume_checkpoint)])
    assert resumed != initial
    assert initial.read_bytes() == original
    assert json.loads(resumed.read_text())["train_args"] == ["--load", str(args.resume_checkpoint)]
