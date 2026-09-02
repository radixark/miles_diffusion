from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import json
from pathlib import Path

import datasets

from scripts import run_diffusion_grpo_sd3_hps_sglang as recipe


def _write_hpdv2_annotation(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {"prompt": " first prompt ", "human_preference": [0, 1], "image_path": ["0.jpg", "1.jpg"]},
                {"prompt": "first prompt", "human_preference": [1, 0], "image_path": ["1.jpg", "2.jpg"]},
                {"prompt": "second prompt", "human_preference": [0, 1], "image_path": ["2.jpg", "3.jpg"]},
                {"prompt": "   ", "human_preference": [1, 0], "image_path": ["3.jpg", "4.jpg"]},
            ]
        ),
        encoding="utf-8",
    )


def test_materialize_hpdv2_prompts_deduplicates_and_uses_prompt_schema(tmp_path):
    source = tmp_path / "train.json"
    output = tmp_path / "train.jsonl"
    _write_hpdv2_annotation(source)

    assert recipe._materialize_hpdv2_prompts(source, output) == output
    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == [
        {"prompt": "first prompt"},
        {"prompt": "second prompt"},
    ]


def test_prepare_downloads_only_annotations_and_reuses_cached_jsonl(tmp_path, monkeypatch):
    source = tmp_path / "train.json"
    output = tmp_path / "train.jsonl"
    _write_hpdv2_annotation(source)
    calls = []

    def fake_download(full_name, include=None, data_dir=None):
        calls.append((full_name, include, data_dir))
        return str(tmp_path)

    monkeypatch.setattr(recipe.U, "hf_download_dataset", fake_download)
    args = recipe.ScriptArgs(data_dir=str(tmp_path.parent))

    assert recipe.prepare(args) == str(tmp_path)
    first_contents = output.read_text(encoding="utf-8")

    def fail_if_reparsed(*args, **kwargs):
        raise AssertionError("cached HPDv2 prompts should not be reparsed")

    monkeypatch.setattr(datasets, "load_dataset", fail_if_reparsed)
    assert recipe.prepare(args) == str(tmp_path)
    assert output.read_text(encoding="utf-8") == first_contents
    assert calls == [
        ("ymhao/HPDv2", "train.json", str(tmp_path.parent)),
        ("ymhao/HPDv2", "train.json", str(tmp_path.parent)),
    ]
