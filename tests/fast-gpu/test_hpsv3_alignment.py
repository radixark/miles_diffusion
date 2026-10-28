from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=600,
    suite="stage-b-3-gpu-h200",
    labels=[],
)

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from miles.rollout.rm_hub.hpsv3 import HPSv3Scorer

# The official HPSv3 code needs transformers 4.45, which cannot share miles' environment, so the reference
# scores come from a subprocess that runs it with its pins in a directory of their own, on miles' own torch
# and CUDA. Same GPU, same kernels: in bf16, as deployed, the port must match it bit for bit, single and
# batched. fp32 shows op-order differences between transformers 4.45 and 5 that bf16 rounds away (up to
# 1.2e-5 measured); a logic error moves mu by 1e-2 or more, so fp32 keeps a 1e-4 tolerance.
_MU_ATOL_FP32 = 1e-4
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_OFFICIAL_SCRIPT = _FIXTURES / "hpsv3_official_scores.py"
_spec = importlib.util.spec_from_file_location("hpsv3_official_scores", _OFFICIAL_SCRIPT)
_official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_official)
# tokenizers ships a compiled extension, so the directory is per interpreter and machine as well as per pins
_OVERLAY_KEY = hashlib.sha256(
    " ".join([*_official.OFFICIAL_PINS, sys.implementation.cache_tag, platform.machine()]).encode()
).hexdigest()[:12]
_OVERLAY = Path.home() / ".cache" / "miles" / f"hpsv3_official_{_OVERLAY_KEY}"


def _install_official_overlay() -> Path:
    """pip install the official pins into their own directory; the running environment is never touched."""
    if _OVERLAY.is_dir():
        return _OVERLAY
    _OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="hpsv3_official_", dir=_OVERLAY.parent))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", str(staging)]
        + _official.OFFICIAL_PINS,
        check=True,
    )
    try:
        staging.rename(_OVERLAY)
    except OSError:  # a concurrent run finished first
        shutil.rmtree(staging, ignore_errors=True)
    return _OVERLAY


@pytest.fixture(scope="module")
def official(tmp_path_factory):
    overlay = _install_official_overlay()
    output = tmp_path_factory.mktemp("hpsv3_official") / "scores.json"
    pythonpath = os.pathsep.join(filter(None, [str(overlay), os.environ.get("PYTHONPATH")]))
    subprocess.run(
        [sys.executable, str(_OFFICIAL_SCRIPT), "--output", str(output)],
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=True,
    )
    scores = json.loads(output.read_text())
    assert scores["versions"]["transformers"] == "4.45.2" and scores["attn_implementation"] == "sdpa"
    return scores


@pytest.fixture(scope="module")
def scorer(official):
    # built after the official subprocess has exited, so the two models never share the GPU
    return HPSv3Scorer(device="cuda")


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _images(official):
    return [_official.make_image(s["width"], s["height"], s["offset"]) for s in official["samples"]]


def _assert_close(actual, official, key, atol=0.0):
    expected = [s[key] for s in official["samples"]]
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=0, atol=atol, msg=key)


def test_inputs_match_official(scorer, official):
    """Prompt template, tokenizer, <|Reward|> id, resize and normalisation all reproduce the official batch."""
    for sample, image in zip(official["samples"], _images(official), strict=True):
        inputs = scorer.build_inputs([sample["prompt"]], [image])
        assert inputs["input_ids"][0].tolist() == sample["input_ids"]
        assert inputs["image_grid_thw"][0].tolist() == sample["image_grid_thw"]
        assert list(inputs["pixel_values"].shape) == sample["pixel_values_shape"]
        assert _sha256(inputs["pixel_values"]) == sample["pixel_values_sha256"]
    assert scorer.reward_token_id == official["reward_token_id"]


def _score_single_and_batched(scorer, official):
    prompts = [s["prompt"] for s in official["samples"]]
    images = _images(official)
    single = [scorer([p], [image])[0] for p, image in zip(prompts, images, strict=True)]
    # one right-padded batch of mixed image sizes, as the reward actors send it
    return single, scorer(prompts, images)


def test_bf16_scores_match_official(scorer, official):
    single, batched = _score_single_and_batched(scorer, official)
    _assert_close(single, official, "mu")
    _assert_close(batched, official, "mu_batched")


def test_fp32_scores_match_official(scorer, official):
    """Same bf16-rounded weights and rotary buffers in fp32 math."""
    scorer.model.float()
    try:
        single, batched = _score_single_and_batched(scorer, official)
    finally:
        # every value is bf16-representable, so casting back restores the shared scorer exactly
        scorer.model.to(torch.bfloat16)
    _assert_close(single, official, "mu_fp32", _MU_ATOL_FP32)
    _assert_close(batched, official, "mu_batched_fp32", _MU_ATOL_FP32)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
