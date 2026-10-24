from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300,
    suite="stage-b-3-gpu-h200",
    labels=[],
)

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from miles.rollout.rm_hub.hpsv3 import HPSv3Scorer

# Recorded by tests/fixtures/gen_hpsv3_reference.py with the official package (transformers 4.45.2), which
# cannot be installed next to miles; its docstring has the environment.
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_REFERENCE = json.loads((_FIXTURES / "hpsv3_reference.json").read_text())
_spec = importlib.util.spec_from_file_location("gen_hpsv3_reference", _FIXTURES / "gen_hpsv3_reference.py")
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)

# Inputs match exactly. With the backbone cast to fp32 the port reproduces the official mu to ~1e-4, so
# the fp32 check is the strict one; the bf16 path differs only by kernel noise across torch/transformers
# releases (up to ~0.1 here), about as large as the official model's own single-vs-batched spread (0.13).
_MU_ATOL_FP32 = 1e-3
_MU_ATOL_BF16 = 0.2


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


@pytest.fixture(scope="module")
def scorer():
    return HPSv3Scorer(device="cuda", attn_implementation="sdpa")


def _samples():
    for sample in _REFERENCE["samples"]:
        yield sample, _gen.make_image(sample["width"], sample["height"], sample["offset"])


def test_inputs_match_official(scorer):
    """Prompt template, tokenizer, <|Reward|> id, resize and normalisation all reproduce the official batch."""
    for sample, image in _samples():
        inputs = scorer.build_inputs([sample["prompt"]], [image])
        assert inputs["input_ids"][0].tolist() == sample["input_ids"]
        assert inputs["image_grid_thw"][0].tolist() == sample["image_grid_thw"]
        assert list(inputs["pixel_values"].shape) == sample["pixel_values_shape"]
        assert _sha256(inputs["pixel_values"]) == sample["pixel_values_sha256"]
    assert scorer.reward_token_id == _REFERENCE["runtime"]["reward_token_id"]


def _assert_mu_close(actual, samples, key, atol):
    torch.testing.assert_close(torch.tensor(actual), torch.tensor([s[key] for s in samples]), rtol=0, atol=atol)


def test_bf16_scores_match_official(scorer):
    samples, images = zip(*_samples(), strict=True)
    single = [scorer([s["prompt"]], [image])[0] for s, image in zip(samples, images, strict=True)]
    # one right-padded batch of mixed image sizes, as the reward actors send it
    batched = scorer([s["prompt"] for s in samples], list(images))

    _assert_mu_close(single, samples, "mu", _MU_ATOL_BF16)
    _assert_mu_close(batched, samples, "mu_batched", _MU_ATOL_BF16)
    assert sorted(range(len(samples)), key=lambda i: single[i]) == sorted(
        range(len(samples)), key=lambda i: samples[i]["mu"]
    )


def test_fp32_scores_match_official(scorer):
    """Same bf16-rounded weights and rotary buffers in fp32 math: a wrong layer, mask or rotary shows up here.

    Runs last in the module: it casts the shared scorer's backbone to fp32.
    """
    scorer.model.float()
    samples, images = zip(*_samples(), strict=True)

    _assert_mu_close(
        [scorer([s["prompt"]], [image])[0] for s, image in zip(samples, images, strict=True)],
        samples,
        "mu_fp32",
        _MU_ATOL_FP32,
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
