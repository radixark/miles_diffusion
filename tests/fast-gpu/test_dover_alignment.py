from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="stage-b-3-gpu-h200", labels=["rollout"])

import random

import numpy as np
import pytest
import torch
from dover import DOVER
from dover.datasets import UnifiedFrameSampler, dover_datasets, spatial_temporal_view_decomposition
from huggingface_hub import hf_hub_download

from miles.rollout.rm_hub.dover import DOVERScorer, _DOVERVideoTransform


def _make_video(frames: int, height: int, width: int) -> torch.Tensor:
    t = torch.arange(frames).view(1, -1, 1, 1)
    y = torch.arange(height).view(1, 1, -1, 1)
    x = torch.arange(width).view(1, 1, 1, -1)
    c = torch.arange(3).view(-1, 1, 1, 1)
    return ((3 * x + 5 * y + 11 * t + 17 * c) % 256).to(torch.uint8)


def _official_views(video: torch.Tensor, monkeypatch) -> dict[str, torch.Tensor]:
    # Feed the official file pipeline decoded pixels, excluding lossy codec differences.
    monkeypatch.setattr(dover_datasets, "VideoReader", lambda path: video.permute(1, 2, 3, 0))
    sample_types = {
        "technical": dict(fragments_h=7, fragments_w=7, fsize_h=32, fsize_w=32, aligned=32),
        "aesthetic": dict(size_h=224, size_w=224),
    }
    samplers = {"technical": UnifiedFrameSampler(32, 3, 2), "aesthetic": UnifiedFrameSampler(1, 32, 2)}
    np.random.seed(0)
    torch.manual_seed(0)
    views, _ = spatial_temporal_view_decomposition("reference.mp4", sample_types, samplers)
    mean = torch.tensor([123.675, 116.28, 103.53])
    std = torch.tensor([58.395, 57.12, 57.375])
    result = {}
    for name, view in views.items():
        num_clips = 3 if name == "technical" else 1
        result[name] = (
            ((view.permute(1, 2, 3, 0) - mean) / std)
            .permute(3, 0, 1, 2)
            .reshape(3, num_clips, -1, 224, 224)
            .transpose(0, 1)
        )
    return result


@pytest.mark.parametrize("shape", [(7, 224, 320), (224, 256, 224)])
def test_preprocessing_matches_official_views_and_preserves_rng(shape, monkeypatch):
    video = _make_video(*shape)
    transform = _DOVERVideoTransform()
    expected = _official_views(video, monkeypatch)
    numpy_state, torch_state, python_state = np.random.get_state(), torch.get_rng_state(), random.getstate()

    actual = transform(video.float() / 255)
    for name in expected:
        assert torch.equal(actual[name], expected[name])
        assert torch.equal(transform(video)[name], actual[name])
    assert torch.equal(torch_state, torch.get_rng_state())
    assert python_state == random.getstate()
    restored = np.random.get_state()
    assert restored[0] == numpy_state[0]
    np.testing.assert_array_equal(restored[1], numpy_state[1])
    assert restored[2:] == numpy_state[2:]


def test_dover_scores_match_official_model(monkeypatch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = hf_hub_download("teowu/DOVER", "DOVER.pth")
    scorer = DOVERScorer(device=device, checkpoint_path=checkpoint_path)
    official = DOVER(
        backbone={"technical": {"type": "swin_tiny_grpb"}, "aesthetic": {"type": "conv_tiny"}},
        backbone_preserve_keys="technical,aesthetic",
        divide_head=True,
        vqa_head={"in_channels": 768, "hidden_channels": 64},
    ).to(device)
    official.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    videos = [_make_video(7, 224, 320), _make_video(224, 256, 224)]
    expected = {name: [] for name in ("aesthetic", "technical", "overall")}
    for video in videos:
        views = _official_views(video, monkeypatch)
        with torch.no_grad():
            # The official evaluator's output order is its input dictionary's order.
            aesthetic, technical = official({name: views[name].to(device) for name in ("aesthetic", "technical")})
        a = (aesthetic.mean().item() + 0.08285) / 0.03774
        t = (technical.mean().item() - 0.1107) / 0.07355
        expected["aesthetic"].append(1 / (1 + np.exp(-a)))
        expected["technical"].append(1 / (1 + np.exp(-t)))
        expected["overall"].append(1 / (1 + np.exp(-(0.3896 * a + 0.6104 * t))))
    for name, scores in expected.items():
        scorer.score_type = name
        assert scorer(videos) == pytest.approx(scores, abs=1e-6)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
