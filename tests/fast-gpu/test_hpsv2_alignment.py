from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=180,
    suite="stage-b-3-gpu-h200",
    labels=[],
)

import numpy as np
import torch
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from huggingface_hub import hf_hub_download
from PIL import Image

from miles.rollout.rm_hub.hps import HPSScorer


def _make_image(height: int, width: int, offset: int) -> Image.Image:
    y, x = np.indices((height, width))
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = (3 * x + y + offset) % 256
    rgb[..., 1] = (x + 5 * y + 2 * offset) % 256
    rgb[..., 2] = (7 * x + 11 * y + 3 * offset) % 256
    return Image.fromarray(rgb)


def _official_hpsv2(checkpoint_path: str, prompts: list[str], images: list[Image.Image]):
    # The strict HPS checkpoint load replaces every parameter, so skip the official scorer's redundant LAION preload.
    model, _, preprocess = create_model_and_transforms(
        "ViT-H-14",
        pretrained=None,
        precision="amp",
        device="cuda",
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # The public scorer loops over pairs; batching both paths identically exercises the paired-diagonal fast path.
    image_batch = torch.stack([preprocess(image) for image in images])
    text_batch = get_tokenizer("ViT-H-14")(prompts)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        outputs = model(image_batch.cuda(), text_batch.cuda())
        scores = torch.diagonal(outputs["image_features"] @ outputs["text_features"].T)
    return image_batch, scores.float().cpu()


def test_hps_scorer_matches_official_hpsv2():
    checkpoint_path = hf_hub_download("xswu/HPSv2", "HPS_v2.1_compressed.pt")
    images = [_make_image(192, 320, 17), _make_image(320, 192, 53)]
    prompts = ["a colorful geometric landscape", "an abstract portrait with vivid lines"]

    scorer = HPSScorer(device="cuda", checkpoint_path=checkpoint_path)
    actual_images = torch.stack([scorer.preprocess(image) for image in images])
    actual_scores = torch.tensor(scorer(prompts, images))
    expected_images, expected_scores = _official_hpsv2(checkpoint_path, prompts, images)

    assert torch.equal(actual_images, expected_images)
    torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
