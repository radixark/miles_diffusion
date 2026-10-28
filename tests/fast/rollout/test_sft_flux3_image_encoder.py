"""FLUX.3 Image encode contract: an encode-only server's extras become one trainer-ready pair."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.backends.fsdp_utils.configs.flux3_image import Flux3ImageTrainPipelineConfig
from miles.rollout.encoder_hub import flux3_image
from miles.utils.types import Sample


def _encoded(branches: int = 1) -> dict:
    text = (torch.randn(80, 8, dtype=torch.bfloat16), torch.zeros(80, 4, dtype=torch.long))
    target = {"latents": torch.randn(16, 4), "latent_ids": torch.zeros(16, 4)}
    return {"flux3_text": [text] * branches, "encode_target": {"visual": target}}


def test_pair_feeds_one_batched_forward():
    pair = flux3_image.pair_from_encoded(_encoded(), Sample(prompt="p"))
    cond = Flux3ImageTrainPipelineConfig().collate_cond_for_sample_batch([pair["cond_kwargs"]], torch.device("cpu"))
    image = torch.stack([pair["latent"]["visual"]])
    assert cond["encoder_hidden_states"].shape[:2] == cond["flux3_txt_4d_rope_pos_ids"].shape[:2] == (1, 80)
    assert cond["flux3_img_4d_rope_pos_ids"].shape[:2] == image.shape[:2] == (1, 16)


def test_cfg_pair_is_rejected_not_truncated():
    # A server without --disagg-encode-only returns [negative, positive]; either half alone trains the wrong prompt.
    with pytest.raises(ValueError):
        flux3_image.pair_from_encoded(_encoded(branches=2), Sample(prompt="p"))
