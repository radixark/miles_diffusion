"""Krea2 cond plumbing: layer-stacked text embeds, position_ids rebuild, key-padding mask."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest
import torch
from safetensors.torch import save

from miles.backends.fsdp_utils.configs.krea2 import Krea2TrainPipelineConfig
from miles.utils.diffusion_rollout_response import _default_deserialize_func, _parse_cond_kwargs
from miles.utils.types import CondKwargs

CPU = torch.device("cpu")
LAYERS, DIM, IMG_LEN = 3, 5, 4


def _cond(seq_len: int, img_len: int = IMG_LEN) -> CondKwargs:
    img_rows = torch.arange(img_len * 3, dtype=torch.float32).reshape(img_len, 3)
    return CondKwargs(
        encoder_hidden_states=[torch.randn(seq_len, LAYERS, DIM)],
        encoder_attention_mask=[torch.ones(seq_len, dtype=torch.bool)],
        pos=torch.cat([torch.zeros(seq_len, 3), img_rows]),
    )


def test_prepare_shapes():
    kw = Krea2TrainPipelineConfig().prepare_cond_kwargs(_cond(7), CPU)
    assert kw["encoder_hidden_states"].shape == (1, 7, LAYERS, DIM)
    assert kw["encoder_attention_mask"].shape == (1, 7)
    assert kw["position_ids"].shape == (7 + IMG_LEN, 3)


def test_collate_pads_ragged_text_and_rebuilds_position_ids():
    cfg = Krea2TrainPipelineConfig()
    kws = [cfg.prepare_cond_kwargs(_cond(5), CPU), cfg.prepare_cond_kwargs(_cond(7), CPU)]
    out = cfg.collate_cond_for_sample_batch(kws, CPU)
    assert out["encoder_hidden_states"].shape == (2, 7, LAYERS, DIM)
    assert out["encoder_attention_mask"].tolist() == [[True] * 5 + [False] * 2, [True] * 7]
    assert torch.equal(out["position_ids"][:7], torch.zeros(7, 3))
    assert torch.equal(out["position_ids"][7:], kws[1]["position_ids"][7:])


def test_collate_uniform_lengths_drop_the_mask():
    cfg = Krea2TrainPipelineConfig()
    kws = [cfg.prepare_cond_kwargs(_cond(6), CPU) for _ in range(2)]
    assert cfg.collate_cond_for_sample_batch(kws, CPU)["encoder_attention_mask"] is None


def test_collate_honors_pad_to_len():
    cfg = Krea2TrainPipelineConfig()
    out = cfg.collate_cond_for_sample_batch([cfg.prepare_cond_kwargs(_cond(5), CPU)], CPU, pad_to_len=9)
    assert out["encoder_hidden_states"].shape == (1, 9, LAYERS, DIM)
    assert out["encoder_attention_mask"].tolist() == [[True] * 5 + [False] * 4]


def test_collate_rejects_mixed_resolutions():
    cfg = Krea2TrainPipelineConfig()
    kws = [cfg.prepare_cond_kwargs(_cond(5, img_len=4), CPU), cfg.prepare_cond_kwargs(_cond(5, img_len=6), CPU)]
    with pytest.raises(ValueError, match="one image resolution"):
        cfg.collate_cond_for_sample_batch(kws, CPU)


def test_parse_cond_kwargs_reads_krea2_env_keys():
    ser = lambda t: {"__tensor__": True, "data": save({"t": t})}  # noqa: E731
    cond = _parse_cond_kwargs(
        {
            "encoder_hidden_states": ser(torch.randn(7, LAYERS, DIM)),
            "encoder_hidden_states_mask": ser(torch.ones(7, dtype=torch.bool)),
            "pos": ser(torch.zeros(7 + IMG_LEN, 3)),
        },
        deserialize_func=_default_deserialize_func,
    )
    assert cond.encoder_hidden_states[0].shape == (7, LAYERS, DIM)
    assert cond.encoder_attention_mask[0].dtype == torch.bool
    assert cond.pos.shape == (7 + IMG_LEN, 3)
