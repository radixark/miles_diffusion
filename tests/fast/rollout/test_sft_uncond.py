"""SFT uncond cond under --sft-distilled-cfg-scale: shared by unconditioned pairs, carried by pairs with references.

    SFT_CACHE_KEY_ARGS ──sha256──► .sft_cache/uncond-<sha>.pt   (encoded by encode_uncond on the first miss)
                                        │ torch.load once per rollout, only if some pair lacks its own
                                        ▼
    every pair without "uncond_cond_kwargs" gets that one dict   (scale 1.0: no key, nothing encoded)
    a pair with references (Ref2VA) keeps the uncond its encode_sample attached

What each test pins: the key follows the encode args; generate_rollout encodes the shared uncond once across
rollouts and hands every unconditioned pair the same object; pairs carrying their own uncond keep it and trigger no
shared encode; scale 1.0 leaves pairs exactly as cached.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.rollout import sft_rollout
from miles.rollout.sft_rollout import generate_rollout, sft_sample_key, sft_uncond_key
from miles.utils.types import Sample


def _args(tmp_path, **overrides):
    base = dict(
        diffusion_model_family="h3",
        sft_encoder_checkpoint="ckpt",
        diffusion_height=768,
        diffusion_width=1344,
        diffusion_output_num_frames=107,
        sft_frame_stride=1,
        diffusion_task="t2va",
        seed=1,
        sft_distilled_cfg_scale=4.0,
        prompt_data=str(tmp_path / "data.jsonl"),
        rollout_batch_size=4,
    )
    base.update(overrides)
    return Namespace(**base)


def test_uncond_key_follows_the_encode_args(tmp_path):
    key = sft_uncond_key(_args(tmp_path))
    assert key.startswith("uncond-") and key.endswith(".pt")
    assert key == sft_uncond_key(_args(tmp_path))
    assert sft_uncond_key(_args(tmp_path, diffusion_output_num_frames=124)) != key
    assert sft_uncond_key(_args(tmp_path, sft_encoder_checkpoint="other")) != key
    # The scale decides how the uncond is used, not what it is.
    assert sft_uncond_key(_args(tmp_path, sft_distilled_cfg_scale=2.0)) == key


class _DataSource:
    def __init__(self, samples):
        self.epoch_id = 0
        self.dataset = SimpleNamespace(shuffle=lambda epoch_id: None)
        self._samples = samples

    def get_samples(self, num_samples):
        return [[sample] for sample in self._samples[:num_samples]]


class _Pool:
    """Stands in for the encode actors; ray.get is patched to pass the "remote" result straight through."""

    encodes = 0

    def __init__(self, args):
        self.actors = [SimpleNamespace(encode_uncond=SimpleNamespace(remote=self._encode_uncond))]

    def wait_for_train_step(self):
        pass

    def _encode_uncond(self):
        _Pool.encodes += 1
        return {"encoder_hidden_states": torch.zeros(1, 1, 4)}


@pytest.fixture
def pool(monkeypatch):
    _Pool.encodes = 0
    monkeypatch.setattr(sft_rollout, "SftEncodePool", _Pool)
    monkeypatch.setattr(sft_rollout.ray, "get", lambda ref: ref)
    return _Pool


def _cached_samples(args, tmp_path, count, with_references=False):
    cache_dir = tmp_path / ".sft_cache"
    cache_dir.mkdir()
    samples = []
    for i in range(count):
        video = tmp_path / f"{i}.mp4"
        video.write_bytes(b"x" * (i + 1))
        sample = Sample(prompt=f"p{i}", target={"visual": str(video)})
        pair = {
            "latent": {"visual": torch.full((2,), float(i))},
            "cond_kwargs": {"encoder_hidden_states": torch.ones(1, 3, 4)},
        }
        if with_references:
            reference = tmp_path / f"{i}.png"
            reference.write_bytes(b"y" * (i + 1))
            sample.conditions = [{"type": "image", "uri": str(reference)}]
            pair["uncond_cond_kwargs"] = {"encoder_hidden_states": torch.full((1, 1, 4), float(i))}
        torch.save(pair, cache_dir / sft_sample_key(args, sample)[0])
        samples.append(sample)
    return samples


def test_every_pair_shares_one_uncond_encoded_once(tmp_path, pool):
    args = _args(tmp_path)
    samples = _cached_samples(args, tmp_path, 4)

    out = generate_rollout(args, 0, _DataSource(samples))

    unconds = [group[0].train_metadata["sft_pair"]["uncond_cond_kwargs"] for group in out.samples]
    assert all(uncond is unconds[0] for uncond in unconds)
    assert torch.equal(unconds[0]["encoder_hidden_states"], torch.zeros(1, 1, 4))
    generate_rollout(args, 1, _DataSource(samples))
    assert pool.encodes == 1


def test_pairs_with_references_keep_their_own_uncond(tmp_path, pool):
    args = _args(tmp_path, diffusion_task="ref2va")
    samples = _cached_samples(args, tmp_path, 3, with_references=True)

    out = generate_rollout(args, 0, _DataSource(samples))

    assert pool.encodes == 0
    for i, group in enumerate(out.samples):
        uncond = group[0].train_metadata["sft_pair"]["uncond_cond_kwargs"]
        assert torch.equal(uncond["encoder_hidden_states"], torch.full((1, 1, 4), float(i)))


def test_scale_one_leaves_pairs_as_cached(tmp_path, pool):
    args = _args(tmp_path, sft_distilled_cfg_scale=1.0)
    samples = _cached_samples(args, tmp_path, 2)

    out = generate_rollout(args, 0, _DataSource(samples))

    assert pool.encodes == 0
    assert all("uncond_cond_kwargs" not in group[0].train_metadata["sft_pair"] for group in out.samples)
