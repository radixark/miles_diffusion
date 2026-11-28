"""The CLAP reward on the real checkpoints: it tells sounds apart, and a clip's score is its own.

Clips are synthesized here (white noise, a 440 Hz tone), so the test needs no audio files.

Covered, for an unfused and a fusion checkpoint: each caption scores its own clip above the other (1); a clip
scores the same alone as in a batch, which the fusion extractor's random "longer" mark would break (2); a clip
longer than the 10 s window scores the same on every call, where CLAP's random crop would not (3).
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    suite="stage-b-3-gpu-h200",
    labels=[],
)

import sys

import pytest
import torch

from miles.rollout.rm_hub.clap import ClapRewardActor, ClapScorer
from miles.utils.types import GeneratedMedia

_SAMPLE_RATE = 44_100
_NOISE_CAPTION = "white noise static hiss"
_TONE_CAPTION = "a steady pure sine tone beep"


def _clips(seconds: float) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    t = torch.arange(int(_SAMPLE_RATE * seconds)) / _SAMPLE_RATE
    noise = torch.randn(2, len(t), generator=generator) * 0.2
    tone = (torch.sin(2 * torch.pi * 440 * t) * 0.5).repeat(2, 1)
    return noise, tone


@pytest.fixture(scope="module", params=["laion/larger_clap_general", "laion/clap-htsat-fused"])
def actor(request) -> ClapRewardActor:
    actor = ClapRewardActor.__new__(ClapRewardActor)
    actor.scorer = ClapScorer(device="cuda" if torch.cuda.is_available() else "cpu", model_path=request.param)
    actor.windows_per_forward = 8
    return actor


def _score(actor: ClapRewardActor, clips: list[torch.Tensor], captions: list[str]) -> list[float]:
    outputs = [GeneratedMedia(audio=clip, audio_sample_rate=_SAMPLE_RATE) for clip in clips]
    return actor.score_batch(outputs, captions)


def test_each_caption_prefers_its_own_clip(actor):
    noise, tone = _clips(seconds=4)

    noise_caption = _score(actor, [noise, tone], [_NOISE_CAPTION] * 2)
    tone_caption = _score(actor, [noise, tone], [_TONE_CAPTION] * 2)

    assert noise_caption[0] > noise_caption[1] + 0.1
    assert tone_caption[1] > tone_caption[0] + 0.1


def test_a_clip_scores_the_same_alone_and_in_a_batch(actor):
    noise, tone = _clips(seconds=4)

    batched = _score(actor, [noise, tone], [_NOISE_CAPTION, _TONE_CAPTION])
    alone = _score(actor, [noise], [_NOISE_CAPTION]) + _score(actor, [tone], [_TONE_CAPTION])

    torch.testing.assert_close(torch.tensor(batched), torch.tensor(alone), rtol=0, atol=1e-5)


def test_a_long_clip_scores_the_same_on_every_call(actor):
    noise, tone = _clips(seconds=12.5)
    long_clip = torch.cat([noise, tone], dim=1)

    first = _score(actor, [long_clip], [_NOISE_CAPTION])
    second = _score(actor, [long_clip], [_NOISE_CAPTION])

    assert first == second


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
