"""A reward pool sends each scorer only the GeneratedMedia fields its class declares in ``required_media``.

Mental model (one audio-video sample, two pools):

    sample.generated_media = GeneratedMedia(visual, audio, audio_sample_rate)
        -> pool of a {"visual"} scorer : visual + sample rate, audio None
        -> pool of an {"audio"} scorer : audio + sample rate,  visual None

Covered: a scorer class whose ``required_media`` is missing, empty or names an unknown field is rejected when its
pool is built (1); a sample without a required field is rejected before anything is sent (2); each pool sends only
the fields its scorer requires (3).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from unittest.mock import Mock

import pytest
import torch

import miles.rollout.rm_hub.core as core_module
from miles.rollout.rm_hub.core import AsyncRewardActorPool
from miles.utils.types import GeneratedMedia, Sample

_VIDEO = torch.zeros(3, 2, 4, 4)
_WAVEFORM = torch.zeros(2, 16)


@pytest.fixture
def actor(monkeypatch):
    """The pool's one actor: a mock in place of a Ray actor, so a test can read what ``score_batch`` was sent."""
    actor = Mock()
    actor.score_batch.remote.side_effect = lambda outputs, prompts: [0.0] * len(outputs)
    ray_actor_cls = Mock()
    ray_actor_cls.options.return_value.remote.return_value = actor
    monkeypatch.setattr(core_module.ray, "remote", lambda scorer_cls: ray_actor_cls)
    monkeypatch.setattr(core_module.ray, "get", lambda refs: refs)
    return actor


def _scorer(required_media):
    return type("Scorer", (), {"required_media": required_media})


def _pool(scorer_cls, name="test"):
    return AsyncRewardActorPool(
        args=Namespace(use_output_store=False),
        actor_cls=scorer_cls,
        actor_kwargs={},
        num_workers=1,
        batch_size=8,
        num_gpus_per_worker=0,
        colocate=False,
        name=name,
    )


def _sent_output(actor) -> GeneratedMedia:
    (outputs, _prompts), _ = actor.score_batch.remote.call_args
    (output,) = outputs
    return output


def test_scorer_without_required_media_is_rejected_when_its_pool_is_built(actor):
    """A scorer that did not say what it scores must fail when its pool is built, not in the middle of a rollout."""
    with pytest.raises(AttributeError, match="required_media"):
        _pool(type("Scorer", (), {}))


@pytest.mark.parametrize("required_media", [frozenset(), frozenset({"video"})], ids=["empty", "unknown_field"])
def test_required_media_must_name_visual_or_audio(actor, required_media):
    """Either would have the pool send the scorer nothing, and the scorer crash on a None it cannot explain."""
    with pytest.raises(TypeError, match="must name visual, audio or both"):
        _pool(_scorer(required_media))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("required_media", "output", "match"),
    [
        (frozenset({"audio"}), GeneratedMedia(visual=_VIDEO), "the sound reward requires audio"),
        (frozenset({"visual"}), GeneratedMedia(audio=_WAVEFORM, audio_sample_rate=16), "requires visual"),
    ],
    ids=["no_audio", "no_visual"],
)
async def test_sample_without_a_required_field_is_rejected_before_sending(actor, required_media, output, match):
    """A scorer must not receive a None it would crash on deep inside its actor."""
    with pytest.raises(ValueError, match=match):
        await _pool(_scorer(required_media), name="sound").score([output], ["p"])
    actor.score_batch.remote.assert_not_called()


@pytest.mark.asyncio
async def test_each_pool_sends_only_the_fields_its_scorer_requires(actor):
    """An image reward must not move (or read from the store) a waveform it ignores, nor an audio reward the video."""
    sample = Sample(prompt="p", generated_output=_VIDEO, generated_audio=_WAVEFORM, audio_sample_rate=16)

    await _pool(_scorer(frozenset({"visual"}))).score([sample.generated_media], [sample.prompt])
    sent_to_image_reward = _sent_output(actor)
    await _pool(_scorer(frozenset({"audio"}))).score([sample.generated_media], [sample.prompt])
    sent_to_audio_reward = _sent_output(actor)

    assert sent_to_image_reward.visual is _VIDEO and sent_to_image_reward.audio is None
    assert sent_to_audio_reward.audio is _WAVEFORM and sent_to_audio_reward.visual is None
    # the sample rate is a scalar, sent either way
    assert sent_to_image_reward.audio_sample_rate == sent_to_audio_reward.audio_sample_rate == 16
