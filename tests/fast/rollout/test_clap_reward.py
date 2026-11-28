"""CLAP reward plumbing that does not need the checkpoint; the real model is tests/fast-gpu/test_clap_scorer.py.

Mental model (one audio-video sample, 48 kHz model, 10 s window):

    sample.generated_media (audio [C, L] at audio_sample_rate)        a StoredTensor with the output store on
        -> clap_rm                      : one GeneratedMedia + prompt per sample to the pool
        -> AsyncRewardActorPool         : requires audio, sends no video
        -> ClapRewardActor.score_batch  : mono -> 48 kHz -> windows that cover the clip -> mean over windows
        -> ClapScorer                   : cosine(text embedding, audio embedding)

Covered: channels are averaged and the clip resampled to the model's rate (1); a clip that fits the window is
scored whole, a longer one on evenly spaced full windows from its start to its end (2); the actor scores each
window with its clip's prompt at the clip's own sample rate and averages them per clip (3); the scorer marks no
window as longer and cuts the prompt at 77 tokens (4); clap_rm hands each sample's media and prompt to its pool
and records its backlog (5); the CLAP pool refuses a sample without a waveform and sends the actor no video (6);
batched_async_rm takes the clap fast path (7); weighted_mixture_rm accepts clap (8); an audio-video command line
parses with the CLAP defaults (9), and colocated CLAP workers are checked like the other pools' (10).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import shlex
import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

import miles.rollout.rm_hub.clap as clap_module
import miles.rollout.rm_hub.core as core_module
from miles.rollout.rm_hub import batched_async_rm
from miles.rollout.rm_hub.clap import ClapRewardActor, ClapScorer, split_windows, waveform_to_mono
from miles.rollout.rm_hub.core import AsyncRewardActorPool
from miles.rollout.rm_hub.weighted_mixture_rm import parse_weights
from miles.utils.types import GeneratedMedia, Sample

_STEREO = torch.stack([torch.ones(24_000), -torch.ones(24_000) * 0.5])


def _actor_with(scorer, *, windows_per_forward: int = 8) -> ClapRewardActor:
    actor = ClapRewardActor.__new__(ClapRewardActor)
    actor.scorer = scorer
    actor.windows_per_forward = windows_per_forward
    return actor


def test_channels_are_averaged_and_the_clip_resampled():
    """CLAP takes one channel at its own rate; a clip fed at another rate is heard at the wrong pitch."""
    mono = waveform_to_mono(_STEREO, 24_000, 48_000)

    assert mono.shape == (48_000,)
    # a constant signal stays constant away from the resampling filter's edges
    assert torch.allclose(mono[1_000:-1_000], torch.full((46_000,), 0.25), atol=1e-3)
    assert torch.equal(waveform_to_mono(_STEREO[0], 48_000, 48_000), _STEREO[0])
    with pytest.raises(ValueError, match=r"\[C, L\] or \[L\]"):
        waveform_to_mono(torch.zeros(1, 2, 8), 48_000, 48_000)


def test_a_long_clip_is_covered_by_full_windows():
    """CLAP crops a longer clip at random, so its reward would change from call to call."""
    short = torch.arange(7.0)
    long = torch.arange(25.0)

    assert [w.tolist() for w in split_windows(short, 10)] == [short.tolist()]
    assert [w.tolist() for w in split_windows(torch.arange(10.0), 10)] == [list(range(10))]
    windows = split_windows(long, 10)
    assert [(int(w[0]), int(w[-1])) for w in windows] == [(0, 9), (8, 17), (15, 24)]


def test_actor_scores_each_window_with_its_prompt_and_averages_per_clip():
    """A clip of several windows must get one reward, and forwards stay within --clap-batch-size."""
    forwards = []

    def scorer(prompts, windows):
        forwards.append([(prompt, len(window)) for prompt, window in zip(prompts, windows, strict=True)])
        return [float(window[0]) for window in windows]

    scorer.sample_rate, scorer.window_samples = 10, 10
    actor = _actor_with(scorer, windows_per_forward=2)
    # each clip is heard at its own rate: 3 samples at 5 Hz are 6 at the model's 10 Hz
    half_rate = GeneratedMedia(audio=torch.zeros(1, 3), audio_sample_rate=5)
    three_windows = GeneratedMedia(audio=torch.arange(25.0).reshape(1, 25), audio_sample_rate=10)

    scores = actor.score_batch([half_rate, three_windows], ["rain", "dog"])

    # the long clip's windows start at samples 0, 8 and 15
    assert scores == [0.0, pytest.approx((0 + 8 + 15) / 3)]
    assert forwards == [[("rain", 6), ("dog", 10)], [("dog", 10), ("dog", 10)]]


def test_scorer_marks_no_window_as_longer_and_cuts_the_prompt():
    """A fusion checkpoint's extractor marks a random clip of the batch as longer, which changes its embedding."""
    calls = {}

    def feature_extractor(waveforms, *, sampling_rate, return_tensors):
        calls["audio"] = (len(waveforms), sampling_rate)
        return {"input_features": torch.zeros(2, 1), "is_longer": torch.tensor([[False], [True]])}

    def tokenizer(prompts, **kwargs):
        calls["text"] = kwargs
        return {"input_ids": torch.zeros(2, 1, dtype=torch.long)}

    def model(**inputs):
        calls["is_longer"] = inputs["is_longer"]
        text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        audio = torch.tensor([[0.6, 0.8], [0.0, 1.0]])
        return SimpleNamespace(text_embeds=text, audio_embeds=audio)

    scorer = ClapScorer.__new__(ClapScorer)
    torch.nn.Module.__init__(scorer)
    scorer.device, scorer.sample_rate = torch.device("cpu"), 48_000
    scorer.feature_extractor, scorer.tokenizer, scorer.model = feature_extractor, tokenizer, model

    scores = scorer(["rain", "dog"], [torch.zeros(4), torch.zeros(4)])

    assert scores == pytest.approx([0.6, 1.0])
    assert not calls["is_longer"].any()
    assert calls["audio"] == (2, 48_000)
    assert calls["text"]["truncation"] is True and calls["text"]["max_length"] == 77


@pytest.mark.asyncio
async def test_clap_rm_hands_each_samples_media_and_prompt_to_its_pool(monkeypatch):
    pool = AsyncMock()
    pool.score.return_value = ([0.4], 3)
    monkeypatch.setattr(clap_module, "AsyncClapPool", lambda args: pool)
    sample = Sample(prompt="rain on a tin roof", generated_audio=_STEREO, audio_sample_rate=24_000)

    assert await clap_module.clap_rm(Namespace(), [sample]) == [0.4]
    ((output,), prompts), _ = pool.score.await_args
    assert output.audio is _STEREO and output.audio_sample_rate == 24_000
    assert prompts == ["rain on a tin roof"]
    assert sample.reward_max_queue_depth == {"clap": 3.0}


@pytest.mark.asyncio
async def test_the_clap_pool_refuses_silence_and_sends_no_video(monkeypatch):
    """Scoring a sample without audio would train on a constant; sending the video would move it for nothing."""
    ray_actor = Mock()
    ray_actor.score_batch.remote.side_effect = lambda outputs, prompts: [0.0] * len(outputs)
    ray_actor_cls = Mock()
    ray_actor_cls.options.return_value.remote.return_value = ray_actor
    monkeypatch.setattr(core_module.ray, "remote", lambda actor_cls: ray_actor_cls)
    monkeypatch.setattr(core_module.ray, "get", lambda refs: refs)
    pool = AsyncRewardActorPool(
        args=Namespace(use_output_store=False),
        actor_cls=ClapRewardActor,
        actor_kwargs={},
        num_workers=1,
        batch_size=8,
        num_gpus_per_worker=0,
        colocate=False,
        name="clap",
    )
    video = torch.zeros(3, 2, 4, 4)

    with pytest.raises(ValueError, match="the clap reward requires audio"):
        await pool.score([Sample(generated_output=video).generated_media], ["a"])
    ray_actor.score_batch.remote.assert_not_called()

    audio_video = Sample(generated_output=video, generated_audio=_STEREO, audio_sample_rate=24_000)
    await pool.score([audio_video.generated_media], ["a"])
    ((sent,), _), _ = ray_actor.score_batch.remote.call_args
    assert sent.visual is None and sent.audio is _STEREO and sent.audio_sample_rate == 24_000


@pytest.mark.asyncio
async def test_batched_async_rm_uses_clap_batch_fast_path(monkeypatch):
    clap_rm = AsyncMock(return_value=[0.3, 0.1])
    monkeypatch.setattr(clap_module, "clap_rm", clap_rm)
    args = Namespace(custom_rm_path=None, rm_type="clap")
    samples = [Sample(index=3), Sample(index=1)]

    assert await batched_async_rm(args, samples) == [0.3, 0.1]
    clap_rm.assert_awaited_once_with(args, samples)


def test_weighted_mixture_accepts_clap():
    assert parse_weights("pickscore=0.5,clap=0.5") == [("pickscore", 0.5), ("clap", 0.5)]


def _parse(monkeypatch, reward_args: str) -> Namespace:
    from miles.backends.fsdp_utils.arguments import load_fsdp_args
    from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args, set_default_diffusion_args

    argv = (
        "test --hf-checkpoint Lightricks/LTX-2.3 --diffusion-task t2v --prompt-data prompts.jsonl "
        "--rollout-batch-size 2 --n-samples-per-prompt 2 --global-batch-size 4 --num-rollout 1 "
        "--actor-num-gpus-per-node 2 --rollout-num-gpus 2 --rollout-num-gpus-per-engine 1 "
        "--micro-batch-size 1 --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        "--diffusion-sde-type cps --diffusion-guidance-scale 1.0 "
        f"--rollout-extra-streams audio --rollout-patch-group ltx_av --rm-type clap {reward_args}"
    )
    monkeypatch.setattr(sys, "argv", shlex.split(argv))
    args = load_fsdp_args(extra_args_provider=get_miles_extra_args_provider())
    set_default_diffusion_args(args)
    miles_validate_args(args)
    return args


def test_an_audio_video_command_line_parses_with_the_clap_defaults(monkeypatch):
    args = _parse(monkeypatch, "")

    assert args.clap_model_path == "laion/larger_clap_general"
    assert (args.clap_num_workers, args.clap_batch_size, args.clap_reward_colocate) == (1, 8, False)


@pytest.mark.parametrize(
    "reward_args, message",
    [
        ("--clap-num-workers 0", "--clap-num-workers must be positive"),
        ("--clap-reward-colocate", "--clap-reward-colocate requires --colocate"),
        ("--colocate --clap-reward-colocate --clap-num-workers 3", r"workers \(3\) exceed rollout_num_gpus \(2\)"),
    ],
)
def test_clap_workers_are_checked_like_the_other_pools(monkeypatch, reward_args, message):
    """A colocated pool the slot check does not count would take a GPU slot another reward was given."""
    with pytest.raises(ValueError, match=message):
        _parse(monkeypatch, reward_args)
