"""CLAP reward: text-audio similarity for rollouts that return a waveform (LTX audio-video, H3 t2va).

The score is the cosine similarity between CLAP's prompt and audio embeddings, the usual "CLAP score". It is
the audio counterpart of PickScore: the video rewards only see frames, so with them alone the audio stream's
advantage comes from the picture.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import ray
import torch
import torchaudio

from miles.utils.misc import SingletonMeta
from miles.utils.types import GeneratedMedia, Sample

from .core import AsyncRewardActorPool, record_reward_queue_depth

# CLAP's text tower was trained on captions cut at 77 tokens (laion_clap's tokenizer call).
_MAX_TEXT_TOKENS = 77


def waveform_to_mono(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    """A ``[C, L]`` (or ``[L]``) waveform as mono float32 ``[L']`` at ``target_sample_rate``."""
    if waveform.ndim not in (1, 2):
        raise ValueError(f"a waveform is [C, L] or [L], got shape {tuple(waveform.shape)}")
    mono = waveform.float().reshape(-1, waveform.shape[-1]).mean(dim=0)
    if sample_rate != target_sample_rate:
        mono = torchaudio.functional.resample(mono, sample_rate, target_sample_rate)
    return mono


def split_windows(mono: torch.Tensor, window_samples: int) -> list[torch.Tensor]:
    """The clip itself when it fits one model window, else evenly spaced full windows that cover it.

    CLAP's own handling of a longer clip is a random crop, which would make the reward of a sample depend on
    the draw; every window given to the model here fits, so nothing is cropped.
    """
    length = mono.shape[0]
    if length <= window_samples:
        return [mono]
    num_windows = math.ceil(length / window_samples)
    starts = np.linspace(0, length - window_samples, num_windows).round().astype(int)
    return [mono[start : start + window_samples] for start in starts]


class ClapScorer(torch.nn.Module):
    """CLAP cosine similarity for (prompt, waveform window) pairs, in [-1, 1]."""

    def __init__(self, *, device: str = "cuda", model_path: str) -> None:
        super().__init__()
        from transformers import ClapModel, ClapProcessor

        self.device = torch.device(device)
        processor = ClapProcessor.from_pretrained(model_path)
        self.tokenizer = processor.tokenizer
        self.feature_extractor = processor.feature_extractor
        self.model = ClapModel.from_pretrained(model_path).eval().to(device=self.device, dtype=torch.float32)
        self.sample_rate: int = self.feature_extractor.sampling_rate
        self.window_samples: int = self.feature_extractor.nb_max_samples

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], windows: Sequence[torch.Tensor]) -> list[float]:
        """``windows`` are mono waveforms at ``sample_rate``, each at most ``window_samples`` long."""
        audio_inputs = self.feature_extractor(
            [window.numpy() for window in windows], sampling_rate=self.sample_rate, return_tensors="pt"
        )
        if "is_longer" in audio_inputs:
            # A fusion checkpoint's extractor marks one random clip of a batch of short clips as longer, and the
            # model then fuses that clip's (identical) crops; no window here is longer, so none is marked.
            audio_inputs["is_longer"] = torch.zeros_like(audio_inputs["is_longer"])
        text_inputs = self.tokenizer(
            list(prompts), padding=True, truncation=True, max_length=_MAX_TEXT_TOKENS, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in (audio_inputs | text_inputs).items()}
        outputs = self.model(**inputs)
        # both come out of the model L2-normalized
        scores = (outputs.text_embeds * outputs.audio_embeds).sum(dim=-1)
        return [float(score) for score in scores.float().cpu()]


class ClapRewardActor:
    required_media = frozenset({"audio"})

    def __init__(self, *, model_path: str, windows_per_forward: int) -> None:
        self.windows_per_forward = windows_per_forward
        use_cuda = bool(ray.get_gpu_ids()) and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        self.scorer = ClapScorer(device="cuda" if use_cuda else "cpu", model_path=model_path)

    def score_batch(self, outputs: list[GeneratedMedia], prompts: list[str]) -> list[float]:
        # a clip longer than the model's window is scored on windows that cover it, averaged
        windows, window_counts = [], []
        for output in outputs:
            mono = waveform_to_mono(output.audio, output.audio_sample_rate, self.scorer.sample_rate)
            clip_windows = split_windows(mono, self.scorer.window_samples)
            windows.extend(clip_windows)
            window_counts.append(len(clip_windows))
        flat_prompts = [p for p, n in zip(prompts, window_counts, strict=True) for _ in range(n)]
        flat_scores = []
        for start in range(0, len(windows), self.windows_per_forward):
            end = start + self.windows_per_forward
            flat_scores.extend(self.scorer(flat_prompts[start:end], windows[start:end]))
        scores, offset = [], 0
        for count in window_counts:
            scores.append(float(sum(flat_scores[offset : offset + count]) / count))
            offset += count
        return scores


class AsyncClapPool(AsyncRewardActorPool, metaclass=SingletonMeta):
    """Ray actor pool for CLAP reward inference."""

    def __init__(self, args, placement_group=None, slots=None) -> None:
        super().__init__(
            args=args,
            actor_cls=ClapRewardActor,
            actor_kwargs={"model_path": args.clap_model_path, "windows_per_forward": args.clap_batch_size},
            num_workers=args.clap_num_workers,
            batch_size=args.clap_batch_size,
            num_gpus_per_worker=args.clap_num_gpus_per_worker,
            colocate=args.clap_reward_colocate,
            name="clap",
            placement_group=placement_group,
            slots=slots,
        )


async def clap_rm(args, samples: Sequence[Sample]) -> list[float]:
    pool = AsyncClapPool(args)
    # the pool refuses a sample without a waveform, so a model that returns no audio fails here
    scores, max_queue_depth = await pool.score([s.generated_media for s in samples], [s.prompt for s in samples])
    record_reward_queue_depth(samples, "clap", max_queue_depth)
    return scores
