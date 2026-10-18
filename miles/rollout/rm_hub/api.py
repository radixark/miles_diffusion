"""Pluggable API reward actors for externally managed services."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from numbers import Real
from typing import Any

import torch

from miles.utils.misc import SingletonMeta, load_function
from miles.utils.types import Sample

from .core import AsyncRewardActorPool, record_reward_queue_depth


class ApiRewardActor(ABC):
    """Build, send, and parse one request per output using backend-specific hooks.

    Hooks may run concurrently on the same actor and must keep per-request state
    local. The transport owns retries and timeouts; exceptions propagate to the caller.
    """

    def score_batch(self, outputs: list[torch.Tensor], prompts: list[str]) -> list[float]:
        if len(outputs) != len(prompts):
            raise ValueError("API reward requires one prompt per output")
        scores = []
        for output, prompt in zip(outputs, prompts, strict=True):
            request = self.build_request(output, prompt)
            response = self.send_request(request)
            score = self.parse_response(response)
            if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
                raise ValueError("API reward scores must be finite numbers")
            scores.append(float(score))
        return scores

    @abstractmethod
    def build_request(self, output: torch.Tensor, prompt: str) -> Any:
        """Build a backend request from one raw CFHW tensor and its prompt."""
        raise NotImplementedError

    @abstractmethod
    def send_request(self, request: Any) -> Any:
        """Send the request through the backend's client and return its response."""
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, response: Any) -> float:
        """Extract one numeric reward and apply backend-specific validation."""
        raise NotImplementedError


class AsyncApiRewardPool(AsyncRewardActorPool, metaclass=SingletonMeta):
    """API reward pool with one zero-GPU actor handling concurrent HTTP requests."""

    name = "custom_api"
    actor_base_cls = ApiRewardActor

    def __init__(self, args) -> None:
        config = getattr(args, f"_{self.name}_rm_config")
        if config is None:
            raise ValueError(f"{self.name} reward requires --{self.name.replace('_', '-')}-rm-config.")
        actor_cls = load_function(config.actor_class)
        if not isinstance(actor_cls, type) or not issubclass(actor_cls, self.actor_base_cls):
            raise TypeError(f"API reward actor_class must be an {self.actor_base_cls.__name__} subclass")
        super().__init__(
            actor_cls=actor_cls,
            actor_kwargs=config.actor_kwargs,
            num_workers=1,
            batch_size=1,
            num_gpus_per_worker=0,
            colocate=False,
            name=self.name,
            actor_max_concurrency=config.max_concurrency,
        )


async def custom_api_rm(args, samples: Sequence[Sample], **kwargs) -> list[float]:
    pool = AsyncApiRewardPool(args)
    scores, max_queue_depth = await pool.score([s.generated_output for s in samples], [s.prompt for s in samples])
    record_reward_queue_depth(samples, "custom_api", max_queue_depth)
    return scores
