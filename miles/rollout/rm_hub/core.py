"""Shared building blocks for asynchronous reward actor pools."""

from __future__ import annotations

import asyncio
import logging

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

_manager_placement_group = None
logger = logging.getLogger(__name__)


def set_manager_placement_group(pg) -> None:
    """Publish the manager's (pg, bundle_indices, gpu_ids) for colocated actor pools."""
    global _manager_placement_group
    _manager_placement_group = pg


def get_manager_placement_group():
    return _manager_placement_group


set_reward_placement_group = set_manager_placement_group
get_reward_placement_group = get_manager_placement_group


class AsyncRewardActorPool:
    """Round-robin pool for Ray reward actors exposing ``score_batch``."""

    def __init__(
        self,
        *,
        actor_cls,
        actor_kwargs: dict,
        num_workers: int,
        batch_size: int,
        num_gpus_per_worker: float,
        colocate: bool,
        name: str,
    ) -> None:
        if colocate:
            pg, bundle_indices, _ = get_reward_placement_group()
            strategies = [
                PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_indices[w],
                )
                for w in range(num_workers)
            ]
            num_gpus_per_worker = 0.05
            num_cpus_per_worker = 0.05
        else:
            strategies = ["DEFAULT"] * num_workers
            num_cpus_per_worker = 1

        self._actors = [
            actor_cls.options(
                num_cpus=num_cpus_per_worker,
                num_gpus=num_gpus_per_worker,
                scheduling_strategy=strategies[i],
            ).remote(**actor_kwargs)
            for i in range(num_workers)
        ]
        self._batch_size = batch_size
        self._round_robin_index = 0
        logger.info(
            "Initialized %s actor pool with %d workers, %.3f GPUs/worker, batch_size=%d.",
            name,
            num_workers,
            num_gpus_per_worker,
            batch_size,
        )

    def _next_actor(self):
        actor = self._actors[self._round_robin_index % len(self._actors)]
        self._round_robin_index += 1
        return actor

    async def score(self, images: list, prompts: list[str]) -> list[float]:
        refs = []
        for start in range(0, len(images), self._batch_size):
            end = start + self._batch_size
            refs.append(self._next_actor().score_batch.remote(images[start:end], prompts[start:end]))

        loop = asyncio.get_running_loop()
        chunked_scores = await loop.run_in_executor(None, ray.get, refs)
        return [float(score) for chunk in chunked_scores for score in chunk]
