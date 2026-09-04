"""Shared building blocks for asynchronous reward actor pools."""

from __future__ import annotations

import asyncio
import logging

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logger = logging.getLogger(__name__)


class AsyncRewardActorPool:
    """Round-robin pool for Ray reward actors exposing ``score_batch``.

    ``placement_group`` is the rollout manager's ``(pg, bundle_indices, gpu_ids)``;
    a colocated pool seats one worker per bundle on it.
    """

    def __init__(
        self,
        *,
        actor_cls,
        actor_kwargs: dict,
        num_workers: int,
        batch_size: int,
        num_gpus_per_worker: float,
        colocate: bool,
        placement_group=None,
        name: str,
    ) -> None:
        if colocate:
            if placement_group is None:
                raise RuntimeError(
                    f"--colocate-reward: the {name} pool must be seated on the rollout placement group "
                    "before the first reward call; mixed GPU reward types need dedicated reward GPUs."
                )
            pg, bundle_indices, _ = placement_group
            # bundle_indices is sorted by (node, gpu); stride so workers spread across nodes
            # instead of stacking onto the first node's GPUs.
            stride = max(1, len(bundle_indices) // num_workers)
            strategies = [
                PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_indices[w * stride],
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
        self._inflight = [0] * num_workers
        logger.info(
            "Initialized %s actor pool with %d workers, %.3f GPUs/worker, batch_size=%d.",
            name,
            num_workers,
            num_gpus_per_worker,
            batch_size,
        )

    def _next_actor_idx(self) -> int:
        i = self._round_robin_index % len(self._actors)
        self._round_robin_index += 1
        return i

    async def score(self, images: list, prompts: list[str]) -> tuple[list[float], int]:
        """Score in batches; also report the deepest dispatch-time backlog this call saw."""
        refs, idxs, max_queue_depth = [], [], 0
        for start in range(0, len(images), self._batch_size):
            end = start + self._batch_size
            i = self._next_actor_idx()
            max_queue_depth = max(max_queue_depth, self._inflight[i])
            self._inflight[i] += 1
            idxs.append(i)
            refs.append(self._actors[i].score_batch.remote(images[start:end], prompts[start:end]))

        loop = asyncio.get_running_loop()
        try:
            chunked_scores = await loop.run_in_executor(None, ray.get, refs)
        finally:
            for i in idxs:
                self._inflight[i] -= 1
        return [float(score) for chunk in chunked_scores for score in chunk], max_queue_depth
