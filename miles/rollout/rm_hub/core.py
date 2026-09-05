"""Shared building blocks for asynchronous reward actor pools."""

from __future__ import annotations

import asyncio
import logging

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.ray.utils import COLOCATED_REWARD_GPU

logger = logging.getLogger(__name__)


def bundle_deal_order(
    bundle_indices: list[int],
    gpu_ids: list[int],
    num_gpus_per_node: int,
    num_gpus_per_engine: int,
) -> list[int]:
    """Order bundles for reward actors: those without a rollout engine's own claim first, spread across GPUs."""
    span = min(num_gpus_per_engine, num_gpus_per_node)
    covered = len(bundle_indices) // span * span

    def key(item: tuple[int, int, int]) -> tuple[int, int, int, int]:
        position, _bundle_index, gpu_id = item
        if position >= covered:
            return (0, -1, gpu_id, position)
        position_in_actor = position % span
        # an engine declares its claim on the first bundle of its span, so deal those last
        if position_in_actor == 0:
            return (1, 0, gpu_id, position)
        return (0, position_in_actor, gpu_id, position)

    slots = zip(range(len(bundle_indices)), bundle_indices, gpu_ids, strict=True)
    return [bundle_index for _, bundle_index, _ in sorted(slots, key=key)]


class ColocatedRewardSlots:
    """Deal placement-group bundles to colocated reward pools, one long-lived actor per bundle.

    Owned by ``RolloutManager`` and shared by every colocated pool.
    """

    def __init__(self, order: list[int]) -> None:
        self._order = order
        self._owners: dict[int, str] = {}
        self._pool_names: set[str] = set()

    @property
    def remaining(self) -> int:
        return len(self._order) - len(self._owners)

    def allocate(self, name: str, num_workers: int) -> list[int]:
        if name in self._pool_names:
            raise RuntimeError(f"--{name}-reward-colocate: {name} already owns reward slots")
        if num_workers > self.remaining:
            raise RuntimeError(
                f"--{name}-reward-colocate: {num_workers} slots requested, but only "
                f"{self.remaining}/{len(self._order)} remain ({self}). "
                f"Reduce --{name}-num-workers or run the pool on dedicated GPUs."
            )
        start = len(self._owners)
        slots = list(self._order[start : start + num_workers])
        self._owners.update({slot: name for slot in slots})
        self._pool_names.add(name)
        return slots

    def __str__(self) -> str:
        return ", ".join(f"bundle {bundle}: {self._owners.get(bundle)}" for bundle in sorted(self._order))


class AsyncRewardActorPool:
    """Round-robin pool for Ray reward actors exposing ``score_batch``.

    Colocated pools take one slot per rollout bundle from the manager's
    ``slots``; standalone pools are default-scheduled at ``num_gpus_per_worker``, which
    only lands on GPUs outside every placement group.
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
        name: str,
        placement_group=None,
        slots: ColocatedRewardSlots | None = None,
    ) -> None:
        if colocate:
            if placement_group is None or slots is None:
                raise RuntimeError(f"--{name}-reward-colocate: the {name} pool was not seated by RolloutManager.")
            pg, _, _ = placement_group
            strategies = [
                PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=bundle)
                for bundle in slots.allocate(name, num_workers)
            ]
            num_gpus_per_worker = COLOCATED_REWARD_GPU
        else:
            strategies = ["DEFAULT"] * num_workers

        self._actors = [
            actor_cls.options(
                num_cpus=num_gpus_per_worker,
                num_gpus=num_gpus_per_worker,
                scheduling_strategy=strategy,
            ).remote(**actor_kwargs)
            for strategy in strategies
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
