"""Shared building blocks for asynchronous reward actor pools."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logger = logging.getLogger(__name__)


def _dispersal_order(
    bundle_indices: list[int],
    gpu_ids: list[int],
    num_gpus_per_node: int,
    num_gpus_per_engine: int,
) -> tuple[int, ...]:
    """Build a best-effort dispersed order from the existing placement-group view."""
    span = min(num_gpus_per_engine, num_gpus_per_node)
    covered = len(bundle_indices) // span * span if span > 0 else 0

    def key(item: tuple[int, int, int]) -> tuple[int, int, int, int]:
        position, _bundle_index, gpu_id = item
        if position >= covered:
            # These bundles are not occupied by a complete rollout actor span.
            return (0, -1, gpu_id, position)

        position_in_actor = position % span
        if position_in_actor == 0:
            # The rollout actor consumes its 0.25-GPU claim on the base bundle.
            return (1, 0, gpu_id, position)
        return (0, position_in_actor, gpu_id, position)

    slots = zip(range(len(bundle_indices)), bundle_indices, gpu_ids, strict=True)
    return tuple(bundle_index for _, bundle_index, _ in sorted(slots, key=key))


class ColocatedRewardSlots:
    """Own the fixed deal order for process-lifetime colocated reward pools.

    Each placement-group bundle admits one long-lived reward actor. Without a
    shared owner, overlapping 0.05-GPU claims remain pending in Ray instead of
    raising a scheduling error.
    """

    def __init__(self, order: tuple[int, ...]) -> None:
        self._order = order
        self._owners: dict[int, str] = {}
        # Keep zero-worker claims and failed pool initialization from re-entering.
        self._pool_names: set[str] = set()

    def allocate(self, name: str, num_workers: int) -> list[int]:
        if name in self._pool_names:
            raise RuntimeError(f"--colocate-reward: {name} already owns reward slots")

        start = len(self._owners)
        remaining = len(self._order) - start
        if num_workers > remaining:
            raise RuntimeError(
                f"--colocate-reward: {name} needs {num_workers} reward slots, but only "
                f"{remaining}/{len(self._order)} remain (in use: {self._usage_summary()}). "
                "Reduce *_num_workers or use dedicated reward GPUs."
            )

        slots = list(self._order[start : start + num_workers])
        self._owners.update({slot: name for slot in slots})
        self._pool_names.add(name)
        return slots

    def _usage_summary(self) -> str:
        return ", ".join(f"{name}×{count}" for name, count in Counter(self._owners.values()).items())


_manager_placement_group = None
_reward_slots: ColocatedRewardSlots | None = None


def set_manager_placement_group(pg, *, num_gpus_per_node: int, num_gpus_per_engine: int) -> None:
    """Publish the manager's placement group and initialize its reward slots."""
    global _manager_placement_group, _reward_slots
    _manager_placement_group = pg
    _, bundle_indices, gpu_ids = pg
    order = _dispersal_order(bundle_indices, gpu_ids, num_gpus_per_node, num_gpus_per_engine)
    _reward_slots = ColocatedRewardSlots(order)


def get_manager_placement_group():
    return _manager_placement_group


def _get_colocated_reward_slots() -> ColocatedRewardSlots:
    if _reward_slots is None:
        raise RuntimeError("Colocated reward pools must be created inside RolloutManager (placement group not set).")
    return _reward_slots


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
            slots = _get_colocated_reward_slots().allocate(name, num_workers)
            pg, _, _ = get_manager_placement_group()
            strategies = [
                PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=slot,
                )
                for slot in slots
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
