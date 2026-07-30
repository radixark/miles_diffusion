import logging
from collections import OrderedDict
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def stack_train_pair_rollout_debug(
    batch: list[dict],
    key: str,
) -> torch.Tensor | None:
    """Stack one rollout debug field across a train micro-batch."""
    if not batch:
        return None
    for item in batch:
        rollout_debug_tensors = item.get("rollout_debug_tensors")
        if not isinstance(rollout_debug_tensors, dict) or rollout_debug_tensors.get(key) is None:
            return None
    return torch.stack([item["rollout_debug_tensors"][key] for item in batch], dim=0)


def scheduler_meta_from_rollout(
    rollout_data: dict,
    *,
    device: torch.device,
    num_train_timesteps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use rollout-side scheduler metadata for train/rollout alignment."""
    if "scheduler_timesteps" not in rollout_data:
        raise ValueError("rollout_data missing scheduler_timesteps")
    timesteps = rollout_data["scheduler_timesteps"].to(device=device, dtype=torch.float32)
    if "scheduler_sigmas" in rollout_data:
        sigmas = rollout_data["scheduler_sigmas"].to(device=device, dtype=torch.float32)
    else:
        sigmas = torch.cat([timesteps / float(num_train_timesteps), timesteps.new_zeros(1)])
    return timesteps, sigmas


class TrainDataDPSplitter:
    """Split flat train-pair payloads across DP ranks.

    Two policies (``mode``):

    - ``contiguous`` (default): rank r gets one contiguous block of train pairs
      ``[r*pairs_per_rank : (r+1)*pairs_per_rank]``. Since pairs are sample-major,
      this gives rank r a contiguous block of samples.
    - ``baseline_stride``: reproduces the legacy TrainRayActor dispatch, which
      partitioned *samples* by stride ``range(r, num_samples, dp_size)`` and then
      tiled them. Used by the ``verify/baseline-batch-parity`` check to confirm the
      refactored rollout-side dispatch can feed each DP rank the exact same
      sample set (in the same order) as the old code path, so any residual
      train-curve difference is attributable to grouping policy, not a bug.
    """

    def split_by_dp(
        self,
        data: dict[str, Any],
        dp_size: int,
        mode: str = "contiguous",
    ) -> list[dict[str, list[dict[str, Any]]]]:
        """Split train data across DP ranks into equal-sized shards."""
        if dp_size <= 0:
            raise ValueError(f"dp_size must be positive, got {dp_size}")
        if mode not in ("contiguous", "baseline_stride"):
            raise ValueError(f"unknown dp split mode {mode!r}")
        train_data = data["train_data"]
        scheduler_timesteps = data.get("scheduler_timesteps")
        scheduler_sigmas = data.get("scheduler_sigmas")
        num_pairs = len(train_data)
        if num_pairs < dp_size:
            raise ValueError(
                f"num_pairs={num_pairs} is smaller than dp_size={dp_size}; "
                "would drop all pairs when enforcing equal DP shards"
            )

        if mode == "baseline_stride":
            rank_pairs = self._stride_partition_by_sample(train_data, dp_size)
        else:
            dropped_pairs = num_pairs % dp_size
            if dropped_pairs:
                logger.warning(
                    "Drop last %s train pairs after DP split so every DP rank has the same number "
                    "of pairs (num_pairs=%s, dp_size=%s)",
                    dropped_pairs,
                    num_pairs,
                    dp_size,
                )
            pairs_per_rank = num_pairs // dp_size
            rank_pairs = [train_data[rank * pairs_per_rank : (rank + 1) * pairs_per_rank] for rank in range(dp_size)]

        shards: list[dict[str, list[dict[str, Any]]]] = []
        for shard_pairs in rank_pairs:
            shard: dict[str, Any] = {"train_data": shard_pairs}
            if scheduler_timesteps is not None:
                shard["scheduler_timesteps"] = scheduler_timesteps
            if scheduler_sigmas is not None:
                shard["scheduler_sigmas"] = scheduler_sigmas
            shards.append(shard)
        return shards

    @staticmethod
    def _stride_partition_by_sample(
        train_data: list[dict[str, Any]],
        dp_size: int,
    ) -> list[list[dict[str, Any]]]:
        """Assign each sample's pairs to rank ``(sample_position % dp_size)``.

        Mirrors the legacy ``_split_train_data_by_dp`` which did
        ``partitions = [range(i, num_samples, dp_size) for i in range(dp_size)]``
        over the per-sample lists. Pairs are sample-major, so grouping by
        ``sample_index`` in first-seen order recovers sample positions; within a
        rank the original sample-major order is preserved (so the actor's
        contiguous micro-batch chunking reproduces the legacy tiles).
        """
        groups: OrderedDict[Any, list[dict[str, Any]]] = OrderedDict()
        for pair in train_data:
            groups.setdefault(pair["sample_index"], []).append(pair)
        sample_groups = list(groups.values())  # first-seen (sample-major) order

        per_sample_counts = {len(g) for g in sample_groups}
        if len(per_sample_counts) != 1:
            raise ValueError(
                f"baseline_stride split requires every sample to contribute the same number of "
                f"train pairs, got counts {sorted(per_sample_counts)}"
            )
        if len(sample_groups) % dp_size != 0:
            raise ValueError(
                f"baseline_stride split requires num_samples ({len(sample_groups)}) divisible by "
                f"dp_size ({dp_size}) for equal shards"
            )

        rank_pairs: list[list[dict[str, Any]]] = [[] for _ in range(dp_size)]
        for pos, grp in enumerate(sample_groups):
            rank_pairs[pos % dp_size].extend(grp)
        return rank_pairs


def build_microbatch_schedule(
    *,
    num_pairs_per_optim_step: int,
    num_optim_steps_per_rollout: int,
    micro_batch_size: int,
) -> list[list[tuple[int, int]]]:
    """Absolute train-pair ranges for every optimizer step and micro-batch."""
    if num_pairs_per_optim_step % micro_batch_size != 0:
        raise ValueError(
            f"num_pairs_per_optim_step={num_pairs_per_optim_step} must be a whole multiple of "
            f"micro_batch_size={micro_batch_size} (no ragged final micro-batch)"
        )
    schedule: list[list[tuple[int, int]]] = []
    for step_id in range(num_optim_steps_per_rollout):
        step_pair_lo = step_id * num_pairs_per_optim_step
        step_pair_hi = step_pair_lo + num_pairs_per_optim_step
        step_ranges = []
        for pair_lo in range(step_pair_lo, step_pair_hi, micro_batch_size):
            pair_hi = min(step_pair_hi, pair_lo + micro_batch_size)
            step_ranges.append((pair_lo, pair_hi))
        schedule.append(step_ranges)
    return schedule


def _chunk_indices(total: int, chunk_size: int) -> list[list[int]]:
    """Split range(total) into contiguous index chunks of size <= chunk_size."""
    chunk_size = max(1, chunk_size)
    return [list(range(start, min(start + chunk_size, total))) for start in range(0, total, chunk_size)]


def build_tiled_microbatch_schedule(
    *,
    num_samples_per_optim_step: int,
    sde_window_size: int,
    num_optim_steps_per_rollout: int,
    sample_microbatch: int,
    tstep_microbatch: int,
    iter_order: str = "sample_major",
) -> list[list[list[int]]]:
    """Legacy 2D (sample x timestep) tile grouping, as flat-pair-index micro-batches.

    Reproduces TrainRayActor._run_optim_window: within each optimizer step's window
    of ``num_samples_per_optim_step`` samples (each carrying ``sde_window_size`` SDE
    timesteps), chunk samples by ``sample_microbatch`` and timesteps by
    ``tstep_microbatch``, iterate the chunks per ``iter_order``, and flatten each
    tile sample-major (matching _forward_tile's ``[sample_indices][:, tstep_indices]``
    reshape). The flat sample-major pair index of ``(sample_pos, tstep_pos)`` within a
    window is ``sample_pos * sde_window_size + tstep_pos``; ``base`` offsets to the
    optimizer step. Tiles are non-contiguous when ``tstep_microbatch < sde_window_size``.

    Returns ``schedule[optim_step][micro_batch]`` = list of absolute pair indices.
    """
    if iter_order not in ("sample_major", "timestep_major"):
        raise ValueError(f"unknown iter_order {iter_order!r}")
    sample_mb = min(max(1, sample_microbatch), num_samples_per_optim_step)
    tstep_mb = min(max(1, tstep_microbatch), sde_window_size)

    schedule: list[list[list[int]]] = []
    for step_id in range(num_optim_steps_per_rollout):
        base = step_id * num_samples_per_optim_step * sde_window_size
        sample_chunks = _chunk_indices(num_samples_per_optim_step, sample_mb)
        tstep_chunks = _chunk_indices(sde_window_size, tstep_mb)
        if iter_order == "sample_major":
            outer_chunks, inner_chunks = tstep_chunks, sample_chunks
        else:
            outer_chunks, inner_chunks = sample_chunks, tstep_chunks

        microbatches: list[list[int]] = []
        for outer in outer_chunks:
            for inner in inner_chunks:
                if iter_order == "sample_major":
                    sample_chunk, tstep_chunk = inner, outer
                else:
                    sample_chunk, tstep_chunk = outer, inner
                microbatches.append([base + sp * sde_window_size + tp for sp in sample_chunk for tp in tstep_chunk])
        schedule.append(microbatches)
    return schedule


def reorder_train_pairs_for_tiling(
    train_data: list[dict[str, Any]],
    *,
    num_optim_steps_per_rollout: int,
    sample_microbatch: int,
    tstep_microbatch: int,
    iter_order: str = "sample_major",
) -> list[dict[str, Any]]:
    """Reorder a rank's sample-major train pairs so the legacy sample x timestep tiles are
    contiguous; the train actor then reproduces them with a plain contiguous schedule.
    Requires uniform tiles -- a contiguous micro_batch_size cannot express ragged ones."""
    num_samples = len({pair["sample_index"] for pair in train_data})
    sde_window_size = len(train_data) // num_samples
    num_samples_per_optim_step = num_samples // num_optim_steps_per_rollout
    eff_sample = min(sample_microbatch, num_samples_per_optim_step)
    eff_tstep = min(tstep_microbatch, sde_window_size)
    if num_samples_per_optim_step % eff_sample or sde_window_size % eff_tstep:
        raise ValueError(
            f"reorder needs uniform tiles: num_samples_per_optim_step={num_samples_per_optim_step} not divisible by "
            f"sample_microbatch={eff_sample} or sde_window_size={sde_window_size} not by tstep_microbatch={eff_tstep}"
        )
    schedule = build_tiled_microbatch_schedule(
        num_samples_per_optim_step=num_samples_per_optim_step,
        sde_window_size=sde_window_size,
        num_optim_steps_per_rollout=num_optim_steps_per_rollout,
        sample_microbatch=sample_microbatch,
        tstep_microbatch=tstep_microbatch,
        iter_order=iter_order,
    )
    return [train_data[i] for step in schedule for micro_batch in step for i in micro_batch]


def validate_sample_aligned_windows(
    *,
    train_pairs: list[dict[str, Any]],
    microbatch_schedule: list[list[tuple[int, int]]],
) -> None:
    """Ensure every optimizer window holds whole samples.

    Objectives whose gradient math sums one sample's pairs within a single
    optimizer step (e.g. NFT's K timestep pairs) opt in via the loss-function
    attribute ``requires_sample_aligned_windows``.
    """
    group_sizes: OrderedDict[Any, int] = OrderedDict()
    for pair in train_pairs:
        key = pair["sample_index"]
        group_sizes[key] = group_sizes.get(key, 0) + 1
    sizes = set(group_sizes.values())
    if len(sizes) > 1:
        raise ValueError(
            f"loss requires sample-aligned optimizer windows, but samples contribute unequal "
            f"pair counts on this rank ({sorted(sizes)}); a sample was likely split across "
            "DP ranks — make the sample count divisible by the DP size"
        )
    for step_ranges in microbatch_schedule:
        window_lo = step_ranges[0][0]
        if window_lo == 0:
            continue
        if train_pairs[window_lo - 1]["sample_index"] == train_pairs[window_lo]["sample_index"]:
            raise ValueError(
                f"loss requires sample-aligned optimizer windows, but the window starting at "
                f"pair {window_lo} splits sample_index={train_pairs[window_lo]['sample_index']}; "
                "make samples-per-rank divisible by --num-steps-per-rollout"
            )


def validate_same_microbatch_counts_across_train_ranks(
    *,
    microbatch_schedule: list[list[tuple[int, int]]],
    parallel_state,
) -> None:
    """Ensure every rank in the flattened DP×SP FSDP mesh runs the same number of micro-batches."""
    local_microbatch_counts = [len(step_ranges) for step_ranges in microbatch_schedule]
    gathered_microbatch_counts = [None] * parallel_state.dp_sp_size
    dist.all_gather_object(
        gathered_microbatch_counts,
        local_microbatch_counts,
        group=parallel_state.dp_sp_group_gloo,
    )
    if any(counts != local_microbatch_counts for counts in gathered_microbatch_counts):
        raise ValueError(
            "Uneven train-pair counts would make training ranks run different numbers of FSDP "
            f"micro-batches per optimizer step: {gathered_microbatch_counts}"
        )
