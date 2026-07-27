"""Gloo worker for the MetricBuffer cross-rank reduction test (4 ranks)."""

import torch
import torch.distributed as dist

from miles.utils.metric_buffer import MetricBuffer, MetricReduce


def _buffer(group=None, **schema) -> MetricBuffer:
    return MetricBuffer(group=group, device=torch.device("cpu"), schema=schema)


def _assert_close(got: float, want: float, what: str) -> None:
    if abs(got - want) > 1e-9 * max(1.0, abs(want)):
        raise AssertionError(f"rank {dist.get_rank()}: {what} is {got!r}, expected {want!r}")


def case_mean_weights_items_not_ranks() -> None:
    """Rank 0 holds 4 items worth 8, the others 1 item worth 0: 8/7, not a mean of means."""
    rank = dist.get_rank()
    metrics = _buffer(loss=MetricReduce.MEAN)
    for _ in range(4 if rank == 0 else 1):
        metrics.emit_mean("loss", total=torch.tensor(2.0 if rank == 0 else 0.0), count=1)
    _assert_close(metrics.reduce()["loss"], 8.0 / 7.0, "loss mean")


def case_max_is_a_real_max() -> None:
    """Only the last rank sees the outlier; a mean of maxes would hide it."""
    rank = dist.get_rank()
    metrics = _buffer(abs_diff=MetricReduce.MAX)
    metrics.emit_max("abs_diff", torch.tensor(100.0 if rank == dist.get_world_size() - 1 else 0.5))
    metrics.emit_max("abs_diff", torch.tensor(0.25))
    _assert_close(metrics.reduce()["abs_diff"], 100.0, "abs_diff max")


def case_metric_only_one_rank_records() -> None:
    """The per-component case: rank 0 hits the phase, the others only have it in the schema."""
    metrics = _buffer(phase_high_noise=MetricReduce.MEAN, phase_low_noise=MetricReduce.MEAN, worst=MetricReduce.MAX)
    if dist.get_rank() == 0:
        metrics.emit_mean("phase_high_noise", total=torch.tensor(6.0), count=3)
        metrics.emit_max("worst", torch.tensor(7.0))
    reduced = metrics.reduce()
    _assert_close(reduced["phase_high_noise"], 2.0, "single-rank phase mean")
    _assert_close(reduced["worst"], 7.0, "single-rank max")
    if "phase_low_noise" in reduced:
        raise AssertionError(f"rank {dist.get_rank()}: a metric no rank recorded must be dropped")


def case_reduces_on_the_given_group_only() -> None:
    """A 2-rank subgroup must ignore the other pair, the way DP ignores SP peers."""
    rank = dist.get_rank()
    # dist.new_group is collective: every rank creates both groups.
    groups = [dist.new_group(ranks=[0, 1]), dist.new_group(ranks=[2, 3])]

    metrics = _buffer(groups[0] if rank < 2 else groups[1], loss=MetricReduce.MEAN)
    metrics.emit_mean("loss", total=torch.tensor(1.0 if rank < 2 else 100.0), count=1)
    _assert_close(metrics.reduce()["loss"], 1.0 if rank < 2 else 100.0, "subgroup mean")


def main() -> None:
    dist.init_process_group(backend="gloo")
    try:
        if dist.get_world_size() != 4:
            raise AssertionError(f"expected 4 ranks, got {dist.get_world_size()}")
        case_mean_weights_items_not_ranks()
        case_max_is_a_real_max()
        case_metric_only_one_rank_records()
        case_reduces_on_the_given_group_only()
        dist.barrier()
        if dist.get_rank() == 0:
            print("[ok] MetricBuffer reduction on 4 ranks", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
