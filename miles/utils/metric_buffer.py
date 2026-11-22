"""Scalar metric accumulation with the reduction op declared once per metric."""

import math
from enum import Enum, auto

import torch
import torch.distributed as dist


class MetricReduce(Enum):
    MEAN = auto()
    MAX = auto()
    REPLICATED = auto()


def _group_size(group) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    return dist.get_world_size(group)


class MetricBuffer:
    """Accumulate scalar metrics over micro-batches, then reduce them over `group`.

    The schema fixes each metric's op and, since the flush packs values positionally,
    a layout every rank agrees on. MEAN takes a local sum plus the item count behind
    it, so the flush weights items rather than ranks. `group` must cover every rank
    holding a distinct share of the batch: DP here, since SP peers duplicate the loss
    rather than split it, and a duplicate cancels out of both a sum/count and a max.
    """

    def __init__(self, *, group, device: torch.device, schema: dict[str, MetricReduce]) -> None:
        self._group = group
        self._device = device
        self._schema = schema
        self._mean_keys = sorted(key for key, op in schema.items() if op is MetricReduce.MEAN)
        self._max_keys = sorted(key for key, op in schema.items() if op is MetricReduce.MAX)
        self._sums = {key: self._scalar(0.0) for key in self._mean_keys}
        self._counts = dict.fromkeys(self._mean_keys, 0.0)
        self._maxes = {key: self._scalar(-math.inf) for key in self._max_keys}
        self._replicated: dict[str, torch.Tensor] = {}

    def _scalar(self, value: torch.Tensor | float) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            return torch.tensor(value, device=self._device, dtype=torch.float64)
        # reshape(()) rejects a non-scalar instead of silently reducing it.
        return value.detach().reshape(()).to(device=self._device, dtype=torch.float64)

    def _check_reduce(self, key: str, expected: MetricReduce) -> None:
        actual = self._schema[key]
        if actual is not expected:
            raise ValueError(f"metric {key!r} reduces as {actual.name}, not {expected.name}")

    def emit_mean(self, key: str, total: torch.Tensor, count: int) -> None:
        self._check_reduce(key, MetricReduce.MEAN)
        self._sums[key] = self._sums[key] + self._scalar(total)
        self._counts[key] += float(count)

    def emit_max(self, key: str, value: torch.Tensor) -> None:
        self._check_reduce(key, MetricReduce.MAX)
        self._maxes[key] = torch.maximum(self._maxes[key], self._scalar(value))

    def emit_replicated(self, key: str, value: torch.Tensor) -> None:
        self._check_reduce(key, MetricReduce.REPLICATED)
        self._replicated[key] = self._scalar(value)

    def reduce(self) -> dict[str, float]:
        """Combine across `group` and materialize; every rank in it must call this."""
        world = _group_size(self._group)
        out: dict[str, float] = {}

        if self._mean_keys:
            counts = torch.tensor(
                [self._counts[key] for key in self._mean_keys], device=self._device, dtype=torch.float64
            )
            packed = torch.cat([torch.stack([self._sums[key] for key in self._mean_keys]), counts])
            if world > 1:
                dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=self._group)
            totals = packed.tolist()
            for index, key in enumerate(self._mean_keys):
                count = totals[len(self._mean_keys) + index]
                # A metric no rank contributed to this step is not a datapoint.
                if count > 0:
                    out[key] = totals[index] / count

        if self._max_keys:
            packed = torch.stack([self._maxes[key] for key in self._max_keys])
            if world > 1:
                dist.all_reduce(packed, op=dist.ReduceOp.MAX, group=self._group)
            for key, value in zip(self._max_keys, packed.tolist(), strict=True):
                if value != -math.inf:
                    out[key] = value

        if self._replicated:
            keys = sorted(self._replicated)
            packed = torch.stack([self._replicated[key] for key in keys])
            out.update(zip(keys, packed.tolist(), strict=True))

        return out
