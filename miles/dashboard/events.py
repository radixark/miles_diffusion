"""Trajectory event protocol, aligned with miles core (miles/dashboard/store.py).

Same TrajectoryEvent envelope as core; the kind enum extends core's gen/tool/attempt
with the diffusion-specific deser/reward stages so our own reader (viewer.py) folds
them while the wire format stays mergeable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


@dataclass
class PhaseEvent:
    name: str
    t0: float
    t1: float
    role: str
    pid: int
    node: str
    gpus: list[int]
    rank: int

    def to_dict(self) -> dict:
        return asdict(self)


class TrajectoryEventKind(StrEnum):
    GEN_START = "gen_start"
    GEN_END = "gen_end"
    DESER_START = "deser_start"
    DESER_END = "deser_end"
    REWARD_START = "reward_start"
    REWARD_END = "reward_end"


# stage name (StageTimer) -> (start kind, end kind)
STAGE_KINDS = {
    "generate": (TrajectoryEventKind.GEN_START, TrajectoryEventKind.GEN_END),
    "deserialize": (TrajectoryEventKind.DESER_START, TrajectoryEventKind.DESER_END),
    "reward": (TrajectoryEventKind.REWARD_START, TrajectoryEventKind.REWARD_END),
}

# kind -> (base span, is_start), for the reader to fold start/end back into spans
SPAN_KINDS = {
    "gen_start": ("gen", True),
    "gen_end": ("gen", False),
    "deser_start": ("deser", True),
    "deser_end": ("deser", False),
    "reward_start": ("reward", True),
    "reward_end": ("reward", False),
}


@dataclass
class TrajectoryEvent:
    ts: float
    kind: str
    sample_index: int
    group_index: int
    turn: int
    weight_version: str
    detail: str
    rollout_id: int

    def to_dict(self) -> dict:
        return asdict(self)
