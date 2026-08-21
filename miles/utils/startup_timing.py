"""Wall-clock probes for the startup path (process launch -> first rollout).

Every probe prints one machine-parseable line to stdout:

    STARTUP_TIMING role=<role> pid=<pid> event=<name> t=<epoch> [dur=<sec>] [k=v ...]

stdout of the driver and of every ray actor funnels into the ray job log, so
``tools/parse_startup_timing.py`` can rebuild the full cross-process timeline
from a single log file. ``print`` (not logging) keeps the probes independent of
per-process logger configuration; timestamps are embedded so log-line ordering
does not matter. This module must stay import-light (no torch/ray) because it
anchors each process's import-start time.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

# Anchor: time this module was imported in the current process. When imported
# before the heavy imports (see train_diffusion.py), `mark("x.enter")` minus
# this anchor measures the process's import/bootstrap cost.
PROC_T0 = time.time()

_role = "unknown"


def set_role(role: str) -> None:
    global _role
    _role = role
    mark("role_set", since_import=f"{time.time() - PROC_T0:.3f}")


def _emit(event: str, t: float, dur: float | None = None, **extra: object) -> None:
    parts = [f"STARTUP_TIMING role={_role}", f"pid={os.getpid()}", f"event={event}", f"t={t:.3f}"]
    if dur is not None:
        parts.append(f"dur={dur:.3f}")
    parts.extend(f"{k}={v}" for k, v in extra.items())
    print(" ".join(parts), flush=True)


def mark(event: str, **extra: object) -> None:
    _emit(event, time.time(), **extra)


@contextmanager
def step(event: str, **extra: object):
    t0 = time.time()
    _emit(f"{event}.begin", t0, **extra)
    try:
        yield
    finally:
        t1 = time.time()
        _emit(f"{event}.end", t1, dur=t1 - t0, **extra)
