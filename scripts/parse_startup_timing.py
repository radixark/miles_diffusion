"""Rebuild the startup timeline from a ray job log containing STARTUP_TIMING lines.

Usage:
    python3 scripts/parse_startup_timing.py /path/to/run.log

Prints (1) a chronological cross-process timeline, (2) a per-step duration
rollup grouped by role, and (3) the driver-phase breakdown with percentages.
Only stdlib; safe to run anywhere.
"""

import re
import sys
from collections import defaultdict

LINE_RE = re.compile(
    r"STARTUP_TIMING role=(?P<role>\S+) pid=(?P<pid>\d+) event=(?P<event>\S+) t=(?P<t>[0-9.]+)(?: dur=(?P<dur>[0-9.]+))?(?P<extra>.*)"
)


def parse(path):
    events = []
    with open(path, errors="replace") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                events.append(
                    {
                        "role": m.group("role"),
                        "pid": m.group("pid"),
                        "event": m.group("event"),
                        "t": float(m.group("t")),
                        "dur": float(m.group("dur")) if m.group("dur") else None,
                        "extra": m.group("extra").strip(),
                    }
                )
    return sorted(events, key=lambda e: e["t"])


def fmt_t(t, t0):
    return f"+{t - t0:8.2f}s"


def main(path):
    events = parse(path)
    if not events:
        print("no STARTUP_TIMING lines found")
        return
    t0 = events[0]["t"]

    print("=" * 100)
    print("CHRONOLOGICAL TIMELINE (t=0 is the first probe)")
    print("=" * 100)
    for e in events:
        dur = f"  dur={e['dur']:8.2f}s" if e["dur"] is not None else ""
        extra = f"  {e['extra']}" if e["extra"] else ""
        print(f"{fmt_t(e['t'], t0)}  {e['role']:<16} {e['event']:<50}{dur}{extra}")

    print()
    print("=" * 100)
    print("PER-STEP DURATIONS (max across ranks per role-prefix, sorted desc)")
    print("=" * 100)
    durs = defaultdict(list)
    for e in events:
        if e["dur"] is not None and e["event"].endswith(".end"):
            durs[e["event"][: -len(".end")]].append((e["role"], e["dur"]))
    rows = []
    for name, vals in durs.items():
        worst_role, worst = max(vals, key=lambda rv: rv[1])
        rows.append((worst, name, worst_role, len(vals)))
    for worst, name, worst_role, n in sorted(rows, reverse=True):
        print(f"{worst:9.2f}s  {name:<55} worst={worst_role}  n={n}")

    print()
    print("=" * 100)
    print("DRIVER PHASES")
    print("=" * 100)
    driver = [e for e in events if e["role"] == "driver" and e["dur"] is not None]
    total = sum(e["dur"] for e in driver if e["event"].count(".") == 2)  # driver.<phase>.end only
    for e in driver:
        if e["event"].count(".") == 2:
            name = e["event"][: -len(".end")]
            pct = 100 * e["dur"] / total if total else 0
            print(f"{e['dur']:9.2f}s  {pct:5.1f}%  {name}")
    print(f"{total:9.2f}s  100.0%  total (sum of driver top-level phases)")


if __name__ == "__main__":
    main(sys.argv[1])
