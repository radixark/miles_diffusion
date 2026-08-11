#!/usr/bin/env python3
"""Compare installed package versions against snapshot/packages.txt, the official image's set.

Exits non-zero on any mismatch or missing package, so "installed" means "matches the image"
rather than "imports". #skip entries and packages install.sh reported skipping are expected
absences, not drift.
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path

# Editable installs report a setuptools_scm version that depends on clone depth rather than on
# the pinned commit; install.sh checks those by commit instead.
VERSION_EXEMPT = {"sglang", "miles"}


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def entry(line: str) -> tuple[str, str]:
    """name and version for one requirement line; version is empty when not comparable."""
    if line.startswith("-e ") or " @ " in line:
        egg = re.search(r"#egg=([A-Za-z0-9_.\-]+)", line)
        return canon(egg.group(1) if egg else line.split("@")[0].strip()), ""
    name, _, version = line.partition("==")
    return canon(name), version.strip()


def read_packages(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """name -> version for the whole image set, and name -> reason for #skip entries."""
    wanted, skipped = {}, {}
    for line in path.read_text().splitlines():
        line = line.strip()
        skip = re.match(r"#skip\[([a-z-]+)\]\s*(.*)", line)
        if skip:
            name, version = entry(skip.group(2))
            wanted[name], skipped[name] = version, skip.group(1)
        elif line and not line.startswith("#"):
            name, version = entry(line)
            wanted[name] = version
    return wanted, skipped


def installed() -> dict[str, str]:
    # An apt dist and a pip dist can both claim one name (PyJWT). distributions() walks sys.path
    # in order, so the first one seen is the copy an import resolves to.
    have: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            have.setdefault(canon(name), dist.version)
    return have


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(Path(__file__).resolve().parent / "snapshot"))
    ap.add_argument("--skipped", help="file listing packages this host could not install")
    args = ap.parse_args()

    snap = Path(args.snapshot)
    wanted, skipped = read_packages(snap / "packages.txt")
    if args.skipped and Path(args.skipped).exists():
        for name in Path(args.skipped).read_text().split():
            skipped[canon(name)] = "host-blocked"

    have = installed()
    matched, mismatched, missing, expected = [], [], [], []
    for name, want in sorted(wanted.items()):
        got = have.get(name)
        if got is None:
            (expected if name in skipped else missing).append((name, want, ""))
        elif name in VERSION_EXEMPT or not want or got == want:
            matched.append(name)
        else:
            (expected if name in skipped else mismatched).append((name, want, got))

    pip_want = re.search(r'^PIP_VER="\$\{PIP_VER:-(.*)\}"', (snap / "pins.env").read_text(), re.M)
    if pip_want and have.get("pip") not in (None, pip_want.group(1)):
        mismatched.append(("pip", pip_want.group(1), have["pip"]))

    extra = sorted(set(have) - set(wanted) - {"pip"})
    print(
        f"  matched {len(matched)}   mismatched {len(mismatched)}   missing {len(missing)}"
        f"   expected-absent {len(expected)}   extra {len(extra)}"
    )
    for name, want, got in mismatched:
        print(f"MISMATCH  {name}: image {want}, installed {got}")
    for name, want, _ in missing:
        print(f"MISSING   {name}=={want}")
    for name in extra:
        print(f"extra     {name}=={have[name]}")

    if mismatched or missing:
        print("environment does not match the official image", file=sys.stderr)
        return 1
    print("environment matches the official image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
