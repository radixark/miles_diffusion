"""Fail the image build if any requirements.txt entry is missing or
version-mismatched (later --no-deps installs can silently stomp them)."""

import sys
from importlib.metadata import PackageNotFoundError, version

from packaging.requirements import Requirement

failures = []
for raw in open(sys.argv[1]):
    line = raw.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        continue
    req = Requirement(line)
    if req.marker is not None and not req.marker.evaluate():
        continue
    try:
        installed = version(req.name)
    except PackageNotFoundError:
        failures.append(f"{req.name}: not installed")
        continue
    if req.specifier and not req.specifier.contains(installed, prereleases=True):
        failures.append(f"{req.name}: installed {installed}, required {req.specifier}")

if failures:
    sys.exit("requirements not satisfied in image:\n  " + "\n  ".join(failures))
print(f"requirements OK ({sys.argv[1]})")
