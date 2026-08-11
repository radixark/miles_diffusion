#!/usr/bin/env bash
# Re-capture snapshot/ from an rx devbox running the official image, and rewrite the pins that
# can be derived from it. Run this when rockdu/miles_diffusion moves; review the diff.
#
#   bash refresh.sh <devbox-name>

set -euo pipefail

BOX="${1:?usage: refresh.sh <devbox-name>}"
SNAP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/snapshot"
run() { rx devbox run "$BOX" -- bash -c "$1"; }

IMAGE="$(rx devbox status "$BOX" | sed -n 's/^Image: *//p' | tr -d ' ')"
echo "[refresh] $BOX -> $IMAGE"

run 'pip freeze' > "$SNAP_DIR/freeze.tmp"
run 'cat /root/.cache/sglang/kernels.lock' > "$SNAP_DIR/kernels.lock"
PIP_VER="$(run 'pip --version' | awk '{print $2}')"

# Single quotes cannot appear in the heredoc below: run() passes it through bash -c "$1", so
# they would terminate the outer quoting. chr(45) is a hyphen.
#
# The sglang commit comes from the editable install's setuptools_scm version, not from
# `git rev-parse HEAD`: on a devbox people work on, HEAD drifts onto local commits that exist
# in no remote, and pinning one of those makes install.sh fail its ancestor check.
SGLANG_SCM="$(run 'python -c "import importlib.metadata as m; print(m.version(\"sglang\"))"' | tr -d '\r\n ')"
SGLANG_COMMIT="$(run "git -C /sgl-workspace/sglang rev-parse ${SGLANG_SCM##*+g}" | tr -d '\r\n ')"
echo "[refresh] sglang $SGLANG_SCM -> $SGLANG_COMMIT"

run 'python - <<'"'"'EOF'"'"'
import time
from pathlib import Path
for tree, label in (("/usr/lib/python3/dist-packages", "apt"),
                    ("/usr/local/lib/python3.12/dist-packages", "pip")):
    for info in sorted(Path(tree).glob("*-info")):
        day = time.strftime("%Y-%m-%d", time.gmtime(info.stat().st_mtime))
        print(f"{info.name.split(chr(45))[0]}\t{label}\t{day}")
EOF' > "$SNAP_DIR/dists.tmp"

python3 - "$SNAP_DIR" "$IMAGE" <<'PY'
import re, sys
from collections import Counter, defaultdict
from pathlib import Path

snap, image = Path(sys.argv[1]), sys.argv[2]
canon = lambda n: re.sub(r"[-_.]+", "-", n).lower()
pins = (snap / "pins.env").read_text()
pin = lambda k: re.search(rf'^{k}="\$\{{{k}:-(.*)\}}"', pins, re.M).group(1)
fa3_repo, fa3_tag = pin("FA3_WHEELS_REPO"), pin("FA3_WHEELS_TAG")

trees, days = defaultdict(set), {}
for line in (snap / "dists.tmp").read_text().splitlines():
    name, tree, day = line.split("\t")
    trees[canon(name)].add(tree)
    if tree == "pip":
        days[canon(name)] = day

# A working devbox is not a pristine image: the image's own pip layers all land on its build
# day, dozens at a time, while a human's later install leaves one or two dated well after.
counts = Counter(days.values())
build_day = max((d for d, n in counts.items() if n >= 10), default="")
drift = {n for n, d in days.items() if d > build_day}
if drift:
    print(f"[refresh] build day {build_day}; dropping later installs: {', '.join(sorted(drift))}")

SKIP = {"deep-ep": "image-only", "flash-mla": "image-only", "sglang": "git", "miles": "git"}
out = [
    f"# Package set of {image}, captured by refresh.sh.",
    "# install.sh installs every uncommented line with --no-deps; #skip lines record what the",
    "# image has that we deliberately do not install. verify_env.py compares against both halves.",
    "#   apt           Ubuntu 24.04 ships it; snapshot/apt.txt installs the same version",
    "#   image-only    built from source inside lmsysorg/sglang, published nowhere",
    "#   git           install.sh installs it from a pinned commit",
    "#   capture-drift installed on the capture box after the image was built",
    "",
]
for line in (snap / "freeze.tmp").read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    egg = re.search(r"#egg=([A-Za-z0-9_.\-]+)", line)
    name = canon(egg.group(1) if egg else re.split(r"[ =@]", line, 1)[0])
    reason = SKIP.get(name)
    if not reason and name in drift:
        reason = "capture-drift"
    if not reason and trees[name] and trees[name] <= {"apt"}:
        reason = "apt"
    # The image installs the FA3 wheel from a local copy, so freeze reports a file:// path that
    # no other machine can resolve. Point it back at the release the Dockerfile pulls from.
    if name == "flash-attn-3" and "file://" in line:
        line = re.sub(r"file://\S*/(flash_attn_3-\S+?\.whl)",
                      rf"https://github.com/{fa3_repo}/releases/download/{fa3_tag}/\1", line)
    out.append(f"#skip[{reason}] {line}" if reason else line)

(snap / "packages.txt").write_text("\n".join(out) + "\n")
installed = sum(1 for x in out if x and not x.startswith("#"))
print(f"[refresh] packages.txt: {installed} installed, {len(out) - installed - 8} skipped")
PY

rm -f "$SNAP_DIR/freeze.tmp" "$SNAP_DIR/dists.tmp"

for kv in "OFFICIAL_IMAGE=$IMAGE" "SGLANG_COMMIT=$SGLANG_COMMIT" "PIP_VER=$PIP_VER"; do
  key="${kv%%=*}"
  sed -i.bak "s|^${key}=.*|${key}=\"\\\${${key}:-${kv#*=}}\"|" "$SNAP_DIR/pins.env"
done
rm -f "$SNAP_DIR/pins.env.bak"
echo "[refresh] pins updated"
echo "[refresh] if the box ran :latest, replace OFFICIAL_IMAGE with the concrete tag it resolved to"
