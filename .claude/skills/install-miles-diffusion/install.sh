#!/usr/bin/env bash
# Install miles_diffusion on a bare CUDA 12.9 Ubuntu 24.04 box, reproducing the package
# versions of the official image. snapshot/ is what it replays; verify_env.py checks the result.
#
#   bash install.sh [--from STEP]     steps: apt pip packages sglang miles verify

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SKILL_DIR/../../.." && pwd)"
SNAP_DIR="$SKILL_DIR/snapshot"
# shellcheck disable=SC1091
source "$SNAP_DIR/pins.env"

STEPS=(apt pip packages sglang miles verify)
PIP=(python3 -m pip)

log()  { printf "\033[1;34m[install]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

step_apt() {
  local pkgs=()
  while read -r line; do
    line="${line%%#*}"
    [[ -n "${line// /}" ]] && pkgs+=("$line")
  done < "$SNAP_DIR/apt.txt"
  log "apt-get install ${#pkgs[@]} packages"
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${pkgs[@]}"
}

step_pip() {
  # The image installs into the system interpreter with PEP 668 off, not into a venv;
  # matching that keeps sys.path and the apt-provided dists identical to it.
  "${PIP[@]}" config set global.break-system-packages true >/dev/null
  # --ignore-installed: these three arrive from apt without a RECORD, so pip refuses to replace
  # them and aborts the packages step. A fourth would fail the same way, with the same fix.
  "${PIP[@]}" install --ignore-installed "pip==$PIP_VER" PyJWT wheel
  "${PIP[@]}" --version
}

step_packages() {
  local list=/tmp/miles-packages.txt
  grep -vE '^\s*(#|$)' "$SNAP_DIR/packages.txt" > "$list"

  # Hosts can mount a binary read-only over the scripts dir (rx devboxes do this for uv);
  # pip cannot replace such a file and aborts the whole transaction at commit time.
  local scripts_dir blocked=()
  scripts_dir="$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
  while read -r mp; do
    [[ "$(dirname "$mp")" == "$scripts_dir" ]] || continue
    grep -qiE "^$(basename "$mp")==" "$list" && blocked+=("$(basename "$mp")")
  done < <(awk '{print $2}' /proc/mounts)
  if [[ ${#blocked[@]} -gt 0 ]]; then
    warn "read-only host mounts block: ${blocked[*]} (absent from a plain docker image)"
    printf '%s\n' "${blocked[@]}" > /tmp/miles-skipped.txt
    grep -viE "$(printf '^%s==|' "${blocked[@]}" | sed 's/|$//')" "$list" > "$list.f" && mv "$list.f" "$list"
  else
    : > /tmp/miles-skipped.txt
  fi

  # --no-deps is the only correct mode: the list is a complete freeze closure, and the image
  # does not satisfy its own metadata (pip check reports ~20 violations there), so resolving
  # returns ResolutionImpossible instead of the versions the image actually has.
  log "pip install $(wc -l < "$list") pinned packages"
  "${PIP[@]}" install --no-deps -r "$list" \
    --extra-index-url "$TORCH_INDEX" \
    --extra-index-url "$SGLANG_KERNEL_INDEX" \
    --extra-index-url "$FLASHINFER_INDEX"
}

step_sglang() {
  mkdir -p "$(dirname "$SGLANG_DIR")"
  if [[ ! -d "$SGLANG_DIR/.git" ]]; then
    git init -q "$SGLANG_DIR"
    git -C "$SGLANG_DIR" remote add origin "$SGLANG_REPO"
  fi
  git -C "$SGLANG_DIR" fetch --filter=blob:none -q origin "$SGLANG_BRANCH"
  local tip target
  tip="$(git -C "$SGLANG_DIR" rev-parse FETCH_HEAD)"
  target="$SGLANG_COMMIT"
  # Fetching the branch and checking ancestry is what stops a fork commit or a local
  # cherry-pick from being installed as if it were upstream.
  git -C "$SGLANG_DIR" merge-base --is-ancestor "$target" "$tip" 2>/dev/null \
    || die "SGLANG_COMMIT $target is not on $SGLANG_BRANCH (tip $tip)"
  git -C "$SGLANG_DIR" checkout --detach -q "$target"
  log "sglang $target, on $SGLANG_BRANCH, $(git -C "$SGLANG_DIR" rev-list --count "$target..$tip") behind tip"

  # SGLANG_BUILD_RUST_EXTS=none skips the gRPC Rust extension; a bare CUDA image has no Rust.
  ( cd "$SGLANG_DIR" && SGLANG_BUILD_RUST_EXTS=none "${PIP[@]}" install -e python --no-deps )
  install -Dm644 "$SNAP_DIR/kernels.lock" "$KERNELS_LOCK_DEST"
}

step_miles() {
  local head tip
  head="$(git -C "$REPO_DIR" rev-parse HEAD)"
  git -C "$REPO_DIR" fetch --filter=blob:none -q origin "$MILES_BRANCH"
  tip="$(git -C "$REPO_DIR" rev-parse FETCH_HEAD)"
  if git -C "$REPO_DIR" merge-base --is-ancestor "$head" "$tip"; then
    log "miles $head, on $MILES_BRANCH, $(git -C "$REPO_DIR" rev-list --count "$head..$tip") behind tip"
  else
    warn "miles HEAD $head is not on $MILES_BRANCH"
  fi
  ( cd "$REPO_DIR" && "${PIP[@]}" install -e . --no-deps )

  log "caching PaddleOCR $PADDLEOCR_LANG models"
  python3 -c "
from paddleocr import PaddleOCR
PaddleOCR(use_angle_cls=False, lang='$PADDLEOCR_LANG', use_gpu=False, show_log=False)"
}

step_verify() {
  python3 "$SKILL_DIR/verify_env.py" --skipped /tmp/miles-skipped.txt
  ( cd "$REPO_DIR" && python3 -c "
import torch, sglang, sglang.multimodal_gen, train_diffusion, paddleocr  # noqa: F401
from miles.backends.fsdp_utils import FSDPTrainRayActor  # noqa: F401
print('torch', torch.__version__, torch.cuda.device_count(), 'gpu(s); sglang', sglang.__version__)" )
}

FROM="${1:-}"
[[ "$FROM" == "--from" ]] && FROM="${2:-}"
started=0
log "target $OFFICIAL_IMAGE"
for s in "${STEPS[@]}"; do
  [[ -n "$FROM" && $started -eq 0 && "$s" != "$FROM" ]] && continue
  started=1
  log "=== $s ==="
  "step_$s"
done
log "done — conda is not involved; run scripts with the system python3"
