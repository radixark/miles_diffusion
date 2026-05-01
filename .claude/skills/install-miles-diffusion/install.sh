#!/usr/bin/env bash
# One-click installer for miles-diffusion on a clean Linux GPU box.
# Idempotent: re-running skips steps that are already done.
#
# Overrides (env vars):
#   ENV_NAME        conda env name (default: miles-diffusion)
#   PY_VER          python version (default: 3.11)
#   SGLANG_DIR      where to clone sglang (default: ../sglang)
#   SGLANG_REPO     sglang git URL (default: https://github.com/Rockdu/sglang.git)
#   SGLANG_BRANCH   sglang branch to check out (default: sglang-diffusion-rollout-test)
#   SGLANG_COMMIT   sglang commit SHA to pin (default: pinned working SHA below)
#   CUDA_VER        torch cuda tag (default: 12.9 -> cu129)
#   TORCH_VER       torch version (default: 2.9.1)
#
# All package versions are pinned. Pins reflect the currently-validated working
# environment for miles-diffusion + sglang-diffusion + flow_grpo OCR reward.
# Override only if you know what you're doing.
#
# sglang source of truth: the sglang-diffusion fork lives at
#   Rockdu/sglang @ sglang-diffusion-rollout-test
# miles-diffusion depends on that branch (multimodal_gen +
# update_weights_from_tensor for RL weight sync). The branch tip moves; we pin
# to a specific commit SHA via SGLANG_COMMIT for bit-reproducibility.

set -euo pipefail

ENV_NAME="${ENV_NAME:-miles-diffusion}"
PY_VER="${PY_VER:-3.11}"
CUDA_VER="${CUDA_VER:-12.9}"
TORCH_VER="${TORCH_VER:-2.9.1}"
SGLANG_REPO="${SGLANG_REPO:-https://github.com/Rockdu/sglang.git}"
SGLANG_BRANCH="${SGLANG_BRANCH:-sglang-diffusion-rollout-test}"
SGLANG_COMMIT="${SGLANG_COMMIT:-0372158dd66bc7cb0740c733bd60047db790ec7d}"

# Tooling pins (pip resolver behaviour depends on these).
PIP_VER="${PIP_VER:-26.0.1}"
WHEEL_VER="${WHEEL_VER:-0.45.1}"
SETUPTOOLS_VER="${SETUPTOOLS_VER:-82.0.1}"
TORCH_MEMORY_SAVER_VER="${TORCH_MEMORY_SAVER_VER:-0.0.9}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SGLANG_DIR="${SGLANG_DIR:-$(dirname "$REPO_DIR")/sglang}"

log()  { printf "\033[1;34m[install]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"; }

# ------------------------------------------------------------------ preflight
log "repo:    $REPO_DIR"
log "env:     $ENV_NAME (python $PY_VER)"
log "sglang:  $SGLANG_DIR ($SGLANG_REPO @ $SGLANG_BRANCH, commit $SGLANG_COMMIT)"
log "cuda:    $CUDA_VER"
log "torch:   $TORCH_VER"
log "tooling: pip==$PIP_VER wheel==$WHEEL_VER setuptools==$SETUPTOOLS_VER"

need git
if command -v mamba >/dev/null 2>&1; then
  CONDA_BIN=mamba
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN=conda
else
  die "conda/mamba not found. Install miniforge: https://github.com/conda-forge/miniforge"
fi
log "using: $CONDA_BIN"

# ---------------------------------------------------------------- apt deps
if command -v apt-get >/dev/null 2>&1; then
  log "apt: libglib2.0-0 libgl1"
  SUDO=""
  [[ $EUID -ne 0 ]] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"
  $SUDO apt-get update -qq || warn "apt-get update failed; continuing"
  $SUDO apt-get install -y libglib2.0-0 libgl1 || warn "apt install failed; continuing (check libGL/libglib presence manually)"
else
  warn "apt-get not available; skip system lib install"
fi

# ---------------------------------------------------------------- conda env
source "$($CONDA_BIN info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "conda env '$ENV_NAME' exists; reusing"
else
  log "creating conda env '$ENV_NAME'"
  $CONDA_BIN create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"
log "python: $(python --version)"

# Pin the build/install tooling so resolution is stable across machines.
python -m pip install "pip==$PIP_VER" "wheel==$WHEEL_VER" "setuptools==$SETUPTOOLS_VER"

# ---------------------------------------------------------------- pytorch
CU_TAG="cu$(echo "$CUDA_VER" | tr -d .)"
if python -c "import torch" 2>/dev/null; then
  CUR_TORCH="$(python -c 'import torch; print(torch.__version__)')"
  if [[ "$CUR_TORCH" == "${TORCH_VER}+${CU_TAG}" || "$CUR_TORCH" == "$TORCH_VER" ]]; then
    log "torch: $CUR_TORCH (already at pinned version)"
  else
    warn "torch installed at $CUR_TORCH, expected ${TORCH_VER}+${CU_TAG}; reinstalling"
    pip install --force-reinstall "torch==$TORCH_VER" --index-url "https://download.pytorch.org/whl/$CU_TAG"
  fi
else
  log "installing torch==$TORCH_VER ($CU_TAG)"
  pip install "torch==$TORCH_VER" --index-url "https://download.pytorch.org/whl/$CU_TAG"
fi

# ---------------------------------------------------------------- sglang-diffusion
# Depends on Rockdu/sglang @ sglang-diffusion-rollout-test (sglang-diffusion
# fork with update_weights_from_tensor for multimodal_gen). Pinned to a
# specific commit so bit-exact rollout behaviour is reproducible.
if [[ ! -d "$SGLANG_DIR" ]]; then
  log "cloning $SGLANG_REPO -> $SGLANG_DIR"
  git clone --branch "$SGLANG_BRANCH" "$SGLANG_REPO" "$SGLANG_DIR"
fi

pushd "$SGLANG_DIR" >/dev/null
# Make sure the "rockdu" remote is registered so `git fetch rockdu` works even
# if the repo was cloned earlier from a different URL.
if ! git remote get-url rockdu >/dev/null 2>&1; then
  git remote add rockdu "$SGLANG_REPO"
fi
# Always fetch the pinned commit (it might not be reachable from a stale cache
# of the branch).
if ! git cat-file -e "$SGLANG_COMMIT^{commit}" 2>/dev/null; then
  log "fetching commit $SGLANG_COMMIT from rockdu"
  git fetch rockdu "$SGLANG_BRANCH"
fi
CUR_SGLANG_COMMIT="$(git rev-parse HEAD)"
if [[ "$CUR_SGLANG_COMMIT" != "$SGLANG_COMMIT" ]]; then
  log "checkout sglang $SGLANG_COMMIT (was $CUR_SGLANG_COMMIT)"
  git checkout --detach "$SGLANG_COMMIT"
else
  log "sglang already at pinned commit $SGLANG_COMMIT"
fi

if python -c "import sglang.multimodal_gen" 2>/dev/null; then
  log "sglang.multimodal_gen already importable; skip pip install"
else
  log "pip install sglang (editable, all extras)"
  pip install -e "python[all]"
fi
popd >/dev/null

# ---------------------------------------------------------------- miles
cd "$REPO_DIR"
log "pip install -r requirements.txt (all pinned)"
pip install -r requirements.txt
log "pip install -e . (miles)"
pip install -e . --no-deps

# ---------------------------------------------------------------- flow_grpo deps
if [[ -f "$REPO_DIR/flow_grpo/setup.sh" ]]; then
  log "installing flow_grpo OCR deps (all pinned: paddleocr, peft, diffusers, ...)"
  pushd "$REPO_DIR/flow_grpo" >/dev/null
  # Skip the `pip install -e .` line inside setup.sh — flow_grpo is a sibling
  # tree we reference, not a package to install into miles' env. The rest of
  # the file is pinned --no-deps pip installs plus apt calls we already did.
  grep -v '^pip install -e . --no-deps$' setup.sh | \
    grep -v '^apt-get install ' | \
    bash
  popd >/dev/null
else
  warn "flow_grpo/setup.sh not found; skipping OCR reward deps"
fi

# ---------------------------------------------------------------- optional
if python -c "import torch_memory_saver" 2>/dev/null; then
  log "torch_memory_saver: $(python -c 'import torch_memory_saver as m; print(getattr(m, \"__version__\", \"?\"))') (already installed)"
else
  log "installing torch_memory_saver==$TORCH_MEMORY_SAVER_VER (optional)"
  pip install "torch_memory_saver==$TORCH_MEMORY_SAVER_VER" || warn "torch_memory_saver install failed; continuing without it"
fi

# ---------------------------------------------------------------- smoke test
log "smoke test: nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || warn "nvidia-smi returned non-zero"
else
  warn "nvidia-smi not found; GPU presence unknown"
fi

log "smoke test: python import train_diffusion"
cd "$REPO_DIR"
python -c "
import train_diffusion  # noqa
from miles.utils.arguments import parse_args  # noqa
from miles.backends.fsdp_utils import FSDPTrainRayActor  # noqa
import sglang.multimodal_gen  # noqa
print('miles-diffusion import OK')
"

log ""
log "=========================================="
log "  install done."
log "  next:"
log "    conda activate $ENV_NAME"
log "    export WANDB_API_KEY=...  # optional"
log "    bash scripts/run-diffusion-grpo-ocr.sh"
log "=========================================="
