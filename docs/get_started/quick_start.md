# Quick Start

This document describes the recommended base environment for
`miles-diffusion`. It covers the common runtime needed by the diffusion training
entrypoint, the pinned sglang-diffusion fork, and the miles package.

Task-specific reward dependencies are intentionally kept out of the base setup.
After the base environment is ready, install the dependencies required by your
target recipe from [Task Dependencies](task_dependencies.md).

## Basic Environment Setup

Miles-diffusion depends on a custom sglang-diffusion fork for multimodal rollout
and RL weight synchronization. The sglang branch can move over time, so the
environment should pin the exact sglang commit instead of installing from a
floating branch tip.

Run the following block from the repository root:

```bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-miles-diffusion}"
PY_VER="${PY_VER:-3.11}"
CUDA_VER="${CUDA_VER:-12.9}"
TORCH_VER="${TORCH_VER:-2.9.1}"
SGLANG_REPO="${SGLANG_REPO:-https://github.com/Rockdu/sglang.git}"
SGLANG_BRANCH="${SGLANG_BRANCH:-sglang-diffusion-rollout-test}"
SGLANG_COMMIT="${SGLANG_COMMIT:-0372158dd66bc7cb0740c733bd60047db790ec7d}"

PIP_VER="${PIP_VER:-26.0.1}"
WHEEL_VER="${WHEEL_VER:-0.45.1}"
SETUPTOOLS_VER="${SETUPTOOLS_VER:-82.0.1}"
TORCH_MEMORY_SAVER_VER="${TORCH_MEMORY_SAVER_VER:-0.0.9}"

REPO_DIR="$(pwd)"
SGLANG_DIR="${SGLANG_DIR:-$(dirname "$REPO_DIR")/sglang}"

if command -v mamba >/dev/null 2>&1; then
  CONDA_BIN=mamba
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN=conda
else
  echo "conda/mamba not found. Install miniforge first: https://github.com/conda-forge/miniforge" >&2
  exit 1
fi

source "$($CONDA_BIN info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[install] conda env '$ENV_NAME' exists; reusing"
else
  echo "[install] creating conda env '$ENV_NAME'"
  "$CONDA_BIN" create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"

python -m pip install "pip==$PIP_VER" "wheel==$WHEEL_VER" "setuptools==$SETUPTOOLS_VER"

CU_TAG="cu$(echo "$CUDA_VER" | tr -d .)"
if python -c "import torch" 2>/dev/null; then
  CUR_TORCH="$(python -c 'import torch; print(torch.__version__)')"
  if [[ "$CUR_TORCH" == "${TORCH_VER}+${CU_TAG}" || "$CUR_TORCH" == "$TORCH_VER" ]]; then
    echo "[install] torch: $CUR_TORCH"
  else
    echo "[install] reinstalling torch==$TORCH_VER from $CU_TAG"
    pip install --force-reinstall "torch==$TORCH_VER" --index-url "https://download.pytorch.org/whl/$CU_TAG"
  fi
else
  echo "[install] installing torch==$TORCH_VER from $CU_TAG"
  pip install "torch==$TORCH_VER" --index-url "https://download.pytorch.org/whl/$CU_TAG"
fi

if [[ ! -d "$SGLANG_DIR" ]]; then
  echo "[install] cloning $SGLANG_REPO -> $SGLANG_DIR"
  git clone --branch "$SGLANG_BRANCH" "$SGLANG_REPO" "$SGLANG_DIR"
fi

pushd "$SGLANG_DIR" >/dev/null
if ! git remote get-url rockdu >/dev/null 2>&1; then
  git remote add rockdu "$SGLANG_REPO"
fi
if ! git cat-file -e "$SGLANG_COMMIT^{commit}" 2>/dev/null; then
  git fetch rockdu "$SGLANG_BRANCH"
fi
CUR_SGLANG_COMMIT="$(git rev-parse HEAD)"
if [[ "$CUR_SGLANG_COMMIT" != "$SGLANG_COMMIT" ]]; then
  git checkout --detach "$SGLANG_COMMIT"
fi
pip install -e "python[all]"
popd >/dev/null

cd "$REPO_DIR"
pip install -r requirements.txt
pip install -e . --no-deps

pip install "torch_memory_saver==$TORCH_MEMORY_SAVER_VER" || true

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
else
  echo "[warn] nvidia-smi not found; GPU visibility was not checked"
fi

python -c "import train_diffusion; from miles.utils.arguments import parse_args; from miles.backends.fsdp_utils import FSDPTrainRayActor; import sglang.multimodal_gen; print('miles-diffusion import OK')"
```

The block is idempotent. Re-running it reuses the conda environment, the sglang
checkout, and already installed packages when they match the configured
versions.

## What the Setup Creates

By default, the setup creates this layout:

```bash
/path/to/miles         # this repository
/path/to/sglang        # Rockdu/sglang checked out at the pinned commit
```

It performs the following steps:

1. creates or reuses a conda environment named `miles-diffusion`;
2. installs pinned Python build tooling;
3. installs pinned PyTorch from the selected CUDA wheel index;
4. clones `Rockdu/sglang` and checks out the pinned sglang-diffusion commit;
5. installs sglang in editable mode with `python[all]`;
6. installs miles dependencies from `requirements.txt`;
7. installs miles itself in editable mode;
8. optionally installs `torch_memory_saver`;
9. runs a Python import smoke test.

Activate the environment after installation:

```bash
conda activate miles-diffusion
python -c "import train_diffusion; import sglang.multimodal_gen; print('OK')"
```

If the import command succeeds, the base environment can load the miles
diffusion training entrypoint and the sglang multimodal rollout module.

## Version Pins

The base setup keeps the key environment choices explicit:

| Component | Default pin | Override variable |
| --- | --- | --- |
| Conda env | `miles-diffusion` | `ENV_NAME` |
| Python | `3.11` | `PY_VER` |
| pip | `26.0.1` | `PIP_VER` |
| wheel | `0.45.1` | `WHEEL_VER` |
| setuptools | `82.0.1` | `SETUPTOOLS_VER` |
| PyTorch | `torch==2.9.1` | `TORCH_VER` |
| CUDA wheel index | `cu129` | `CUDA_VER=12.9` |
| sglang repo | `https://github.com/Rockdu/sglang.git` | `SGLANG_REPO` |
| sglang branch | `sglang-diffusion-rollout-test` | `SGLANG_BRANCH` |
| sglang commit | `0372158dd66bc7cb0740c733bd60047db790ec7d` | `SGLANG_COMMIT` |
| torch_memory_saver | `0.0.9` | `TORCH_MEMORY_SAVER_VER` |

Miles package dependencies are pinned in `requirements.txt`, including:

```text
accelerate==1.12.0
datasets==4.4.2
pillow==11.3.0
ray[default]==2.53.0
sglang-router==0.3.0
transformers==5.5.4
wandb==0.23.1
```

The sglang source revision is pinned by commit SHA. This is important because
miles-diffusion relies on the sglang-diffusion fork for multimodal rollout and
weight synchronization; installing from only the branch name is not reproducible
enough for debugging or sharing results.

## Configurable Setup

You can override the defaults before running the setup block:

```bash
export ENV_NAME=miles-diffusion
export PY_VER=3.11
export CUDA_VER=12.9
export TORCH_VER=2.9.1
export SGLANG_DIR=/path/to/sglang
export SGLANG_REPO=https://github.com/Rockdu/sglang.git
export SGLANG_BRANCH=sglang-diffusion-rollout-test
export SGLANG_COMMIT=0372158dd66bc7cb0740c733bd60047db790ec7d
```

Only override `SGLANG_COMMIT` when intentionally testing a new
sglang-diffusion revision.

## Task Dependencies

The base setup intentionally does not install task-specific reward dependencies.
Before running a recipe, install the dependency set required by that task:

- [Task Dependencies](task_dependencies.md)
