#!/usr/bin/env bash
# Optional dependencies stay in dedicated environments below the selected work directory.
set -euo pipefail
role="${1:?Usage: bash plugins/libero/setup.sh train|eval|serve WORK_DIR}"
work="${2:?Specify the directory for environments, dependency checkouts and caches}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
plugin="$repo/plugins/libero"
command -v uv >/dev/null || { echo 'Install uv first; see docs/models/pi05/pi05.md.' >&2; exit 1; }
mkdir -p "$work"
work="$(cd "$work" && pwd)"
mkdir -p "$work/envs" "$work/third_party"

checkout() {
    local url="$1" revision="$2" destination="$3"
    if [[ ! -e "$destination" ]]; then
        git init -q "$destination"
        git -C "$destination" remote add origin "$url"
        git -C "$destination" fetch --depth 1 origin "$revision"
        git -C "$destination" checkout --detach FETCH_HEAD
    fi
    [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]] || {
        echo "Unexpected revision in $destination; use a new work directory." >&2; exit 1;
    }
}

case "$role" in
train)
    checkout https://github.com/Physical-Intelligence/openpi.git 215abfb217dbac7d5f1273282331b9b1866c0479 "$work/third_party/openpi"
    checkout https://github.com/huggingface/lerobot.git 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 "$work/third_party/lerobot"
    [[ -x "$work/envs/train/bin/python" ]] || uv venv --python 3.12 "$work/envs/train"
    py="$work/envs/train/bin/python"
    uv pip install --python "$py" -r "$plugin/requirements-train.txt"
    # OpenPI/LeRobot supply data utilities, not the training model. Their default
    # model dependency pins conflict with Miles; install only the selected profile above.
    uv pip install --python "$py" --no-deps -e "$repo" -e "$plugin" \
        -e "$work/third_party/openpi" -e "$work/third_party/openpi/packages/openpi-client" \
        -e "$work/third_party/lerobot"
    ;;
eval)
    py="$work/envs/train/bin/python"
    [[ -x "$py" ]] || { echo 'Run setup.sh train first.' >&2; exit 1; }
    checkout https://github.com/Lifelong-Robot-Learning/LIBERO.git 8f1084e3132a39270c3a13ebe37270a43ece2a01 "$work/third_party/LIBERO"
    uv pip install --python "$py" -r "$plugin/requirements-eval.txt"
    # eval_libero uses the explicit checkout to resolve LIBERO's namespace and assets.
    ;;
serve)
    checkout https://github.com/sgl-project/sglang.git 6393e66b8649b4e6e22f2d94a933e69059192900 "$work/third_party/sglang"
    [[ -x "$work/envs/serve/bin/python" ]] || uv venv --python 3.12 "$work/envs/serve"
    py="$work/envs/serve/bin/python"
    uv pip install --python "$py" -r "$plugin/requirements-serve.txt"
    # pi05's diffusion endpoint does not use the LLM Rust extensions or LLM kernels.
    SGLANG_BUILD_RUST_EXTS=none uv pip install --python "$py" --no-deps --no-build-isolation \
        -e "$work/third_party/sglang/python"
    ;;
*) echo "Unknown role: $role (expected train, eval or serve)" >&2; exit 1 ;;
esac
