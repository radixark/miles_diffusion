# Source after setup: source plugins/libero/env.sh /absolute/path/to/work
export PI05_WORK="$(cd "${1:?Pass the work directory used by setup.sh}" && pwd)"
export HF_HOME="$PI05_WORK/cache/huggingface"
export HF_LEROBOT_HOME="$PI05_WORK/datasets"
export OPENPI_DATA_HOME="$PI05_WORK/cache/openpi"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
