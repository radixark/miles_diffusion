"""SD3.5-medium OCR GRPO through the sglang-diffusion /rollout/generate path.

2-GPU colocate: FSDP DP=2 plus 2 rollout engines time-multiplexed on the same GPUs.

SD3.5 is gated, so HF_TOKEN must be set even when the weights are cached — sglang still
fetches model_index.json from the hub at startup.

Usage:
    python3 scripts/run_diffusion_grpo_sd3_ocr_sglang.py
    MILES_SCRIPT_DEBUG_ALIGNMENT=1 python3 scripts/run_diffusion_grpo_sd3_ocr_sglang.py
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "stabilityai/stable-diffusion-3.5-medium"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_ocr"
WANDB_PROJECT = "miles-diffusion-grpo"

# master_sglang carries native SD3 /rollout/generate support; prepending it to PYTHONPATH
# shadows the editable install at /sgl-workspace/sglang.
MASTER_SGLANG_PYTHON = "/sgl-workspace/master_sglang/sglang/python"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 600
    data_dir: str = "/root/datasets"
    debug_alignment: bool = False
    extra_args: str = ""


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{DATASET_SUBSET}/**", data_dir=args.data_dir)
    return f"{local_dir}/{DATASET_SUBSET}"


def execute(args: ScriptArgs, data_dir: str) -> None:
    run_name = f"diffusion_grpo_sd3_ocr_sglang_{U.create_run_id()}"

    ckpt_args = f"--hf-checkpoint {MODEL} --save {args.output_dir}/{run_name}/ckpt "

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 16 "
        f"--num-rollout {args.num_rollout} "
        "--global-batch-size 64 "
        "--rollout-microgroup-size 8 "
        "--train-dp-split-mode stride "
        "--diffusion-num-steps 10 "
        "--diffusion-guidance-scale 4.5 "
        "--diffusion-noise-level 0.7 "
        "--diffusion-height 512 "
        "--diffusion-width 512 "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window "
        "--diffusion-num-sde-steps 10 "
        "--diffusion-sde-window-range 0,10 "
    )

    eval_args = "--diffusion-eval-num-steps 40 "

    grpo_args = (
        "--advantage-estimator grpo --globalize-reward-std --diffusion-clip-range 1e-4 --diffusion-kl-beta 0.04 "
    )

    optimizer_args = "--lr 3e-4 --adam-beta2 0.999 --weight-decay 1e-4 "

    lora_args = "--use-lora --lora-ipc-weight-sync --lora-rank 32 --lora-alpha 64 --lora-init-weights gaussian "

    reward_args = "--rm-type ocr "

    wandb_args = U.get_default_wandb_args(
        __file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=8, wandb_log_image_interval=10
    )

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 8 "
        "--sglang-dit-precision fp16 "
        "--sglang-vae-slicing "
        "--update-weight-buffer-size 2147483648 "
    )

    train_backend_args = "--train-backend fsdp --diffusion-forward-dtype fp16 "

    perf_args = "--gradient-checkpointing --micro-batch-size-sample 16 --micro-batch-size-tstep 5 "

    misc_args = (
        "--actor-num-gpus-per-node 2 "
        "--rollout-num-gpus 2 "
        "--rollout-num-gpus-per-engine 1 "
        "--num-gpus-per-node 2 "
        "--colocate "
        "--deterministic-mode "
    ) + ("--diffusion-debug-mode --debug-skip-optimizer-step " if args.debug_alignment else "")

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {eval_args} {grpo_args} {optimizer_args} "
            f"{lora_args} {reward_args} {wandb_args} {sglang_args} {train_backend_args} {perf_args} "
            f"{misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=2,
        config=args,
        extra_env_vars={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": MASTER_SGLANG_PYTHON,
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            **({"MILES_VERIFY_WEIGHT_SYNC": "1"} if args.debug_alignment else {}),
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    data_dir = prepare(args)
    execute(args, data_dir)


if __name__ == "__main__":
    typer.run(main)
