"""SD3.5-medium DiffusionNFT training with PickScore.

Batch shape follows the UniRL 100-rollout override: 8 prompts x 8 samples, micro=4, on 2
train GPUs plus a dedicated reward GPU.

NFT needs a reference model, supplied here by the EMA copy (--ref-mode ema), and samples
under pi_old via --ema-rollout-policy ema. noise_level=0 with sde_type=ode makes the
rollout deterministic, which NFT requires.

Smoke mode swaps in the small OCR dataset and a tiny batch, for checking the pipeline end
to end without a real run.

Usage:
    python3 scripts/run_diffusion_nft_sd3_pickscore.py
    MILES_SCRIPT_SMOKE=1 python3 scripts/run_diffusion_nft_sd3_pickscore.py
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "stabilityai/stable-diffusion-3.5-medium"
DATASET = "rockdu/miles-diffusion-datasets"
WANDB_PROJECT = "miles-diffusion-nft"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 0  # 0 picks the smoke/full default
    data_dir: str = "/root/datasets"
    smoke: bool = False
    extra_args: str = ""


def _subset(args: ScriptArgs) -> str:
    return "flowgrpo_ocr" if args.smoke else "flowgrpo_pickscore"


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{_subset(args)}/**", data_dir=args.data_dir)
    return f"{local_dir}/{_subset(args)}"


def execute(args: ScriptArgs, data_dir: str) -> None:
    run_name = f"diffusion_nft_sd3_pickscore_{U.create_run_id()}"
    num_rollout = args.num_rollout or (1 if args.smoke else 100)

    ckpt_args = f"--hf-checkpoint {MODEL} --save {args.output_dir}/{run_name}/ckpt --save-interval 20 "

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        f"--num-rollout {num_rollout} "
        "--num-steps-per-rollout 1 "
        "--diffusion-num-steps 10 "
        "--diffusion-guidance-scale 1.0 "
        "--diffusion-noise-level 0.0 "
        "--diffusion-sde-type ode "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.ode_and_return_last "
        "--diffusion-height 512 "
        "--diffusion-width 512 "
    ) + (
        "--rollout-batch-size 2 --n-samples-per-prompt 2 --rollout-microgroup-size 2 "
        if args.smoke
        else "--rollout-batch-size 8 --n-samples-per-prompt 8 --rollout-microgroup-size 8 "
    )

    eval_args = "--diffusion-eval-num-steps 50 --skip-eval-before-train " + (
        "" if args.smoke else f"--eval-prompt-data pickscore_test {data_dir}/test.jsonl --eval-interval 30 "
    )

    grpo_args = (
        "--loss-type nft "
        "--diffusion-nft-beta 1.0 "
        "--diffusion-nft-timestep-fraction 0.99 "
        "--advantage-estimator grpo "
        "--globalize-reward-std "
    )

    ema_args = (
        "--ref-mode ema "
        "--use-ema "
        "--ema-rollout-policy ema "
        "--ema-decay-init 0.001 "
        "--ema-decay-ramp 0.001 "
        "--ema-decay-max 0.5 "
        "--ema-decay-flat-steps 0 "
    )

    optimizer_args = "--lr 3e-4 --adam-beta2 0.999 --weight-decay 1e-4 --clip-grad 1.0 "

    lora_args = "--use-lora --lora-ipc-weight-sync --lora-rank 32 --lora-alpha 64 --lora-init-weights gaussian "

    reward_args = (
        "--rm-type ocr "
        if args.smoke
        else (
            "--rm-type pickscore "
            "--pickscore-num-workers 1 "
            "--pickscore-num-gpus-per-worker 1.0 "
            "--pickscore-batch-size 8 "
            "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
            "--pickscore-model-path yuvalkirstain/PickScore_v1 "
        )
    )

    wandb_args = U.get_default_wandb_args(
        __file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=8, wandb_log_image_interval=10
    )

    sglang_args = (
        "--use-miles-router "
        "--rollout-fetch-in-parser "
        "--rollout-parser-num-workers 16 "
        "--sglang-server-concurrency 8 "
        "--sglang-dit-precision fp16 "
        "--sglang-vae-slicing "
        "--update-weight-buffer-size 2147483648 "
    )

    train_backend_args = "--train-backend fsdp --diffusion-forward-dtype fp16 "

    perf_args = "--gradient-checkpointing " + ("--micro-batch-size 2 " if args.smoke else "--micro-batch-size 4 ")

    misc_args = (
        "--actor-num-gpus-per-node 2 "
        "--rollout-num-gpus 2 "
        "--rollout-num-gpus-per-engine 1 "
        f"--num-gpus-per-node {2 if args.smoke else 3} "
        "--colocate "
        "--deterministic-mode "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {eval_args} {grpo_args} {ema_args} "
            f"{optimizer_args} {lora_args} {reward_args} {wandb_args} {sglang_args} "
            f"{train_backend_args} {perf_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=2 if args.smoke else 3,
        config=args,
        extra_env_vars={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    data_dir = prepare(args)
    execute(args, data_dir)


if __name__ == "__main__":
    typer.run(main)
