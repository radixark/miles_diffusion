"""Synchronous Krea-2-Raw DiffusionNFT training with OCR on 8 colocated GPUs.

Same NFT, LoRA and per-step batch (6 prompts x 8 samples) as run_diffusion_nft_krea2_async.py,
run through train_diffusion.py: FSDP DP=8 shares the GPUs with eight single-GPU sglang
engines, and the LoRA adapters sync over CUDA IPC. Micro-batch 2 with gradient checkpointing,
since the engines take part of every GPU.

Usage:
    python3 scripts/run_diffusion_nft_krea2_8gpu.py
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "krea/Krea-2-Raw"
DATASET = "rockdu/miles-diffusion-datasets"
SUBSET = "flowgrpo_ocr"
WANDB_PROJECT = "diffusionNFT"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 100
    data_dir: str = "/root/datasets"
    extra_args: str = ""


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{SUBSET}/**", data_dir=args.data_dir)
    return f"{local_dir}/{SUBSET}"


def execute(args: ScriptArgs, data_dir: str) -> None:
    run_name = f"diffusion_nft_krea2_ocr_8gpu_{U.create_run_id()}"

    ckpt_args = (
        f"--hf-checkpoint {MODEL} --diffusion-task t2i --save {args.output_dir}/{run_name}/ckpt --save-interval 20 "
    )

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        f"--num-rollout {args.num_rollout} "
        "--num-steps-per-rollout 1 "
        "--diffusion-num-steps 10 "
        "--diffusion-guidance-scale 1.0 "
        "--diffusion-noise-level 0.0 "
        "--diffusion-sde-type ode "
        "--diffusion-height 1024 "
        "--diffusion-width 1024 "
        "--diffusion-debug-mode "
        "--rollout-microgroup-size 1 "
        "--rollout-batch-size 6 --n-samples-per-prompt 8 "
    )

    eval_args = (
        "--diffusion-eval-num-steps 52 --skip-eval-before-train "
        # the full 1018-prompt OCR eval runs once, after 100 rollouts, as in the async recipe
        f"--eval-prompt-data ocr_test {data_dir}/test.jsonl --eval-interval 100 "
    )

    grpo_args = (
        "--loss-type nft "
        "--diffusion-nft-beta 1.0 "
        "--diffusion-nft-num-dropped-timesteps 1 "
        "--advantage-estimator grpo "
        "--globalize-reward-std "
    )

    ema_args = (
        "--use-ema "
        "--rollout-weights ema "
        "--ema-decay-init 0.001 "
        "--ema-decay-ramp 0.001 "
        "--ema-decay-max 0.5 "
        "--ema-decay-flat-steps 0 "
    )

    optimizer_args = "--lr 3e-4 --adam-beta2 0.999 --weight-decay 1e-4 --clip-grad 1.0 "

    lora_args = "--use-lora --lora-ipc-weight-sync --lora-rank 32 --lora-alpha 64 --lora-init-weights gaussian "

    wandb_args = U.get_default_wandb_args(
        __file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=8, wandb_log_image_interval=10
    )

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 8 "
        "--sglang-dit-precision bf16 "
        "--sglang-vae-slicing "
        "--update-weight-buffer-size 2147483648 "
    )

    train_backend_args = (
        "--train-backend fsdp --diffusion-forward-dtype bf16 --micro-batch-size 2 --gradient-checkpointing "
    )

    misc_args = (
        "--actor-num-gpus-per-node 8 "
        "--rollout-num-gpus 8 "
        "--rollout-num-gpus-per-engine 1 "
        "--num-gpus-per-node 8 "
        "--colocate "
        "--deterministic-mode "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {eval_args} {grpo_args} {ema_args} "
            f"{optimizer_args} {lora_args} --rm-type ocr {wandb_args} {sglang_args} "
            f"{train_backend_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=8,
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
