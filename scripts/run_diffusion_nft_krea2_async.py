"""Async Krea-2-Raw DiffusionNFT training with OCR on 6 training + 2 rollout GPUs.

Same NFT shape as run_diffusion_nft_krea2.py, run through train_diffusion_async.py: each
batch is sampled under the EMA while the previous batch trains, and the LoRA adapters
sync to the separate rollout GPUs over NCCL. One prompt x 8 samples per training GPU,
micro-batch 4, no gradient checkpointing.

Usage:
    python3 scripts/run_diffusion_nft_krea2_async.py
    python3 scripts/run_diffusion_nft_krea2_async.py --four-gpu-ci
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
    # CI cap: half of every GPU pool and of the batch, 3 training + 1 rollout GPUs with 3 prompts per step
    four_gpu_ci: bool = False


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{SUBSET}/**", data_dir=args.data_dir)
    return f"{local_dir}/{SUBSET}"


def execute(args: ScriptArgs, data_dir: str) -> None:
    run_name = f"diffusion_nft_krea2_ocr_async_{U.create_run_id()}"
    num_train_gpus, num_rollout_gpus = (3, 1) if args.four_gpu_ci else (6, 2)
    num_gpus = num_train_gpus + num_rollout_gpus

    ckpt_args = f"--hf-checkpoint {MODEL} --save {args.output_dir}/{run_name}/ckpt --save-interval 20 "

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
        f"--rollout-batch-size {num_train_gpus} --n-samples-per-prompt 8 "
    )

    eval_args = (
        "--diffusion-eval-num-steps 52 --skip-eval-before-train "
        # the 1018-prompt OCR eval takes over an hour on 2 rollout GPUs, so it runs once, after 100 rollouts
        f"--eval-prompt-data ocr_test {data_dir}/test.jsonl --eval-interval 100 "
    )

    grpo_args = (
        "--loss-type nft "
        "--diffusion-nft-beta 1.0 "
        "--diffusion-nft-timestep-fraction 0.99 "
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

    # without --colocate the adapters sync over NCCL and the rollout merges them
    lora_args = "--use-lora --lora-rank 32 --lora-alpha 64 --lora-init-weights gaussian "

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

    train_backend_args = "--train-backend fsdp --diffusion-forward-dtype bf16 --micro-batch-size 4 "

    misc_args = (
        f"--actor-num-gpus-per-node {num_train_gpus} "
        f"--rollout-num-gpus {num_rollout_gpus} "
        "--rollout-num-gpus-per-engine 1 "
        f"--num-gpus-per-node {num_gpus} "
        "--deterministic-mode "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {eval_args} {grpo_args} {ema_args} "
            f"{optimizer_args} {lora_args} --rm-type ocr {wandb_args} {sglang_args} "
            f"{train_backend_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=num_gpus,
        train_script="train_diffusion_async.py",
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
