"""Krea-2-Raw DiffusionNFT training (OCR by default, PickScore via --reward).

Same NFT shape as run_diffusion_nft_sd3_pickscore.py: EMA reference (--ref-mode ema),
rollout under pi_old (--ema-rollout-policy ema), deterministic ODE rollout
(noise_level=0, sde_type=ode) with no CFG. Krea-2 specifics: bf16, 1024px, and one
sample per rollout request (the engine's krea2 pipeline has no per-request output
expansion). Rollout debug tensors are collected (--diffusion-debug-mode).

OCR is the default reward: text rendering improves visibly and its accuracy curve is
steep, so both the metric and the wandb images validate the run. It needs no reward
GPU. --reward pickscore switches to the aesthetic direction on one extra GPU.

Smoke mode shrinks the batch for checking the pipeline end to end without a real run.

--train-async switches to one-step async training (train_diffusion_async.py): train and
rollout run on separate GPU pools (2+2) with CPU-staged LoRA weight sync, so the trained
batch is one weight version stale.

Usage:
    python3 scripts/run_diffusion_nft_krea2.py
    python3 scripts/run_diffusion_nft_krea2.py --reward pickscore
    python3 scripts/run_diffusion_nft_krea2.py --train-async
    MILES_SCRIPT_SMOKE=1 python3 scripts/run_diffusion_nft_krea2.py
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "krea/Krea-2-Raw"
DATASET = "rockdu/miles-diffusion-datasets"
WANDB_PROJECT = "diffusionNFT"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 0  # 0 picks the smoke/full default
    data_dir: str = "/root/datasets"
    smoke: bool = False
    reward: str = "ocr"  # ocr | pickscore
    train_async: bool = False
    extra_args: str = ""


def _use_ocr(args: ScriptArgs) -> bool:
    return args.smoke or args.reward == "ocr"


def _subset(args: ScriptArgs) -> str:
    return "flowgrpo_ocr" if _use_ocr(args) else "flowgrpo_pickscore"


def _num_gpus(args: ScriptArgs) -> int:
    return (4 if args.train_async else 2) + (0 if _use_ocr(args) else 1)


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{_subset(args)}/**", data_dir=args.data_dir)
    return f"{local_dir}/{_subset(args)}"


def execute(args: ScriptArgs, data_dir: str) -> None:
    mode = "async" if args.train_async else "colocate"
    run_name = f"diffusion_nft_krea2_{args.reward}_{mode}_{U.create_run_id()}"
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
        "--diffusion-height 1024 "
        "--diffusion-width 1024 "
        "--diffusion-debug-mode "
        "--rollout-microgroup-size 1 "
    ) + (
        "--rollout-batch-size 2 --n-samples-per-prompt 2 "
        if args.smoke
        else "--rollout-batch-size 8 --n-samples-per-prompt 8 "
    )

    eval_args = "--diffusion-eval-num-steps 52 --skip-eval-before-train " + (
        "" if args.smoke else f"--eval-prompt-data {args.reward}_test {data_dir}/test.jsonl --eval-interval 30 "
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

    lora_args = "--use-lora --lora-rank 32 --lora-alpha 64 --lora-init-weights gaussian " + (
        "" if args.train_async else "--lora-ipc-weight-sync "
    )

    reward_args = (
        "--rm-type ocr "
        if _use_ocr(args)
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
        "--sglang-server-concurrency 8 "
        "--sglang-dit-precision bf16 "
        "--sglang-vae-slicing "
        "--update-weight-buffer-size 2147483648 "
    )

    train_backend_args = "--train-backend fsdp --diffusion-forward-dtype bf16 "

    perf_args = "--gradient-checkpointing " + ("--micro-batch-size 1 " if args.smoke else "--micro-batch-size 2 ")

    misc_args = (
        "--actor-num-gpus-per-node 2 "
        "--rollout-num-gpus 2 "
        "--rollout-num-gpus-per-engine 1 "
        f"--num-gpus-per-node {_num_gpus(args)} "
        + ("" if args.train_async else "--colocate ")
        + "--deterministic-mode "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {eval_args} {grpo_args} {ema_args} "
            f"{optimizer_args} {lora_args} {reward_args} {wandb_args} {sglang_args} "
            f"{train_backend_args} {perf_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=_num_gpus(args),
        train_script="train_diffusion_async.py" if args.train_async else "train_diffusion.py",
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
