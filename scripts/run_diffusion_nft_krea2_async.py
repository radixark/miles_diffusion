"""Async Krea-2-Raw DiffusionNFT training with the OCR reward.

Rollout k+1 is generated while step k trains, on separate GPU pools: 6 FSDP training
GPUs and 2 single-GPU rollout engines, which receive the LoRA adapters over NCCL after
every step. Same NFT shape as run_diffusion_nft_krea2.py: EMA pi_old (--use-ema),
rollout under the EMA (--rollout-weights ema), deterministic ODE rollout
(noise_level=0, sde_type=ode) with no CFG, bf16, 1024px. The loss uses the latest EMA
as pi_old, one EMA step ahead of the EMA that sampled the prefetched batch. Eval runs
on a fixed 64-prompt subset of the OCR test set every 25 steps.

--train-gpus, --rollout-gpus and --rollout-batch-size scale the layout. 3 + 1 GPUs with
3 prompts keeps the full run's per-GPU load: 8 samples per training rank and 24 per
rollout engine.

Needs SGLang with distributed weight updates for diffusion engines
(sgl-project/sglang#42854).

Usage:
    python3 scripts/run_diffusion_nft_krea2_async.py
    python3 scripts/run_diffusion_nft_krea2_async.py --train-gpus 3 --rollout-gpus 1 --rollout-batch-size 3
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "krea/Krea-2-Raw"
DATASET = "rockdu/miles-diffusion-datasets"
SUBSET = "flowgrpo_ocr"
WANDB_PROJECT = "diffusionNFT"
EVAL_SUBSET_SIZE = 64


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 100
    train_gpus: int = 6
    rollout_gpus: int = 2
    rollout_batch_size: int = 6
    data_dir: str = "/root/datasets"
    extra_args: str = ""


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{SUBSET}/**", data_dir=args.data_dir)
    data_dir = f"{local_dir}/{SUBSET}"
    write_eval_subset(data_dir)
    return data_dir


def write_eval_subset(data_dir: str) -> None:
    subset = Path(data_dir) / f"test_{EVAL_SUBSET_SIZE}.jsonl"
    if subset.exists():
        return
    lines = (Path(data_dir) / "test.jsonl").read_text().splitlines()
    picked = sorted(random.Random(0).sample(range(len(lines)), EVAL_SUBSET_SIZE))
    subset.write_text("\n".join(lines[i] for i in picked) + "\n")


def execute(args: ScriptArgs, data_dir: str) -> None:
    run_name = f"diffusion_nft_krea2_ocr_async_{U.create_run_id()}"
    num_gpus = args.train_gpus + args.rollout_gpus

    ckpt_args = f"--hf-checkpoint {MODEL} --save {args.output_dir}/{run_name}/ckpt --save-interval 50 "

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
        f"--rollout-batch-size {args.rollout_batch_size} "
        "--n-samples-per-prompt 8 "
    )

    eval_args = (
        "--diffusion-eval-num-steps 52 --skip-eval-before-train "
        f"--eval-prompt-data ocr_test{EVAL_SUBSET_SIZE} {data_dir}/test_{EVAL_SUBSET_SIZE}.jsonl "
        "--eval-interval 25 "
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

    # Without --colocate, LoRA always pushes only lora_A/lora_B over NCCL and the rollout merges them.
    lora_args = "--use-lora --lora-rank 32 --lora-alpha 64 --lora-init-weights gaussian "

    reward_args = "--rm-type ocr "

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

    perf_args = "--micro-batch-size 4 "

    misc_args = (
        f"--actor-num-gpus-per-node {args.train_gpus} "
        f"--rollout-num-gpus {args.rollout_gpus} "
        "--rollout-num-gpus-per-engine 1 "
        f"--num-gpus-per-node {num_gpus} "
        "--deterministic-mode "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {eval_args} {grpo_args} {ema_args} "
            f"{optimizer_args} {lora_args} {reward_args} {wandb_args} {sglang_args} "
            f"{train_backend_args} {perf_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=num_gpus,
        config=args,
        train_script="train_diffusion_async.py",
        extra_env_vars={"HF_TOKEN": os.environ.get("HF_TOKEN", "")},
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    data_dir = prepare(args)
    execute(args, data_dir)


if __name__ == "__main__":
    typer.run(main)
