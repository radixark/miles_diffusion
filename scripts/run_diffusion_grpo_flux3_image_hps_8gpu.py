"""FLUX.3 image full-finetune Flow-GRPO with the HPSv2.1 reward, fully colocated on 8 GPUs.

Follows the Wan2.2 full-finetune GRPO recipe: no LoRA, fp32 master and reduce, bf16 forward,
lr 1e-5; 10 sampling steps with Flow-SDE noise_level 0.7 on one SDE step per rollout, drawn
from the early (high-noise) steps 1,2; old log-probs from the rollout; clip 1e-4, no KL.
Rollout and training both run CFG 3.5, so the policy is the guided velocity.

Layout: train, rollout and HPS reward share the same 8 GPUs (--colocate --hps-reward-colocate,
one HPS worker per rollout engine). Full weights sync over IPC each rollout.

Usage:
    python3 scripts/run_diffusion_grpo_flux3_image_hps_8gpu.py
    python3 scripts/run_diffusion_grpo_flux3_image_hps_8gpu.py --smoke
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

DATASET = "rockdu/miles-diffusion-datasets"
SUBSET = "hpdv2"
NUM_GPUS = 8
# Requests in flight per engine; each engine runs one at a time, the rest hide response decoding.
ROLLOUT_CONCURRENCY = 4


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model_path: str = "black-forest-labs/flux-3-image-klein-base"
    num_rollout: int = 0  # 0 picks the smoke/full default
    data_dir: str = "/root/datasets"
    smoke: bool = False
    extra_args: str = ""


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{SUBSET}/**", data_dir=args.data_dir)
    return f"{local_dir}/{SUBSET}"


def execute(args: ScriptArgs, data_dir: str) -> None:
    run_name = f"diffusion_grpo_flux3_image_hps_8gpu_{U.create_run_id()}"
    num_rollout = args.num_rollout or (1 if args.smoke else 200)

    ckpt_args = (
        f"--hf-checkpoint {args.model_path} --diffusion-task t2i "
        f"--save {args.output_dir}/{run_name}/ckpt --save-interval 50 "
    )

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        f"--num-rollout {num_rollout} "
        "--num-steps-per-rollout 1 "
        "--diffusion-num-steps 10 "
        "--diffusion-guidance-scale 3.5 "
        "--diffusion-noise-level 0.7 "
        "--diffusion-height 1024 "
        "--diffusion-width 1024 "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_random_choice "
        "--diffusion-num-sde-steps 1 "
        "--diffusion-sde-candidate-steps 1,2 "
        "--rollout-microgroup-size 1 "
    ) + (
        "--rollout-batch-size 4 --n-samples-per-prompt 4 "
        if args.smoke
        else "--rollout-batch-size 16 --n-samples-per-prompt 16 "
    )

    # No eval: the train reward tracks progress, and eval rollouts would compete for the colocated GPUs.
    eval_args = "--skip-eval-before-train "

    grpo_args = (
        "--advantage-estimator grpo "
        "--globalize-reward-std "
        "--diffusion-clip-range 1e-4 "
    )

    optimizer_args = "--lr 1e-5 --adam-beta2 0.999 --weight-decay 1e-4 "

    reward_args = (
        "--rm-type hps "
        "--hps-version v2.1 "
        "--hps-reward-colocate "
        f"--hps-num-workers {NUM_GPUS} "
        "--hps-batch-size 8 "
    )

    # Parsers fetch and decode responses in parallel; --rollout-fetch-in-parser needs one parser per
    # concurrency slot (concurrency x engines).
    sglang_args = (
        "--use-miles-router "
        f"--sglang-server-concurrency {ROLLOUT_CONCURRENCY} "
        "--rollout-fetch-in-parser "
        f"--rollout-parser-num-workers {ROLLOUT_CONCURRENCY * NUM_GPUS} "
        "--sglang-pipeline-class-name Flux3ImagePipeline "
        "--update-weight-buffer-size 4294967296 "
    )

    train_backend_args = (
        "--train-backend fsdp --diffusion-forward-dtype bf16 --fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 "
        "--fsdp-attention-backend _flash_3 "
    )

    perf_args = "--gradient-checkpointing --micro-batch-size 1 "

    misc_args = (
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        f"--rollout-num-gpus {NUM_GPUS} "
        "--rollout-num-gpus-per-engine 1 "
        f"--num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        "--deterministic-mode "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {eval_args} {grpo_args} "
            f"{optimizer_args} {reward_args} {sglang_args} "
            f"{train_backend_args} {perf_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=NUM_GPUS,
        config=args,
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    data_dir = prepare(args)
    execute(args, data_dir)


if __name__ == "__main__":
    typer.run(main)
