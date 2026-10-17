"""FLUX.3 image DiffusionNFT training with the PickScore reward, fully colocated on 8 GPUs.

Same NFT shape as run_diffusion_nft_krea2.py: EMA pi_old (--use-ema), rollout under pi_old
(--rollout-weights ema), deterministic ODE rollout (noise_level=0, sde_type=ode), LoRA synced
over IPC. NFT trains the unguided (conditional-only) velocity, so the rollout samples without
CFG as well: at a scale of 1 FLUX.3's engine still runs the whole CFG pair, and the flux3_image
rollout patch group (--rollout-patch-group) runs only the conditional half instead.
One sample per rollout request. The checkpoint is gated on the Hub (needs HF_TOKEN);
--model-path also takes a local package directory.

Layout: train, rollout and PickScore reward all share the same 8 GPUs (--colocate
--pickscore-reward-colocate, one PickScore worker per rollout engine).

Usage:
    python3 scripts/run_diffusion_nft_flux3_image_pickscore_8gpu.py
    python3 scripts/run_diffusion_nft_flux3_image_pickscore_8gpu.py --smoke
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

DATASET = "rockdu/miles-diffusion-datasets"
SUBSET = "flowgrpo_pickscore"
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
    run_name = f"diffusion_nft_flux3_image_pickscore_8gpu_{U.create_run_id()}"
    num_rollout = args.num_rollout or (1 if args.smoke else 100)

    ckpt_args = (
        f"--hf-checkpoint {args.model_path} --diffusion-task t2i "
        f"--save {args.output_dir}/{run_name}/ckpt --save-interval 20 "
    )

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        f"--num-rollout {num_rollout} "
        "--num-steps-per-rollout 1 "
        "--diffusion-num-steps 16 "
        "--diffusion-guidance-scale 1.0 "
        "--diffusion-noise-level 0.0 "
        "--diffusion-sde-type ode "
        "--diffusion-height 1024 "
        "--diffusion-width 1024 "
        "--rollout-microgroup-size 1 "
    ) + (
        "--rollout-batch-size 4 --n-samples-per-prompt 2 "
        if args.smoke
        else "--rollout-batch-size 8 --n-samples-per-prompt 8 "
    )

    # No eval: the train reward tracks progress, and eval rollouts would compete for the colocated GPUs.
    eval_args = "--skip-eval-before-train "

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

    # dynamic keeps LoRA unmerged in the engine, the same as the trainer's forward; a merge drifts bitwise.
    lora_args = (
        "--use-lora --lora-ipc-weight-sync --lora-rank 32 --lora-alpha 64 --lora-init-weights gaussian "
        "--sglang-lora-merge-mode dynamic "
    )

    reward_args = (
        "--rm-type pickscore "
        "--pickscore-reward-colocate "
        f"--pickscore-num-workers {NUM_GPUS} "
        "--pickscore-batch-size 8 "
        "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
        "--pickscore-model-path yuvalkirstain/PickScore_v1 "
    )

    # Parsers fetch and decode responses in parallel; --rollout-fetch-in-parser needs one parser per
    # concurrency slot (concurrency x engines).
    sglang_args = (
        "--use-miles-router "
        f"--sglang-server-concurrency {ROLLOUT_CONCURRENCY} "
        "--rollout-fetch-in-parser "
        f"--rollout-parser-num-workers {ROLLOUT_CONCURRENCY * NUM_GPUS} "
        "--sglang-pipeline-class-name Flux3ImagePipeline "
        "--rollout-patch-group flux3_image "
        "--update-weight-buffer-size 2147483648 "
    )

    train_backend_args = (
        "--train-backend fsdp --diffusion-forward-dtype bf16 --fsdp-master-dtype fp32 --fsdp-attention-backend _flash_3 "
    )

    perf_args = "--gradient-checkpointing --micro-batch-size 1 "

    misc_args = (
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        f"--rollout-num-gpus {NUM_GPUS} "
        "--rollout-num-gpus-per-engine 1 "
        f"--num-gpus-per-node {NUM_GPUS} "
        "--colocate "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {eval_args} {grpo_args} {ema_args} "
            f"{optimizer_args} {lora_args} {reward_args} {sglang_args} "
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
