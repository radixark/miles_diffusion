"""Wan2.2-T2V-A14B full-finetune video GRPO with PickScore, 2 nodes x 8 GPU + 1 reward GPU.

Multi-node: this script submits into an EXISTING ray cluster (16 colocated train/rollout GPUs
plus one reward GPU on a separate node) — see docs/user-guide/launch-script.md (Multi-node training) for the bring-up, then run with
MILES_SCRIPT_EXTERNAL_RAY=1. The reward workers are default-scheduled, so they land on the only
GPU the engines do not occupy: the reward node.

Full finetune (no LoRA): fp32 master/reduce, bf16 forward, lr 1e-5. Train<->rollout output diff is
exactly 0 with `--rollout-patch-group wan` + `--sglang-attention-backend torch_sdpa` (measured
sustained over 200-rollout production runs; the importance ratio is exactly 1).

Settled sampling knobs, each validated against the alternatives:
  flow_shift 5.0     3.0 (the diffusers/UniPC value) under-converges stylized prompts on the
                     rollout Euler at 10 steps; the official-repo 12.0 flattens style. 5.0 splits
                     the dual-expert steps 5-high/5-low.
  noise_level 0.7    the engine's own rollout default (FlowGRPO's 0.9 also works).
  negative prompt    unset -> the engine's per-model default (Wan official Chinese prompt).
  reward             PickScore mean over ALL frames (no --pickscore-num-frames).

Known limitation: with --diffusion-sde-candidate-steps 1,2 the SDE training step sits at
sigma >= 0.875 under any shift, so only `transformer` (high-noise expert) receives gradient;
`transformer_2` stays at the checkpoint.

Per rollout: 24 prompts x 16 samples = 384 videos, microgroups of 4 per engine request.
Engine sp_degree=2 (2 GPUs per engine, 8 engines); trainer dp_replicate=2 x ulysses sp=4.
The 4 GiB weight-sync buffer takes the full-finetune sync from 92 s to ~15 s (latency-bound
on per-bucket round-trips, not bandwidth).

Usage (after docs/user-guide/launch-script.md (Multi-node training) bring-up, on the head node):
    MILES_SCRIPT_EXTERNAL_RAY=1 python3 scripts/run_diffusion_grpo_wan22_pickscore_17gpu_multinode.py
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_pickscore"
WANDB_PROJECT = "miles-diffusion-grpo"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 10000
    data_dir: str = "/root/datasets"
    extra_args: str = ""
    # CI cap: same recipe on 1 node x 4 GPUs — batch /4, FSDP shard 4 (SP kept, so dp_replicate 1),
    # two sp2 rollout engines, reward colocated one worker per GPU. Runs on the launcher's own
    # local ray instead of an external cluster.
    four_gpu_ci: bool = False


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{DATASET_SUBSET}/**", data_dir=args.data_dir)
    return f"{local_dir}/{DATASET_SUBSET}"


def execute(args: ScriptArgs, data_dir: str) -> None:
    assert args.four_gpu_ci or U.get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY"), (
        "This recipe needs the 17-GPU ray cluster from docs/user-guide/launch-script.md (Multi-node training); "
        "bring it up first, then run with MILES_SCRIPT_EXTERNAL_RAY=1."
    )

    run_name = f"diffusion_grpo_wan22_pickscore_17gpu_{U.create_run_id()}"

    ckpt_args = f"--hf-checkpoint {MODEL} "
    if not args.four_gpu_ci:
        ckpt_args += f"--save {args.output_dir}/{run_name}/ckpt --save-interval 100 "

    rollout_batch_size = 6 if args.four_gpu_ci else 24
    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        f"--rollout-batch-size {rollout_batch_size} "
        "--n-samples-per-prompt 16 "
        f"--num-rollout {args.num_rollout} "
        "--num-steps-per-rollout 1 "
        "--rollout-microgroup-size 4 "
        "--micro-batch-size-sample 4 "
        "--micro-batch-size-tstep 1 "
        "--diffusion-num-steps 10 "
        "--diffusion-output-num-frames 5 "
        "--diffusion-guidance-scale 4.0 "
        "--diffusion-guidance-scale-2 3.0 "
        "--diffusion-noise-level 0.7 "
        "--diffusion-height 480 "
        "--diffusion-width 832 "
        "--diffusion-flow-shift 5.0 "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_random_choice "
        "--diffusion-num-sde-steps 1 "
        "--diffusion-sde-candidate-steps 1,2 "
        "--diffusion-recompute-old-log-prob "
        "--rollout-patch-group wan "
        "--sglang-attention-backend torch_sdpa "
    )

    eval_args = (
        ""
        if args.four_gpu_ci
        else f"--eval-prompt-data pickscore_test {data_dir}/test.jsonl "
        "--eval-interval 100 "
        "--diffusion-eval-num-steps 28 "
        "--skip-eval-before-train "
    )

    grpo_args = "--advantage-estimator grpo --diffusion-clip-range 1e-4 "

    optimizer_args = "--lr 1e-5 --adam-beta2 0.999 --weight-decay 1e-4 "

    reward_placement = (
        "--colocate-reward --pickscore-num-workers 4 "
        if args.four_gpu_ci
        else "--pickscore-num-workers 4 --pickscore-num-gpus-per-worker 0.25 "
    )
    reward_args = (
        "--rm-type pickscore "
        f"{reward_placement}"
        "--pickscore-batch-size 8 "
        "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
        "--pickscore-model-path yuvalkirstain/PickScore_v1 "
    )

    wandb_args = U.get_default_wandb_args(
        __file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=4, wandb_log_image_interval=5
    )

    sglang_args = (
        "--use-miles-router "
        "--rollout-fetch-in-parser "
        "--sglang-server-concurrency 8 "
        "--miles-router-health-check-failure-threshold 30 "
        "--update-weight-buffer-size 4294967296 "
        "--sglang-vae-cpu-offload "
    )

    train_backend_args = (
        "--train-backend fsdp --fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 --diffusion-forward-dtype bf16 "
        "--update-weight-target-module transformer,transformer_2 "
        "--gradient-checkpointing "
    )

    topology_args = (
        (
            "--actor-num-nodes 1 "
            "--actor-num-gpus-per-node 4 "
            "--num-gpus-per-node 4 "
            "--rollout-num-gpus 4 "
            "--rollout-num-gpus-per-engine 2 "
            "--rollout-parser-num-workers 16 "
            "--dp-replicate-size 1 "
        )
        if args.four_gpu_ci
        else (
            "--actor-num-nodes 2 "
            "--actor-num-gpus-per-node 8 "
            "--num-gpus-per-node 8 "
            "--rollout-num-gpus 16 "
            "--rollout-num-gpus-per-engine 2 "
            "--rollout-parser-num-workers 64 "
            "--dp-replicate-size 2 "
        )
    ) + (
        "--sglang-tp-size 1 "
        "--sglang-sp-degree 2 "
        "--sequence-parallel-size 4 "
        "--ulysses-degree 4 "
        "--colocate "
        "--deterministic-mode "
        "--diffusion-debug-mode "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {eval_args} {grpo_args} {optimizer_args} "
            f"{reward_args} {wandb_args} {sglang_args} {train_backend_args} "
            f"{topology_args} {args.extra_args}"
        ),
        num_gpus_per_node=4 if args.four_gpu_ci else 8,
        config=args,
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"},
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    data_dir = prepare(args)
    execute(args, data_dir)


if __name__ == "__main__":
    typer.run(main)
