"""Train Wan2.2-TI2V-5B with Flow-GRPO, FSDP, and LoRA.

The recipe runs SGLang rollout and the FSDP trainer on the same GPUs in
colocated mode. PickScore evaluates uniformly sampled video frames, GRPO
normalizes rewards within each prompt group, and only LoRA parameters are
optimized and synchronized back to the rollout engine.

Prerequisites:
    - Install the project environment and activate it.
    - Make the model available as a Hugging Face model ID or local Diffusers
      directory. The default is ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``.
    - Make the PickScore processor and model available from Hugging Face, or
      override their paths through ``--extra-args`` when using local copies.
    - Use at least two samples per prompt so GRPO can compare generations.

Dataset format:
    The input is JSONL. Each row contains a prompt under ``input`` and a
    conditioning image under ``metadata.image_path``. Image paths should be
    absolute paths visible to every Ray worker.

    {"input": "A camera moves around the subject.",
     "metadata": {"image_path": "/data/images/frame.jpg"}}

Two-GPU training with the default Hugging Face model:
    python3 scripts/run_diffusion_grpo_wan22_ti2v.py \
        --data-jsonl /data/train.jsonl \
        --num-gpus 2 \
        --cuda-visible-devices 0,1

Training from a local model directory:
    python3 scripts/run_diffusion_grpo_wan22_ti2v.py \
        --model /models/Wan2.2-TI2V-5B-Diffusers \
        --data-jsonl /data/train.jsonl \
        --num-gpus 2 \
        --cuda-visible-devices 0,1

Enable periodic evaluation with a separate JSONL file:
    python3 scripts/run_diffusion_grpo_wan22_ti2v.py \
        --data-jsonl /data/train.jsonl \
        --eval-data-jsonl /data/validation.jsonl \
        --eval-interval 20

Lower-cost configuration for checking a new installation:
    python3 scripts/run_diffusion_grpo_wan22_ti2v.py \
        --model /models/Wan2.2-TI2V-5B-Diffusers \
        --data-jsonl /data/train.jsonl \
        --num-gpus 2 \
        --cuda-visible-devices 0,1 \
        --num-rollout 1 \
        --rollout-batch-size 1 \
        --n-samples-per-prompt 2 \
        --num-steps-per-rollout 1 \
        --height 256 --width 448 --num-frames 5 \
        --num-diffusion-steps 3 \
        --lora-rank 4 --lora-alpha 4 \
        --extra-args "--diffusion-guidance-scale 1.0 --diffusion-num-sde-steps 1"

Operational notes:
    - ``--num-gpus`` controls both FSDP world size and SGLang tensor parallel
      size. Use two or more GPUs for actual parameter sharding.
    - Checkpoints are written below ``--output-dir`` every ``--save-interval``
      rollout iterations and once at the end.
    - Set ``WANDB_API_KEY`` to enable W&B; otherwise tracking is skipped.
    - ``--extra-args`` appends native ``train_diffusion.py`` options and can be
      used for scheduler, reward-model, resume, or performance overrides.
    - Run the script with ``--help`` for all recipe-level options.
"""

import shlex
from dataclasses import dataclass
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U


MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WANDB_PROJECT = "miles-diffusion-grpo"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model: str = MODEL
    data_jsonl: str = ""
    eval_data_jsonl: str = ""
    input_key: str = "input"
    num_gpus: int = 2
    num_rollout: int = 1000
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 8
    num_steps_per_rollout: int = 2
    save_interval: int = 10
    eval_interval: int = 20
    height: int = 480
    width: int = 832
    num_frames: int = 49
    num_diffusion_steps: int = 20
    lora_rank: int = 64
    lora_alpha: int = 128
    extra_args: str = ""


def _validate_args(args: ScriptArgs) -> None:
    if not args.data_jsonl:
        raise SystemExit("set --data-jsonl to a JSONL file containing a prompt and metadata.image_path")
    if not Path(args.data_jsonl).is_file():
        raise FileNotFoundError(args.data_jsonl)
    if args.eval_data_jsonl and not Path(args.eval_data_jsonl).is_file():
        raise FileNotFoundError(args.eval_data_jsonl)
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be at least 1")


def execute(args: ScriptArgs) -> None:
    _validate_args(args)
    run_name = f"diffusion_grpo_wan22_ti2v_{U.create_run_id()}"
    model = shlex.quote(args.model)
    data_jsonl = shlex.quote(args.data_jsonl)
    input_key = shlex.quote(args.input_key)
    save_dir = shlex.quote(f"{args.output_dir}/{run_name}/ckpt")

    ckpt_args = (
        f"--hf-checkpoint {model} "
        "--diffusion-model-family wan2_2_ti2v "
        f"--save {save_dir} "
        f"--save-interval {args.save_interval} "
    )

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_jsonl} "
        f"--input-key {input_key} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--num-steps-per-rollout {args.num_steps_per_rollout} "
        f"--num-rollout {args.num_rollout} "
        "--rollout-microgroup-size 1 "
        "--micro-batch-size-sample 1 "
        "--micro-batch-size-tstep 1 "
        "--diffusion-train-iter-order sample_major "
    )

    diffusion_args = (
        f"--diffusion-num-steps {args.num_diffusion_steps} "
        f"--diffusion-output-num-frames {args.num_frames} "
        f"--diffusion-height {args.height} "
        f"--diffusion-width {args.width} "
        "--diffusion-fps 24 "
        "--diffusion-guidance-scale 5.0 "
        "--diffusion-flow-shift 5.0 "
        "--diffusion-noise-level 0.7 "
        "--diffusion-sde-type sde "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window "
        "--diffusion-num-sde-steps 2 "
        f"--diffusion-sde-window-range 1,{args.num_diffusion_steps} "
    )

    eval_args = ""
    if args.eval_data_jsonl and args.eval_interval > 0:
        eval_data_jsonl = shlex.quote(args.eval_data_jsonl)
        eval_args = (
            f"--eval-prompt-data pickscore_val {eval_data_jsonl} "
            f"--eval-interval {args.eval_interval} "
            "--n-samples-per-eval-prompt 1 "
            f"--diffusion-eval-num-steps {args.num_diffusion_steps} "
            "--skip-eval-before-train "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        "--globalize-reward-std "
        "--diffusion-clip-range 1e-4 "
        "--diffusion-kl-beta 0.0 "
    )

    optimizer_args = "--lr 1e-4 --adam-beta2 0.999 --weight-decay 1e-4 "

    lora_args = (
        "--use-lora "
        "--lora-ipc-weight-sync "
        f"--lora-rank {args.lora_rank} "
        f"--lora-alpha {args.lora_alpha} "
        "--lora-init-weights gaussian "
    )

    reward_args = (
        "--rm-type pickscore "
        "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
        "--pickscore-model-path yuvalkirstain/PickScore_v1 "
        "--pickscore-num-frames 8 "
        "--pickscore-num-workers 1 "
        "--pickscore-num-gpus-per-worker 0 "
        "--pickscore-batch-size 8 "
    )

    wandb_args = U.get_default_wandb_args(
        __file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=4, wandb_log_image_interval=10
    )

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 2 "
        f"--sglang-tp-size {args.num_gpus} "
        "--sglang-sp-degree 1 "
        "--sglang-dit-precision bf16 "
        "--update-weight-buffer-size 2147483648 "
    )

    train_backend_args = (
        "--train-backend fsdp "
        "--fsdp-master-dtype bf16 "
        "--fsdp-reduce-dtype bf16 "
        "--diffusion-forward-dtype bf16 "
        "--update-weight-target-module transformer "
    )

    perf_args = "--gradient-checkpointing --rollout-parser-num-workers 2 "

    placement_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus} "
        f"--rollout-num-gpus {args.num_gpus} "
        f"--rollout-num-gpus-per-engine {args.num_gpus} "
        f"--num-gpus-per-node {args.num_gpus} "
        "--colocate "
        "--colocate-reward "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {diffusion_args} {eval_args} {grpo_args} "
            f"{optimizer_args} {lora_args} {reward_args} {wandb_args} {sglang_args} "
            f"{train_backend_args} {perf_args} {placement_args} {args.extra_args}"
        ),
        num_gpus_per_node=args.num_gpus,
        config=args,
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    execute(args)


if __name__ == "__main__":
    typer.run(main)
