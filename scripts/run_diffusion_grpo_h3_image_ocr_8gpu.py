"""MiniMax H3 Image t2i OCR Flow-GRPO with LoRA on 8 colocated GPUs.

FSDP DP=8 shares the GPUs with four TP2 sglang engines; the LoRA adapters sync over CUDA IPC.

H3 Image is a 6-step distilled model: its DiT has one output head per denoising step and the
release fixes the sigma grid, so the step count, the flow shift and the guidance are not knobs.
What the recipe chooses, each from a measurement (docs/models/h3/h3-image.md):
  * cps, not sde: on this grid flow-SDE noise costs image quality in proportion to its level,
    cps at 0.8 costs none.
  * Every step but the last is noisy and trained; the last ends at sigma 0, where cps has no
    noise. With a 2-step window a prompt's samples still differ by their initial noise, which no
    trained step explains, and the reward of 8 fixed prompts rose half as fast per rollout.
  * The engine applies the adapters at forward time (dynamic) instead of merging them: merged
    into bf16 weights, part of each small update is rounded away and the engine drifts from the
    trainer as the adapter grows.
  * The trainer recomputes the old log-probs: the engine's and the trainer's differ by about
    1e-4 on the base model, the size of the clip range.
  * lr 1e-4: at 2e-4 the recipe fits its train prompts faster and gets worse on every other one
    (eval OCR 0.60 -> 0.55 in 20 rollouts); at 1e-4 the eval rises.
  * No gradient checkpointing: one pair per forward peaks at 16 GB on the trainer, and
    checkpointing made the training phase a third longer.

The reward is OCR alone and nothing holds the policy near the base model (kl 0). Past about 200
rollouts the reward keeps rising because the scene is erased, until only the text is left: the
recipe stops there. Its images are not clean before that either, with a repeating texture from
about 140 rollouts (docs/models/h3/h3-image.md, reference results).

Usage:
    python3 scripts/run_diffusion_grpo_h3_image_ocr_8gpu.py

    # train/rollout alignment diagnostic: freeze the weights so the debug metrics measure
    # pure train-vs-rollout deviation rather than parameter drift
    python3 scripts/run_diffusion_grpo_h3_image_ocr_8gpu.py --num-rollout 2 \
        --eval-interval 0 --extra-args "--debug-skip-optimizer-step"
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "MiniMaxAI/MiniMax-H3-Image"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_ocr"
WANDB_PROJECT = "miles-diffusion-grpo"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 200
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 16
    num_steps_per_rollout: int = 2
    eval_interval: int = 20
    save_interval: int = 20
    data_dir: str = "/root/datasets"
    extra_args: str = ""


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{DATASET_SUBSET}/**", data_dir=args.data_dir)
    return f"{local_dir}/{DATASET_SUBSET}"


def execute(args: ScriptArgs, prompt_dir: str) -> None:
    run_name = f"diffusion_grpo_h3_image_ocr_8gpu_{U.create_run_id()}"

    ckpt_args = (
        f"--hf-checkpoint {MODEL} --diffusion-task t2i "
        f"--save {args.output_dir}/{run_name}/ckpt --save-interval {args.save_interval} "
    )

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {prompt_dir}/train.jsonl "
        "--input-key input "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--num-steps-per-rollout {args.num_steps_per_rollout} "
        f"--num-rollout {args.num_rollout} "
        # the H3 Image forward takes one sample and one step at a time
        "--rollout-microgroup-size 1 "
        "--micro-batch-size-sample 1 "
        "--micro-batch-size-tstep 1 "
        "--diffusion-train-iter-order sample_major "
    )

    diffusion_args = (
        "--diffusion-num-steps 6 "
        "--diffusion-guidance-scale 1.0 "
        "--diffusion-height 512 "
        "--diffusion-width 512 "
        "--diffusion-sde-type cps "
        "--diffusion-noise-level 0.8 "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window "
        "--diffusion-num-sde-steps 5 "
        "--diffusion-sde-window-range 0,5 "
        # the engine returns its own model output and transition mean for the trainer to compare with
        "--diffusion-debug-mode "
    )

    eval_args = ""
    if args.eval_interval:
        eval_args = (
            f"--eval-interval {args.eval_interval} "
            f"--eval-prompt-data ocr_test {prompt_dir}/test.jsonl "
            "--n-samples-per-eval-prompt 1 "
        )

    grpo_args = (
        "--advantage-estimator grpo --globalize-reward-std --diffusion-clip-range 1e-4 --diffusion-kl-beta 0.0 "
        "--diffusion-recompute-old-log-prob "
    )

    optimizer_args = "--lr 1e-4 --adam-beta2 0.999 --weight-decay 1e-4 "

    lora_args = "--use-lora --lora-ipc-weight-sync --lora-rank 64 --lora-alpha 128 --lora-init-weights gaussian "

    reward_args = "--rm-type ocr --ocr-num-workers 16 --rollout-parser-num-workers 8 "

    wandb_args = U.get_default_wandb_args(
        __file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=8, wandb_log_image_interval=10
    )

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 4 "
        "--sglang-lora-merge-mode dynamic "
        "--sglang-tp-size 2 "
        "--sglang-sp-degree 1 "
        "--sglang-ulysses-degree 1 "
        "--sglang-ring-degree 1 "
        "--sglang-dit-precision bf16 "
        "--update-weight-buffer-size 2147483648 "
    )

    train_backend_args = (
        "--train-backend fsdp "
        "--fsdp-master-dtype bf16 "
        "--fsdp-reduce-dtype bf16 "
        "--diffusion-forward-dtype bf16 "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--rollout-num-gpus 8 "
        "--rollout-num-gpus-per-engine 2 "
        "--num-gpus-per-node 8 "
        "--colocate "
        "--deterministic-mode "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {diffusion_args} {eval_args} {grpo_args} {optimizer_args} "
            f"{lora_args} {reward_args} {wandb_args} {sglang_args} {train_backend_args} "
            f"{misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=8,
        config=args,
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    prompt_dir = prepare(args)
    execute(args, prompt_dir)


if __name__ == "__main__":
    typer.run(main)
