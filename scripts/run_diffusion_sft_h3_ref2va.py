"""8-GPU MiniMax H3 Ref2VA LoRA SFT on a jsonl dataset of ordered conditions and audio-video targets.

Like run_diffusion_sft_h3_t2va.py, the sft_rollout plugin encodes each round's cache misses through a
colocated, non-resident encoder actor pool; Ref2VA adds the audio VAE and Qwen3-VL processor so references
and the target soundtrack are encoded too, and trains both heads of transformer_ref.

Dataset rows, relative to the jsonl; conditions follow SGLang's H3 condition schema, which validates them unchanged:
    {"prompt": "<Picture 1> speaks with the voice of <Audio 1>.",
     "conditions": [{"type": "image", "uri": "refs/person.png", "role": "reference"},
                    {"type": "audio", "uri": "refs/voice.wav", "role": "reference"}],
     "target": {"visual": "targets/clip.mp4"}}
Target videos must be exactly 1344x768 / 107 frames at 24 fps; their own soundtrack is the audio target, and a
silent target trains only its video.

Usage:
    python3 scripts/run_diffusion_sft_h3_ref2va.py --prompt-data /abs/train.jsonl
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "MiniMaxAI/MiniMax-H3"
WANDB_PROJECT = "miles-diffusion-sft"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    prompt_data: str = ""
    resume_ckpt: str = ""
    start_rollout: int = -1
    num_epoch: int = 3
    extra_args: str = ""


def execute(args: ScriptArgs) -> None:
    run_name = f"diffusion_sft_h3_ref2va_{U.create_run_id()}"

    ckpt_args = (
        f"--hf-checkpoint {MODEL} --sft-encoder-checkpoint {MODEL} "
        f"--save {args.output_dir}/{run_name}/ckpt --save-interval 20 "
        # Ref2VA weights live in the transformer_ref component.
        "--update-weight-target-module transformer_ref "
    )
    if args.resume_ckpt:
        ckpt_args += f"--load {args.resume_ckpt} "
        if args.start_rollout >= 0:
            ckpt_args += f"--start-rollout-id {args.start_rollout} "

    rollout_args = (
        "--rollout-function-path miles.rollout.sft_rollout.generate_rollout "
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--rollout-batch-size 32 "
        f"--num-epoch {args.num_epoch} "
        "--num-steps-per-rollout 4 "
        # H3 serving grid: 16:9 canvas at short_edge=768, 24 fps, 17n+5 frames.
        "--diffusion-height 768 "
        "--diffusion-width 1344 "
        "--diffusion-output-num-frames 107 "
        # H3 distilled CFG into the checkpoint; the family rejects guided training.
        "--diffusion-guidance-scale 1.0 "
    )

    sft_args = (
        "--loss-type sft_loss "
        "--train-only "
        "--custom-convert-samples-to-train-data-path miles.rollout.sft_rollout.convert_samples_to_train_data "
        "--custom-rollout-log-function-path miles.rollout.sft_rollout.log_rollout_data "
        "--custom-prepare-train-batch-path miles.backends.fsdp_utils.loss_hub.sft.prepare_sft_batch "
        "--custom-loss-function-path miles.backends.fsdp_utils.loss_hub.sft.sft_loss_formula "
        "--diffusion-task ref2va "
        "--sft-offload-encoder "
    )

    optimizer_args = "--lr 3e-5 --adam-beta2 0.999 --weight-decay 0.01 --adam-eps 1e-15 "

    lora_args = "--use-lora --lora-rank 64 --lora-alpha 128 --lora-init-weights gaussian "

    wandb_args = U.get_default_wandb_args(__file__, run_id=run_name, project=WANDB_PROJECT)

    train_backend_args = (
        "--train-backend fsdp "
        "--fsdp-master-dtype fp32 "
        "--fsdp-reduce-dtype fp32 "
        "--diffusion-forward-dtype bf16 "
        # Each target stream samples sigmas on its own serving schedule: video shift 12, audio shift 3.
        "--fsdp-flow-shift visual=12,audio=3 "
    )

    # The packed joint forward takes one sample per micro-batch.
    perf_args = "--micro-batch-size 1 --gradient-checkpointing "

    misc_args = "--actor-num-gpus-per-node 8 --num-gpus-per-node 8 "

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {sft_args} {optimizer_args} {lora_args} "
            f"{wandb_args} {train_backend_args} {perf_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=8,
        config=args,
        train_script="train_sft.py",
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    execute(args)


if __name__ == "__main__":
    typer.run(main)
