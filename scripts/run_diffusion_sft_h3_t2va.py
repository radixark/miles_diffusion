"""8-GPU MiniMax H3 t2va LoRA SFT on a (video, prompt) jsonl dataset.

No sglang engines: the sft_rollout plugin lazily encodes each round's cache misses
through a colocated encoder actor pool (Qwen3-VL-32B layer-50 text encoder + H3 video
VAE), writing one content-addressed file per sample into .sft_cache/ next to the jsonl.
Epoch 2 onward is all cache hits. The encoder is non-resident: it is dropped after every
miss burst, so train steps never share the GPU with its ~70GB of weights. Prefer running
scripts/precompute_sft_cache.py first so the whole dataset is encoded with one encoder
load per GPU instead of one per rollout round.

Dataset rows: {"prompt": "...", "metadata": {"video": "/abs/path.mp4"}}
Videos must already sit on H3's serving grid: short_edge=768 canvas, 24 fps, and a
17n+5 frame count (the default spec is 1344x768 / 107 frames, ~4.46 s). See
scripts/prepare_wisa_h3_lighting.py for a dataset pipeline that produces this format.

Per rollout step: 64 samples, num_steps_per_rollout=4, so 16 samples per optimizer
step over 8 dp ranks is 2 samples per rank at mbs=1.

Usage:
    MILES_SCRIPT_DATA_JSONL=/abs/data.jsonl python3 scripts/run_diffusion_sft_h3_t2va.py
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "MiniMaxAI/MiniMax-H3"
WANDB_PROJECT = "miles-diffusion-sft"

# H3 DiT LoRA targets: packed self-attn and FFN (miles-side diffusers module names).
LORA_TARGET_MODULES = "attn.to_q attn.to_k attn.to_v attn.to_out.0 ff.net.0.proj ff.net.2"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    data_jsonl: str = ""
    resume_ckpt: str = ""
    start_rollout: int = -1
    num_epoch: int = 3
    extra_args: str = ""


def execute(args: ScriptArgs) -> None:
    if not args.data_jsonl:
        raise SystemExit("set --data-jsonl (or MILES_SCRIPT_DATA_JSONL) to a jsonl with prompt + metadata.video")
    run_name = f"diffusion_sft_h3_t2va_{U.create_run_id()}"

    ckpt_args = (
        f"--hf-checkpoint {MODEL} --sft-encoder-checkpoint {MODEL} "
        f"--save {args.output_dir}/{run_name}/ckpt --save-interval 20 "
    )
    if args.resume_ckpt:
        ckpt_args += f"--load {args.resume_ckpt} "
        if args.start_rollout >= 0:
            ckpt_args += f"--start-rollout-id {args.start_rollout} "

    rollout_args = (
        "--rollout-function-path miles.rollout.sft_rollout.generate_rollout "
        f"--prompt-data {args.data_jsonl} "
        "--input-key prompt "
        "--rollout-batch-size 64 "
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
        "--sft-frame-stride 1 "
        "--sft-offload-encoder "
    )

    optimizer_args = "--lr 3e-5 --adam-beta2 0.999 --weight-decay 0.01 --adam-eps 1e-15 "

    lora_args = (
        "--use-lora "
        "--lora-rank 64 "
        "--lora-alpha 128 "
        f"--lora-target-modules {LORA_TARGET_MODULES} "
        "--lora-init-weights gaussian "
    )

    wandb_args = U.get_default_wandb_args(__file__, run_id=run_name, project=WANDB_PROJECT)

    train_backend_args = (
        "--train-backend fsdp "
        "--fsdp-master-dtype fp32 "
        "--fsdp-reduce-dtype fp32 "
        "--diffusion-forward-dtype bf16 "
        # Match the engine's t2va serving schedule (video shift 12) so training
        # sigmas cover the same grid inference will sample.
        "--fsdp-flow-shift 12.0 "
    )

    perf_args = "--micro-batch-size 1 --gradient-checkpointing "

    misc_args = "--actor-num-gpus-per-node 8 --num-gpus-per-node 8 "

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {sft_args} {optimizer_args} {lora_args} "
            f"{wandb_args} {train_backend_args} {perf_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=8,
        config=args,
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    execute(args)


if __name__ == "__main__":
    typer.run(main)
