"""scripts/run_diffusion_sft_h3_t2va.py on one card, with the same training math.

The 8-GPU recipe's schedule is preserved exactly, not approximated. global_batch_size is 8
either way -- arguments.py derives it as rollout_batch_size x n_samples_per_prompt /
num_steps_per_rollout, with no world-size term -- so the only thing one card changes is the
layout: 8 dp ranks x 1 micro-batch becomes 1 rank x 8 gradient-accumulation micro-batches,
still at micro_batch_size=1, still 4 optimizer steps per rollout. The gradient scale follows:
actor.py divides by num_local_pairs, which is 8 on the single rank and 1 on each of eight
ranks before FSDP2's mesh-mean divides by 8 again. Optimizer, LoRA, sigma grid, and loss are
the 8-GPU recipe's flag for flag.

Two flags do differ, both forced by having one card:

--fsdp-cpu-offload. H3 is 33.5B parameters, so even a bf16 master is ~66 GB and leaves no
room for activations on a 139.8 GB H200. CPUOffloadPolicy keeps the master shard and the
LoRA-only optimizer state in host RAM and all-gathers one transformer block at a time as
bf16, so GPU residency is activations plus a single block.

--fsdp-master-dtype bf16 instead of fp32, which under that offload changes no math. Only the
LoRA adapters train, and PEFT keeps those in fp32 whatever the base dtype; the frozen base is
gathered as bf16 for the forward at either setting, because MixedPrecisionPolicy applies
param_dtype=bf16 to every wrap and H3 declares no param_dtype_patterns. Measured: step 1 of
an otherwise identical run reproduces the fp32 loss to every digit logged (3.276656e-01).
diffusers' _keep_in_fp32_modules still holds proj_in, proj_out, audio_proj_in,
audio_proj_out, time_embedder and rope in fp32. --fsdp-reduce-dtype stays fp32 -- that one
governs the LoRA gradients, which do train.

What one card costs, measured against the 8-GPU reference run on the same dataset with a
warm cache (per-step figures are perf/actor_train_time, so encoding is excluded):

    GPU peak         39.5 GB, activations plus one block -- no weights are resident
    host RAM         217 GB peak through FSDP init, 99 GB steady (~95 GB of it pinned)
    init             ~6 min
    optimizer step   278 s against 30.5 s
    rollout          17.3 min against 122 s

The 9x wall-clock gap is 8x rank count and ~14% for the PCIe round trip the offload adds to
every block, on both the forward and the gradient-checkpoint recompute. A cold .sft_cache
adds more, and only here: one encode worker instead of eight, ~480-590 s per rollout through
the first epoch, during which the ~67 GB encoder rather than the train step owns the GPU
peak. Cache entries are content-addressed on the media and the encode settings and are
identical to the 8-GPU recipe's, so a directory it already filled is reused as-is.

What is not reproducible across the two, and cannot be: prepare_sft_batch seeds the sigma and
noise draw on (rollout_id, microbatch_id, dp_rank), so every sample lands on a different sigma
when the topology changes. Compare the two runs on trend and on the per-sigma buckets, never
point by point. Within one topology, --extra-args "--deterministic-mode" does pin the run
bit-for-bit; it costs ~25% per step and ~3.6 GB of GPU.

Usage:
    python3 scripts/run_diffusion_sft_h3_t2va_1gpu.py   # downloads DATASET on first run
    # custom data: append --prompt-data /abs/train.jsonl via --extra-args (last flag wins)
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "MiniMaxAI/MiniMax-H3"
DATASET = "rockdu/WISA-80K-Practical-Dynamics-254"
WANDB_PROJECT = "miles-diffusion-sft"

# H3 DiT LoRA targets: packed self-attn and FFN (miles-side diffusers module names).
LORA_TARGET_MODULES = "attn.to_q attn.to_k attn.to_v attn.to_out.0 ff.net.0.proj ff.net.2"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    data_dir: str = "/root/datasets"
    resume_ckpt: str = ""
    start_rollout: int = -1
    num_epoch: int = 3
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U.hf_download_dataset(DATASET, data_dir=args.data_dir)


def execute(args: ScriptArgs) -> None:
    run_name = f"diffusion_sft_h3_t2va_1gpu_{U.create_run_id()}"

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
        f"--prompt-data {args.data_dir}/{DATASET.split('/')[1]}/train.jsonl "
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
        # bf16 master, unlike the 8-GPU recipe: only the LoRA adapters train, PEFT keeps
        # those in fp32 whatever the base dtype, and the frozen base is gathered as bf16
        # for the forward at either setting -- so fp32 buys nothing here and costs 66GB of
        # host RAM. diffusers' _keep_in_fp32_modules still protects proj_in/proj_out/
        # audio_proj_in/audio_proj_out/time_embedder/rope. Reduce stays fp32: that one
        # governs the LoRA gradients.
        "--fsdp-master-dtype bf16 "
        "--fsdp-reduce-dtype fp32 "
        "--diffusion-forward-dtype bf16 "
        # Match the engine's t2va serving schedule (video shift 12) so training
        # sigmas cover the same grid inference will sample.
        "--fsdp-flow-shift 12.0 "
        # Even a bf16 master has nowhere to live on one card; see the module docstring.
        "--fsdp-cpu-offload "
    )

    perf_args = "--micro-batch-size 1 --gradient-checkpointing "

    misc_args = "--actor-num-gpus-per-node 1 --num-gpus-per-node 1 "

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {sft_args} {optimizer_args} {lora_args} "
            f"{wandb_args} {train_backend_args} {perf_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=1,
        config=args,
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
