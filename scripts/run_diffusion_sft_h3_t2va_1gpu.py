"""1-GPU MiniMax H3 t2va LoRA SFT — the 8-GPU recipe's batch schedule on a single card.

Same training math as scripts/run_diffusion_sft_h3_t2va.py: 32 samples per rollout,
num_steps_per_rollout=4, so global_batch_size stays 8 samples per optimizer step. What
changes is only how those 8 samples are laid out: 8 dp ranks x 1 micro-batch becomes
1 rank x 8 gradient-accumulation micro-batches, still at micro_batch_size=1. Optimizer,
LoRA, sigma grid, dtypes, and the loss are the 8-GPU recipe's, flag for flag.

Why --fsdp-cpu-offload is not optional here: H3 is a 33.5B-parameter model, so the fp32
master copy the recipe trains against is ~134 GB. Sharded 8 ways that is the ~16 GB/rank
the 8-GPU run reports; on one 139.8 GB H200 it leaves nothing for activations.
CPUOffloadPolicy keeps the fp32 shard and the (LoRA-only) optimizer state in host RAM and
all-gathers one transformer block at a time as bf16, so GPU residency is activations plus a
single block. This preserves the master/forward dtypes rather than trading them away.

Measured against the 8-GPU reference run at these defaults, same dataset. Both runs report
sft_cache_miss: 0, and the step figures are perf/actor_train_time -- the training loop only,
with encoding excluded (it lands in perf/train_wait_time):

    GPU peak      39.5 GB of 139.8 GB, activations only -- see the note below
    host RAM      ~313 GB while FSDP materializes the shards, ~188 GB steady after
    init          ~12 min to load, shard, and broadcast the fp32 state
    optim step    278 s, against 30.5 s on 8 GPUs
    micro-batch   34.8 s, against 30.5 s on 8 GPUs

So the offload costs ~14% per micro-batch -- the PCIe round trip for every block, on both
the forward and the gradient-checkpoint recompute. The remaining 9x wall-clock gap is
simply having an eighth of the ranks.

Encoding is the other single-worker cost, and it shows up outside those figures: one encode
actor instead of eight at ~13.5 s per clip, measured as ~480 s of train_wait_time per
rollout (32 misses) on a cold cache against 1-16 s once warm. It also owns the run's GPU
peak while it runs -- the encoder is ~67 GB against the 39.5 GB train step -- and a fully
warm cache never constructs the pool at all. Cache entries are content-addressed on the
media and the encode settings and are identical to the ones the 8-GPU recipe writes, so a
directory that recipe already filled is reused as-is.

--fsdp-master-dtype is not a lever on the GPU number: under --fsdp-cpu-offload no weights
are resident and the gathered copy is bf16 either way, so the 39.5 GB is activations plus
one block. Setting it to bf16 halves host RAM instead (~134 GB to ~67 GB for the frozen
base; PEFT keeps the trainable LoRA in fp32 regardless), which is the knob for a low-RAM
host rather than a low-memory card.

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
        "--fsdp-master-dtype fp32 "
        "--fsdp-reduce-dtype fp32 "
        "--diffusion-forward-dtype bf16 "
        # Match the engine's t2va serving schedule (video shift 12) so training
        # sigmas cover the same grid inference will sample.
        "--fsdp-flow-shift 12.0 "
        # The 134GB fp32 master has nowhere to live on one card; see the module docstring.
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
