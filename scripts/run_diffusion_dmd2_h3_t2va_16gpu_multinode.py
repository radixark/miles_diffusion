"""MiniMax H3 t2va 4-step DMD2 distillation, 2 nodes x 8 GPU, data-free (prompts only).

Multi-node: this script submits into an EXISTING ray cluster of 2 x 8 colocated train/rollout GPUs --
see docs/user-guide/launch-script.md (Multi-node training) for the bring-up, then run it on the head
node with MILES_SCRIPT_EXTERNAL_RAY=1.

Recipe: FastVideo's FastH3 Preview v1 Dense / Data-Free (checkpoint_metadata.json of
FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree, released at step 1000), mapped onto
--loss-type dmd2:
  * full-parameter student and critic, both from MiniMax-H3, frozen MiniMax-H3 teacher; fp32 master,
    bf16 forward; AdamW (0.9, 0.999), weight decay 0.01, constant lr 2e-6 for both, grad clip 1.0;
  * 4 student steps on the uniform shift-12 grid (FastH3's rungs 999/749/500/250 within 1.4e-4 in
    sigma), backward simulation by stopping each rollout at a random step, plain Euler (ode) rollout;
  * teacher unguided (H3 has no CFG; FastH3 real_score_guidance_scale 1.0), critic loss in x0 space;
  * scoring and critic sigma: unshifted range 0.001-0.999 read through shift 5. FastH3 sets
    score_timestep_shift 2.4 on its own clock, documented as FastGen's shift 5 (12 / 5 = 2.4);
  * one rollout = 5 critic steps of 64 samples + 1 student step on the last 64 (FastH3's
    generator_update_interval 5 at global batch 64), so 1000 FastH3 steps = 200 rollouts.
Deviations: 768p 16:9 only (FastH3 mixes shapes); only the video stream is distilled (the rollout
records no audio trajectory), so audio follows whatever the shared layers learn -- listen to the media.

Prompts: the 248,221 VidProM prompts of vidprom_filtered_extended.txt (the list Self-Forcing, NVlabs/rcm
and FastVideo's Wan data-free DMD train on), downloaded at a pinned revision and split here
deterministically: train 247,219 / test 1,000 / eval_media 26 (16 test prompts + 10 FastGen-DMD
demo prompts that also describe sound). The output hashes are pinned, so every node builds the same files.

What to watch: eval media at rollout 0 is the base model at 4 steps; every --eval-interval rollouts
the student renders the same prompts and seeds. Videos (with audio once the engine returns it) go to
W&B eval_media/ and to <output_dir>/<run>/eval_media/eval_<rollout>/. References on the same prompts
and seeds, for side-by-side comparison:
    MILES_SCRIPT_EXTERNAL_RAY=1 python3 scripts/run_diffusion_dmd2_h3_t2va_16gpu_multinode.py --reference teacher
    MILES_SCRIPT_EXTERNAL_RAY=1 python3 scripts/run_diffusion_dmd2_h3_t2va_16gpu_multinode.py --reference fasth3
teacher = MiniMax-H3 at its default 50 steps; fasth3 = the FastH3 Dense / Data-Free student at 4 steps.

Usage (after the multi-node bring-up, on the head node):
    MILES_SCRIPT_EXTERNAL_RAY=1 python3 scripts/run_diffusion_dmd2_h3_t2va_16gpu_multinode.py
"""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "MiniMaxAI/MiniMax-H3"
FASTH3_MODEL = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
WANDB_PROJECT = "miles-diffusion-dmd2"

PROMPT_REPO = "gdhe17/Self-Forcing"
PROMPT_FILE = "vidprom_filtered_extended.txt"
PROMPT_REVISION = "2f8b779212da279d212c22a509b66ad6552f350e"
PROMPT_SHA256 = "7896742f468bc8aef9e4547424d1ce0a951acdb2a82233790155401a99bf5aa5"
PROMPT_SPLIT_SEED = 1234
NUM_TEST_PROMPTS = 1000
NUM_EVAL_TEST_PROMPTS = 16
# FastGen-DMD scripts/inference/prompts/flux3_t2va_demo.txt: short prompts that also describe sound.
DEMO_PROMPTS = [
    "a small red boat bobbing on gentle ocean waves at sunset, seagulls calling in the distance",
    "a golden retriever running through autumn leaves in a park, leaves crunching underfoot",
    "close-up of rain falling on a city street at night, neon reflections, soft rain sounds",
    "a steam locomotive pulling into a rural station, whistle blowing, steam hissing",
    "a campfire crackling on a beach under a starry sky, waves breaking softly",
    "a busy street market in Marrakech, vendors calling out, colorful fabrics swaying",
    "time-lapse of clouds rolling over snowy mountain peaks, wind howling",
    "a jazz drummer playing a fast solo in a smoky club, cymbals shimmering",
    "children playing soccer on a sunny field, laughter and a whistle",
    "a waterfall cascading into a jungle pool, birds singing, water roaring",
]
SPLIT_SHA256 = {
    "train.jsonl": "ec9ecd057ec5d3aa4b383224bc9071edfb73cf4a77b6594a5c9708068c39bc71",
    "test.jsonl": "be22ea8ecdbe6d909099cd943fcd20cb4e5b0870160212b3483ad085e9f0684c",
    "eval_media.jsonl": "9a9faf270a94fe98adbe28dfcb059a6acbc6d5f3d5f8da7a3e9a51c57e18fe60",
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 200
    # 5 critic steps x 64 samples; the student steps on the last 64.
    rollout_batch_size: int = 320
    eval_interval: int = 5
    save_interval: int = 50
    duration_seconds: float = 5.0
    data_dir: str = "/root/datasets"
    # "teacher" or "fasth3": render the eval prompts with that model instead of training.
    reference: str = ""
    extra_args: str = ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, prompts: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(json.dumps({"input": prompt}, ensure_ascii=False) + "\n")


def prepare(args: ScriptArgs) -> str:
    """Download the pinned prompt list and split it; returns the directory of the jsonl files."""
    from huggingface_hub import hf_hub_download

    root = Path(args.data_dir) / "vidprom-extended"
    prompt_dir = root / "prompts"
    if all((prompt_dir / name).exists() and _sha256(prompt_dir / name) == h for name, h in SPLIT_SHA256.items()):
        return str(prompt_dir)

    raw = Path(hf_hub_download(PROMPT_REPO, PROMPT_FILE, revision=PROMPT_REVISION, local_dir=root / "raw"))
    if _sha256(raw) != PROMPT_SHA256:
        raise RuntimeError(f"{raw} does not match the pinned sha256 {PROMPT_SHA256}")
    prompts = list(dict.fromkeys(line.strip() for line in raw.read_text(encoding="utf-8").splitlines()))
    prompts = [prompt for prompt in prompts if prompt]
    order = list(range(len(prompts)))
    random.Random(PROMPT_SPLIT_SEED).shuffle(order)
    shuffled = [prompts[i] for i in order]
    test = shuffled[:NUM_TEST_PROMPTS]
    splits = {
        "train.jsonl": shuffled[NUM_TEST_PROMPTS:],
        "test.jsonl": test,
        "eval_media.jsonl": test[:NUM_EVAL_TEST_PROMPTS] + DEMO_PROMPTS,
    }
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        _write_jsonl(prompt_dir / name, split)
        if _sha256(prompt_dir / name) != SPLIT_SHA256[name]:
            raise RuntimeError(f"{prompt_dir / name} does not match the pinned split sha256")
    return str(prompt_dir)


def _fasth3_checkpoint(data_dir: str) -> str:
    """MiniMax-H3 with FastH3's transformer: the export shares every other component and both shifts."""
    from huggingface_hub import snapshot_download

    base = Path(snapshot_download(MODEL))
    student = Path(snapshot_download(FASTH3_MODEL, allow_patterns=["transformer/*"]))
    overlay = Path(data_dir) / "MiniMax-H3-FastH3-4step-dense"
    overlay.mkdir(parents=True, exist_ok=True)
    for entry in base.iterdir():
        link = overlay / entry.name
        if not link.is_symlink():
            link.symlink_to(student / "transformer" if entry.name == "transformer" else entry)
    return str(overlay)


def execute(args: ScriptArgs, prompt_dir: str) -> None:
    run_name = f"diffusion_dmd2_h3_t2va_{args.reference or 'student'}_{U.create_run_id()}"
    run_dir = f"{args.output_dir}/{run_name}"

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {prompt_dir}/train.jsonl "
        "--input-key input "
        f"--rollout-batch-size {args.rollout_batch_size} "
        "--n-samples-per-prompt 1 "
        "--num-steps-per-rollout 1 "
        "--rollout-microgroup-size 1 "
        "--rm-type none "
        "--rollout-video-dtype uint8 "
    )

    diffusion_args = (
        "--diffusion-task t2va "
        "--diffusion-num-steps 4 "
        "--diffusion-guidance-scale 1.0 "
        "--diffusion-h3-aspect-ratio 16:9 "
        f"--diffusion-h3-duration-seconds {args.duration_seconds} "
        "--diffusion-flow-shift 12.0 "
        "--diffusion-audio-flow-shift 3.0 "
        # H3 drops fps from the request; this sets the playback rate of the logged and saved media.
        "--diffusion-fps 24 "
    )

    eval_args = (
        f"--eval-prompt-data dmd2_media {prompt_dir}/eval_media.jsonl "
        "--n-samples-per-eval-prompt 1 "
        f"--save-eval-media-dir {run_dir}/eval_media "
    )

    # Every eval video; training rollouts carry no media, since --rm-type none skips their decode.
    wandb_args = U.get_default_wandb_args(
        __file__,
        run_id=run_name,
        project=WANDB_PROJECT,
        wandb_log_num_images=NUM_EVAL_TEST_PROMPTS + len(DEMO_PROMPTS),
    )

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 2 "
        "--sglang-tp-size 2 "
        "--sglang-sp-degree 1 "
        "--sglang-ulysses-degree 1 "
        "--sglang-ring-degree 1 "
        "--sglang-dit-precision bf16 "
        "--update-weight-buffer-size 4294967296 "
    )

    topology_args = (
        "--actor-num-nodes 2 "
        "--actor-num-gpus-per-node 8 "
        "--num-gpus-per-node 8 "
        "--rollout-num-gpus 16 "
        "--rollout-num-gpus-per-engine 2 "
        "--rollout-parser-num-workers 16 "
        "--colocate "
        # H3 denoising blocks the engine's event loop for tens of seconds per sample.
        "--rollout-health-check-interval 60 "
        "--miles-router-health-check-failure-threshold 30 "
    )

    if args.reference:
        if args.reference == "teacher":
            checkpoint, eval_steps = MODEL, 50
        elif args.reference == "fasth3":
            checkpoint, eval_steps = _fasth3_checkpoint(args.data_dir), 4
        else:
            raise ValueError(f"--reference must be teacher or fasth3, got {args.reference!r}")
        mode_args = (
            f"--hf-checkpoint {checkpoint} "
            "--debug-rollout-only "
            "--num-rollout 0 "
            "--eval-interval 1 "
            f"--diffusion-eval-num-steps {eval_steps} "
        )
    else:
        mode_args = (
            f"--hf-checkpoint {MODEL} "
            f"--save {run_dir}/ckpt "
            f"--save-interval {args.save_interval} "
            f"--num-rollout {args.num_rollout} "
            f"--eval-interval {args.eval_interval} "
            "--loss-type dmd2 "
            f"--teacher-load {MODEL} "
            "--dmd2-teacher-guidance-scale 1.0 "
            "--dmd2-critic-steps-per-rollout 5 "
            "--dmd2-score-sigma-range 0.001 0.999 "
            "--dmd2-critic-loss-space x0 "
            "--fsdp-flow-shift visual=5.0 "
            "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.stop_at_random_step "
            "--diffusion-sde-type ode "
            "--lr 2e-6 "
            "--critic-lr 2e-6 "
            "--weight-decay 0.01 "
            "--train-backend fsdp "
            "--fsdp-master-dtype fp32 "
            "--fsdp-reduce-dtype fp32 "
            "--diffusion-forward-dtype bf16 "
            "--gradient-checkpointing "
        )

    U.execute_train(
        train_args=(
            f"{mode_args} {rollout_args} {diffusion_args} {eval_args} {wandb_args} {sglang_args} "
            f"{topology_args} {args.extra_args}"
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
