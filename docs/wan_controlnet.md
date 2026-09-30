# Wan2.2-TI2V-5B pose ControlNet SFT

This experimental Miles integration freezes the pretrained Wan transformer and
trains a newly initialized control branch on aligned RGB/pose video pairs. It uses
native `train_sft.py`, the existing SFT encoder pool, FSDP2, and flow-matching loss.
It does not launch a SGLang generation server or use RL rewards.

The implementation copies selected pretrained Wan blocks into a separate branch,
adds a trainable pose patch embedding, and initializes every residual output
projection to zero. Four control blocks attach after backbone layers 0, 7, 15,
and 22. The zero-initialized model begins with the base model's output. After
training opens the output projections, gradients reach the pose encoder and copied
blocks. The original backbone parameters remain frozen; gradients through its
operations must remain enabled.

## Data

Each JSONL row contains a caption, the RGB target, and its aligned rendered pose:

```json
{"prompt":"A person raises their left arm.","metadata":{"video":"rgb/001.mp4","control_video":"pose/001.mp4"}}
```

Relative media paths resolve against the JSONL directory. Each pair must have the
same resolution, frame count, frame rate, crop, and timeline. A pose-only dataset
cannot supply the target for this training objective. Pose extraction errors can
teach incorrect conditioning and should be inspected before training.

The starter recipe uses 832×480 RGB/pose video and 49 frames at 24 FPS. Wan2.2-TI2V-5B uses
spatial VAE compression followed by transformer patchification, so dimensions
must be multiples of 32; frame counts must be `4k+1`. The same frame stride is
applied to each member of the pair. Prepare clips at the intended output frame
rate rather than treating frame index as physical time across different rates.

The frozen UMT5 encoder supplies text embeddings. The frozen Wan VAE produces
normalized target and pose latents. The target uses posterior sampling; pose
conditioning uses the posterior mode. SFT adds noise only to target latents and
optimizes predicted velocity against `noise - clean_target`.

Cached samples include control file identity and preprocessing settings. If a
preprocessing implementation changes without a cache version change, remove that
experiment's stale `.sft_cache` before re-encoding.

## Launch

Install the repository's pinned dependencies. Use an environment with the pinned
Diffusers version, PyTorch/FSDP2, Ray, Transformers, FFmpeg, and the usual Miles
training dependencies. The launcher accepts a local Diffusers snapshot directory
or `Wan-AI/Wan2.2-TI2V-5B-Diffusers`; Hub loading may download uncached components.

A starter layout uses four training GPUs plus one dedicated frozen-encoder GPU:

```bash
python scripts/run_diffusion_sft_wan_controlnet.py \
  --base-model /models/Wan2.2-TI2V-5B-Diffusers \
  --data-jsonl /data/pose/train.jsonl \
  --output-dir /runs/wan-pose-first \
  --cuda-visible-devices 0,1,2,3,4 \
  --train-gpus 4 --steps 200 --steps-per-rollout 2 \
  --control-blocks 4 --lr 2e-5
```

For Hub loading, replace `--base-model` with the model ID or omit that option.
Use `--dry-run` to inspect the exact native Miles arguments without creating a Ray
cluster. The launcher starts its own local Ray instance; it does not terminate
other Ray or SGLang jobs. Choose GPUs that are available for this experiment.

The default microbatch is one sample per training GPU. `--samples-per-gpu` controls
gradient accumulation per optimizer step. The defaults use flow shift 5 and 20
warmup steps. `--steps` counts optimizer
steps and must be divisible by `--steps-per-rollout`. `--save-interval` counts SFT
rollouts, so the default two steps per rollout and save interval 20 save every
40 optimizer steps. These are starter settings, not a convergence claim.

An omitted `--initial-controlnet` creates a new branch from the base model.
Passing an exported branch directory continues its training. For a full Miles
resume, use `--resume-checkpoint RUN/checkpoints`, which includes the iteration
tracker, optimizer, and scheduler state. Keep the base model, branch architecture,
and data settings consistent across a resume. A resume preserves the original
`launch.json` and writes a separate `resume_launch_*.json` event.

Optional Weights & Biases logging uses credentials from the process environment:

```bash
# Authenticate with `wandb login`, or inject WANDB_API_KEY using your secret manager.
export WANDB_PROJECT=wan-pose-control
export WANDB_ENTITY=my-team
export WANDB_MODE=online
python scripts/run_diffusion_sft_wan_controlnet.py \
  --data-jsonl /data/pose/train.jsonl --output-dir /runs/wan-pose-tracked \
  --cuda-visible-devices 0,1,2,3,4 --wandb --run-name pose-first
```

Use `WANDB_MODE=offline` to record locally. The launcher does not put credentials
in command arguments or its `launch.json` manifest.

## Export and inference

Training checkpoints use native Miles DCP and may include the frozen backbone.
Export a reusable branch without loading those backbone tensors:

```bash
python scripts/export_wan_controlnet.py \
  --ckpt-dir /runs/wan-pose-first/checkpoints/iter_0000019 \
  --base-checkpoint /models/Wan2.2-TI2V-5B-Diffusers \
  --num-control-blocks 4 --out /runs/wan-pose-first/controlnet
```

Use an iteration directory that actually exists. Export writes
`controlnet.safetensors`, `controlnet_config.json` (including exact block indices
and base-model identity), and `export_info.json`. Only the control tensors are
read from DCP; a meta-device backbone validates the structure. New exports also
record and verify the complete public backbone configuration, including normalization
and attention settings. Earlier experimental exports without that field load with
an explicit warning and only partial configuration checks; re-export their DCP
checkpoint to obtain full validation.

Use the **same original base weights used during training**. A control checkpoint
contains only the added branch. Stored base paths are provenance; a moved snapshot
or its `transformer` subdirectory can have a different path string without a
structural mismatch. Matching architecture alone does not establish identical
base weights.

The repository includes a minimal inference CLI. Supply an already aligned,
constant-frame-rate pose MP4 with exactly the requested frame count and size:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_wan_controlnet.py \
  --base-model /models/Wan2.2-TI2V-5B-Diffusers \
  --controlnet-checkpoint /runs/wan-pose-first/controlnet \
  --pose-video /data/pose/heldout_001.mp4 \
  --prompt "A person wearing a red jacket dances in a studio." \
  --output /runs/wan-pose-first/heldout_001.mp4 \
  --width 832 --height 480 --frames 49 --fps 24 \
  --steps 50 --guidance-scale 5 --flow-shift 5 --seed 42
```

The CLI uses the VAE posterior mode, training's latent normalization, FP32 VAE and
text encoder, BF16 transformer, and the official UniPC flow scheduler. It retains
`expand_timesteps` from the base pipeline configuration. It never silently
resizes the input pose or changes its timing. VAE tiling is off by default and
can be enabled explicitly. An adjacent JSON file records the generation settings,
full scheduler configuration, dependency versions, and branch metadata. For a matched baseline, rerun with `--control-scale 0` and a
different output path.

For an unmodified Diffusers Wan pipeline, replace its transformer with the
wrapper and bind a condition while generation runs:

```python
from miles.backends.fsdp_utils.models.wan_controlnet import WanPoseControlTransformer

pipe.transformer = WanPoseControlTransformer.from_controlnet(
    pipe.transformer, "/runs/wan-pose-first/controlnet"
)
with pipe.transformer.control_condition(pose_latents, control_scale=1.0):
    result = pipe(prompt=prompt, num_frames=49, height=480, width=832)
```

`pose_latents` must use the same VAE and normalization as training and match the
noisy generation latents in batch, channels, time, spatial size, device, and dtype.
This is a latent tensor, not the MP4 path. Use `control_scale=0` for the base
comparison. The context restores the previous condition even after errors;
concurrent requests must use separate wrapper instances.

## Supported scope

The initial recipe is single-transformer Wan2.2-TI2V-5B pose-conditioned SFT, with
one target video per prompt, SP=1, and a fixed batch shape. LoRA training, RL,
SGLang ControlNet serving, sequence parallelism, multi-expert Wan 14B, and explicit
first-frame conditioning are outside this integration. Nonempty
`attention_kwargs` are rejected on the control path; no adapter scaling is
silently applied. FSDP2 mixed precision and gradient checkpointing keep the
frozen backbone in the differentiable path while updating only the branch.

A lower training loss or a memorized training clip does not demonstrate better
anatomy. Compare fixed prompts, held-out poses, seeds, and sampling settings;
measure pose adherence, major anatomical errors, and visual quality separately.
