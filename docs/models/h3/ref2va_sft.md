---
title: H3 Ref2VA SFT
description: Ordered multimodal conditions and joint video/audio supervision with independent noise schedules.
---

# H3 Ref2VA SFT

H3 names reference-to-audio-video generation `ref2va`. This SFT path trains both
video and audio targets, using the official `transformer_ref` diffusers component
and the native `Ref2VA` video/audio VAEs. The existing `t2va` SFT recipe remains
available for video-only datasets.

## Dataset

Each JSONL row has the same three parts as an SGLang H3 request: a text
`prompt` (the `--input-key` column), an ordered `conditions` list in SGLang's
condition schema (`type`, `uri`, `role`, and optionally `frame_index` /
`start_time_seconds`), and a `target`. For SFT the target names the supervised
media. For readability, this example is expanded over multiple lines; put each
sample on one line in the actual file.

```json
{
  "prompt": "<Picture 1> speaks with the voice of <Audio 1>, moving like <Video 1>.",
  "conditions": [
    {"type": "image", "uri": "references/person.png", "role": "reference"},
    {"type": "audio", "uri": "https://example.org/voice.wav", "role": "reference"},
    {"type": "video", "uri": "references/motion.mp4", "role": "reference", "start_time_seconds": 1.0}
  ],
  "target": {"visual": "targets/clip.mp4"}
}
```

Media support absolute paths, paths relative to the JSONL file, and HTTP(S) URLs. Downloads are atomic persistent snapshots under
`.sft_cache/media`; use a new URL or remove that cached download to refresh a
changing remote resource. All workers must be able to access the dataset and
its cache, as with the existing SFT producer. Conditions are passed to SGLang's
validator unchanged, so `role` is required and unknown fields are rejected.

The target video must already be the training canvas (1344×768 in the recipe) with square pixels and
exactly `(--diffusion-output-num-frames - 1) * --sft-frame-stride + 1` frames (107 in the recipe); it is
rejected rather than resized or trimmed. Every Ref2VA row needs at least one condition; rows without conditions train only with `--diffusion-task t2va`.

`video` references may be silent; `video_audio` references require an audio
track. The audio target is the target video's own soundtrack, cut over the same
interval as its frames, decoded at its native sample rate and resampled once to
the audio VAE's 32 kHz. A separate `target.audio` file is rejected; mux it into
the video first. A short soundtrack is an error. A silent target video trains the
video only: its audio rows are noised silence that keeps the packed sequence native
but carries no loss, as ai-toolkit does for silent clips.

The condition list is not Qwen chat messages.
H3 uses Qwen3-VL as a feature encoder **without its chat template**: text is
tokenized with `add_special_tokens=False`, with no system/user/assistant wrapper.
For Ref2VA, H3 builds its own input sequence from ordered `<Picture i>:`,
`<Audio j>:`, and `<Video k>:` labels and explicit visual tokens, then appends
the prompt text.

The prompt and the full condition list participate in cache identity. Media references retain
their relative order in H3's ordinal labels and packed RoPE positions.
Image/video references enter both H3's text-encoder input sequence and the video
VAE; audio references enter the audio VAE and only their ordinal labels enter
the text encoder. The supervised target is separate from the reference
conditions and is not exposed to the text encoder.

Hybrid Ref2VA samples may add image conditions with `role: "keyframe"` and
`frame_index: 0` or `-1` (first, last, or first followed by last). At least one
independent reference is required. Native H3 places these guide latents on the
target timeline; it does not add them to the reference text presentation.

## Run

Use the normal miles-diffusion training image with the repository's pinned
diffusers version, FFmpeg/FFprobe, and SGLang's `sglang-miles` branch. The Ref2VA
adapter targets the H3 native encoding/packing API at SGLang revision
`14a1fa7d93e87355375b2c3b6fa17255a4b8aa07`.

```bash
python scripts/run_diffusion_sft_h3_ref2va.py --prompt-data /data/train.jsonl
```

The recipe trains LoRA on 8 GPUs at 768×1344, 107 frames at 24 FPS, micro-batch
size 1, video flow shift 12, and audio flow shift 3. The default LoRA targets are H3's shared
attention and FFN layers, which receive gradients from both modalities.
Targets use H3's `17n+5` frame grid and a selected duration of 4–15 seconds.
Canvas dimensions must match a native H3 aspect-ratio preset after its area cap
and 32-pixel rounding; for example, nominal 16:9 resolves to 1344×768.
Set training flags through `--extra-args`. When assembling a custom command,
pass `--diffusion-task ref2va --update-weight-target-module transformer_ref
--micro-batch-size 1 --fsdp-flow-shift visual=12,audio=3`.
A checkpoint should use the official root pipeline layout
(including `model_index.json` and `transformer_ref/`).

## Training representation

An encoder returns plain, weights-only-loadable dictionaries:

```python
{
    "latent": {"visual": video_rows, "audio": audio_rows},
    "cond_kwargs": packed_h3_conditioning,
}
```

`latent` maps each latent stream name to its clean latent. Single-stream families
cache only `visual`. The DiT forward
(`compute_noise_pred`) takes and returns dicts keyed by stream name; H3's packs
`visual` and `audio` into one forward when both are present.

For each stream, training independently samples a timestep and Gaussian noise,
forms `x_sigma = (1-sigma)*x0 + sigma*noise`, and supervises `noise-x0`.
Each stream's noise schedule is shifted by its own `--fsdp-flow-shift` term
(`visual=12,audio=3`; Ref2VA requires both keys).
Adding other streams does not change a stream's random draws. The visual stream's
timestep also selects the DiT component. Per pair, each stream's MSE is averaged
over its own elements before the streams are summed, so the much larger video
token count does not drown out audio. H3 receives `t=1-sigma` and its native output is
negated to match this objective. References keep their native fixed noise
augmentation and timestep rules. Reference rows never contribute target loss.

Cache keys include ordered condition semantics, every local media revision, and
the encoding configuration, so the old video-only cache is never reused.

## Validation

The CPU suite covers the request format, media load, stream preparation and
loss, and the H3 adapter, including a small real diffusers H3 forward/backward
in fp32 and bf16 through both modality heads:

```bash
python -m pytest -q \
  tests/fast/backends/fsdp_utils/test_h3_streams.py \
  tests/fast/backends/fsdp_utils/test_loss_hub_sft.py \
  tests/fast/rollout/test_diffusion_request.py \
  tests/fast/rollout/test_sft_read_media.py \
  tests/fast/rollout/test_sft_h3_ref2va_encoder.py
```

These tests do not establish full-checkpoint GPU encoder parity, distributed FSDP
training behavior, or generated sample quality; those require a GPU run.
