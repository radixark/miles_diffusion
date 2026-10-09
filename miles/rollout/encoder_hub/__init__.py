"""Frozen-encoder logic per model family and --diffusion-task, decoupled from the training-side TrainPipelineConfig.

Each family module provides:
- ``load_encoder(args, device)``: load the frozen encode components (tokenizer/text
  encoder/VAE) from the ``--sft-encoder-checkpoint`` HF name or path;
- ``encode_sample(encoder, sample, media_clip, generator, args)``: encode one SFT ``Sample`` (``prompt``,
  ``conditions``, ``target``, all media already local) and its target clip
  ``{"video": [C, T, H, W] uint8, "fps", "frame_times_seconds", "audio_sample_rate"}`` into the cached train
  sample (clean latents keyed by stream name, e.g. ``{"visual": ...}``, + cond kwargs). Inputs the family cannot
  train on (e.g. conditions) are rejected, never dropped;
- ``validate_args(args)``: family-specific encode constraints.
"""


def get_encoder(family: str | None, task: str | None):
    if family == "wan2_2":
        from miles.rollout.encoder_hub import wan2_2

        return wan2_2
    if family == "h3" and task == "ref2va":
        from miles.rollout.encoder_hub.h3 import ref2va

        return ref2va
    if family == "h3":
        from miles.rollout.encoder_hub.h3 import t2va

        return t2va
    raise ValueError(f"no encoder_hub entry for model family {family!r}")
