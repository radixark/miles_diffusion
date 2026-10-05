"""Frozen-encoder logic per model family, decoupled from the training-side TrainPipelineConfig.

Each family module provides:
- ``load_encoder(args, device)``: load the frozen encode components (tokenizer/text
  encoder/VAE) from the ``--sft-encoder-checkpoint`` HF name or path;
- ``encode_sample(encoder, sample, media_clip, generator, args)``: encode one SFT ``Sample``
  and its target clip ``{"video": [C, T, H, W] uint8, "fps", "frame_times_seconds"}``
  into the cached train sample (clean latents keyed by stream name, e.g. ``{"visual": ...}``, + cond kwargs);
- ``validate_args(args)``: family-specific encode constraints.
"""


def get_encoder(family: str | None):
    if family == "wan2_2":
        from miles.rollout.encoder_hub import wan2_2

        return wan2_2
    if family == "h3":
        from miles.rollout.encoder_hub import h3

        return h3
    raise ValueError(f"no encoder_hub entry for model family {family!r}")
