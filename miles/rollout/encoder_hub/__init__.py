"""Frozen-encoder logic per model family, decoupled from the training-side TrainPipelineConfig.

Each family module provides:
- ``load_encoder(args, device)``: load the frozen encode components (tokenizer/text
  encoder/VAE) from the ``--sft-encoder-checkpoint`` HF name or path;
- ``encode_sample(encoder, pixels, prompt, generator)``: encode one media/prompt pair
  into a cached train sample (clean latent + cond kwargs);
- ``validate_args(args)``: family-specific encode constraints.
"""


def get_encoder(family: str | None):
    if family == "wan2_2":
        from miles.rollout.encoder_hub import wan2_2

        return wan2_2
    raise ValueError(f"no encoder_hub entry for model family {family!r}")
