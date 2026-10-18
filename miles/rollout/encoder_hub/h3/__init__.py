"""MiniMax H3 offline SFT encode: ``common`` holds what every task shares, one module per --diffusion-task.

Each task module ends with the encoder hub interface sft_rollout calls: ``validate_args``, ``load_encoder`` and
``encode_sample``; everything above it is private to the module.
"""
