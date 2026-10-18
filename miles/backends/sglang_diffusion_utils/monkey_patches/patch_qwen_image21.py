"""Qwen-Image 2.1 rollout patches: recompute the text prefix every step, like the trainer.

The engine caches every layer's prefix K/V after the first denoising step, while the trainer
(diffusers) reruns the full joint sequence every step; the cache moves
``train/log_prob_mean_abs_diff``. Passing ``prefix_caches=None`` makes the DiT recompute the prefix
every step and store nothing, so no K/V reaches the rollout env either.

The env also snapshots ``layouts`` (RoPE tables the trainer rebuilds itself), and the rollout API
copies that per-sample list whole into every sample's response.
"""

from __future__ import annotations


def apply() -> None:
    from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import QwenImage21PipelineConfig

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return batch.extra["qwen21_positive"] | {"prefix_caches": None}

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype=None):
        return batch.extra["qwen21_negative"] | {"prefix_caches": None}

    def gather_denoising_env_static_for_sp(self, batch, cond_kwargs):
        return {key: value for key, value in cond_kwargs.items() if key != "layouts"}

    QwenImage21PipelineConfig.prepare_pos_cond_kwargs = prepare_pos_cond_kwargs
    QwenImage21PipelineConfig.prepare_neg_cond_kwargs = prepare_neg_cond_kwargs
    QwenImage21PipelineConfig.gather_denoising_env_static_for_sp = gather_denoising_env_static_for_sp
