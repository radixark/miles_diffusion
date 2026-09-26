"""Qwen-Image 2.1 rollout patches: full-prefill every step + fused kernels off.

Train-side (diffusers) reruns the full joint sequence every SDE step. The
engine's request-scoped prefix cache and bit-exact fused kernels are faster
for serving but move ``train/log_prob_mean_abs_diff``. Keep the
``prefix_caches`` kwarg (the denoising stage indexes it) and wipe the dicts
before each predict so every step prefills.
"""

from __future__ import annotations


def apply() -> None:
    from sglang.multimodal_gen.runtime.layers.lora import linear as lora_linear
    from sglang.multimodal_gen.runtime.models.dits import qwen_image21 as qwen21_mod
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
        QwenImage21DenoisingStage,
    )

    from miles.backends.sglang_diffusion_utils.monkey_patches import patch_qwen_image

    for gate_name in ("_ROPE_FUSION", "_SILU_MUL_FUSION", "_QK_NORM_FUSION", "_MODULATION_FUSION"):
        gate = getattr(qwen21_mod, gate_name, None)
        if gate is not None and hasattr(gate, "disable"):
            gate.disable()

    orig_predict = QwenImage21DenoisingStage._predict_noise

    def _predict_noise(self, current_model, latent_model_input, timestep, target_dtype, guidance, **kwargs):
        caches = kwargs.get("prefix_caches")
        if caches:
            for sample_caches in caches:
                for layer in sample_caches:
                    if isinstance(layer, dict):
                        layer.clear()
        return orig_predict(self, current_model, latent_model_input, timestep, target_dtype, guidance, **kwargs)

    QwenImage21DenoisingStage._predict_noise = _predict_noise

    # Same PEFT-ordered LoRA add as Qwen-Image 1.0 so IPC-synced adapters match train.
    lora_linear.BaseLayerWithLoRA.forward = patch_qwen_image._lora_base_forward
    lora_linear.RowParallelLinearWithLoRA.forward = patch_qwen_image._lora_row_parallel_forward
    lora_linear.ColumnParallelLinearWithLoRA.forward = patch_qwen_image._lora_column_parallel_forward
    lora_linear.LinearWithLoRA.forward = patch_qwen_image._lora_nn_linear_forward
