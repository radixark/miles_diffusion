from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="stage-b-2-gpu-h200",
    labels=[],
)

import torch
from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope as DiffusersQwenEmbedRope
from sglang.multimodal_gen.runtime.models.dits.qwen_image import QwenEmbedRope as SglQwenEmbedRope

from miles.backends.fsdp_utils.configs.qwen_image import _rebuild_pos_embed_freqs_on_cuda

# Qwen-Image production config (sgl-d qwen_image.py instantiation).
THETA, AXES_DIM = 10000, [16, 56, 56]


class _Holder(torch.nn.Module):
    """Minimal stand-in for the train-side transformer: one CUDA param + the rope."""

    def __init__(self, rope):
        super().__init__()
        self.rope = rope
        self.anchor = torch.nn.Parameter(torch.zeros(1, device="cuda"))


def test_rebuilt_freqs_match_sglang_d_bitwise():
    # Rollout side: sgl-d meta-inits, then its forward rebuilds the freqs on CUDA.
    with torch.device("meta"):
        sgl_rope = SglQwenEmbedRope(theta=THETA, axes_dim=AXES_DIM, scale_rope=True)
    sgl_rope.forward((1, 8, 8), [32], torch.device("cuda"))
    assert sgl_rope.pos_freqs.device.type == "cuda"

    # Train side: diffusers builds the freqs on CPU at init (ULP-off vs CUDA);
    # _rebuild_pos_embed_freqs_on_cuda must restore bitwise equality.
    holder = _Holder(DiffusersQwenEmbedRope(theta=THETA, axes_dim=AXES_DIM, scale_rope=True))
    cpu_built = holder.rope.pos_freqs.clone()
    _rebuild_pos_embed_freqs_on_cuda(holder)

    assert holder.rope.pos_freqs.device.type == "cuda"
    assert torch.equal(holder.rope.pos_freqs, sgl_rope.pos_freqs)
    assert torch.equal(holder.rope.neg_freqs, sgl_rope.neg_freqs)
    # Premise of the fix: CPU-built freqs differ from CUDA-built by fp32 ULPs.
    # If this ever fails, the rebuild (and this test) can be retired.
    assert not torch.equal(cpu_built.cuda(), sgl_rope.pos_freqs)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
