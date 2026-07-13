"""Disable SP-parallel VAE decode so images are bitwise SP-invariant.

With sp_degree > 1 the sgl-d VAEs dispatch to ``parallel_tiled_decode``
(each SP rank decodes different tiles, results are gathered and
seam-blended). That path is numerically different from the single-rank
decode a 1-GPU engine runs, so rollout images stop being bitwise equal
between sp=1 and sp>1 even when the denoised latents match exactly.

For RL rollout parity we force the plain (replicated) decode path: every SP
rank decodes the full latent exactly like a 1-GPU engine would. Costs the
memory/compute savings of distributed decode — irrelevant for RL image
resolutions; revisit if video models OOM here.

Two dispatch sites exist:
- ``ParallelTiledVAE.decode`` gates on ``self.use_parallel_tiling`` (generic
  models that don't override decode).
- ``AutoencoderKLQwenImage._decode_with_parallel_dispatch`` gates on its own
  ``self.use_parallel_decode`` and shadows the parent decode entirely.
WanVAE has the same subclass pattern — patch it the same way when Wan
rollout goes SP.
"""

from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
)
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE

_original_decode = ParallelTiledVAE.decode
_original_qwen_dispatch = AutoencoderKLQwenImage._decode_with_parallel_dispatch


def _patched_decode(self, z):
    original = self.use_parallel_tiling
    self.use_parallel_tiling = False
    try:
        return _original_decode(self, z)
    finally:
        self.use_parallel_tiling = original


def _patched_qwen_dispatch(self, z):
    original = self.use_parallel_decode
    self.use_parallel_decode = False
    try:
        return _original_qwen_dispatch(self, z)
    finally:
        self.use_parallel_decode = original


def apply() -> None:
    ParallelTiledVAE.decode = _patched_decode
    AutoencoderKLQwenImage._decode_with_parallel_dispatch = _patched_qwen_dispatch
