"""Disable SP-parallel VAE tiling so decode is bitwise SP-invariant.

With sp_degree > 1, ParallelTiledVAE.decode dispatches to
``parallel_tiled_decode`` (each SP rank decodes different tiles, results are
gathered and seam-blended). That path is numerically different from the
single-rank decode a 1-GPU engine runs, so rollout images stop being bitwise
equal between sp=1 and sp>1 even when the denoised latents match exactly.

For RL rollout parity we force the plain (replicated) decode path: every SP
rank decodes the full latent exactly like a 1-GPU engine would. Costs the
memory/compute savings of distributed decode — irrelevant for RL image
resolutions; revisit if video models OOM here.
"""

from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE

_original_decode = ParallelTiledVAE.decode


def _patched_decode(self, z):
    original = self.use_parallel_tiling
    self.use_parallel_tiling = False
    try:
        return _original_decode(self, z)
    finally:
        self.use_parallel_tiling = original


def apply() -> None:
    ParallelTiledVAE.decode = _patched_decode
