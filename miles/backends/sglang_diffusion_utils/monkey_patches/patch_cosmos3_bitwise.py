"""Cosmos3 bitwise-parity patches for sgl-d (train-side reference: diffusers).

Direction discipline (never downgrade precision):

- Genuine precision-policy gaps are fixed on the LOW side. The only one found
  is the train side's time_embedder (fixed there via the FSDP precision spec;
  sgl-d already runs it fp32 — nothing to patch here).
- Kernel-organization differences are re-expressed on the sgl-d side as the
  exact op sequence diffusers runs, at equal precision:

  * ``MergedColumnParallelLinear`` fuses Q/K/V (and gate/up) into one GEMM.
    Measured on cosmos3's UND shapes (M=29, bf16, H200): the fused GEMM's Q
    columns differ from the standalone Q GEMM by 3.5e-3 rel. Unfuse into
    per-slice ``F.linear`` calls — each slice then runs the same GEMM the
    diffusers module runs, bitwise.
  * ``RMSNorm`` (flashinfer ``rmsnorm`` / ``fused_add_rmsnorm``) is rerouted
    through ``F.rms_norm`` with the residual add made explicit in the input
    dtype. The train side patches diffusers' RMSNorm onto the same
    ``F.rms_norm`` (raising it from round-before-mul to fp32-through-mul), so
    both stacks run the identical kernel.
  * ``SiluAndMul`` (fused sgl-kernel, one rounding) becomes eager
    ``F.silu(gate) * up`` (two roundings) — diffusers' exact op order.
  * The fused qk-norm+rope JIT kernels are disabled; the split path runs the
    same ``F.rms_norm`` + eager rope muls as diffusers.
  * The GEN attention backend is pinned to TORCH_SDPA via backend selection
    (the sanctioned channel — see monkey_patches.__init__ on why USPAttention
    itself must not be patched); diffusers dispatches to the same
    ``F.scaled_dot_product_attention``.
  * CFG runs cond/uncond as two sequential batch-1 forwards instead of one
    batch-2 forward. cuBLAS is not bitwise batch-invariant on cosmos3's
    shapes (down_proj M=29->58 and proj_out M=390->780 both break), and the
    train side is single-sample by construction (``compute_noise_pred``
    asserts ``not cfg_batching``). This also removes the uncond text padding
    (11 -> 29) that batch-2 forced.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply() -> None:
    _force_torch_sdpa_backend()
    _patch_rmsnorm_f_rms_norm()
    _patch_merged_column_linear_unfused()
    _patch_silu_and_mul_eager()
    _patch_qk_norm_rope_split_eager()
    _patch_cfg_sequential()


def _force_torch_sdpa_backend() -> None:
    from sglang.multimodal_gen.runtime.layers.attention.selector import global_force_attn_backend
    from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum

    global_force_attn_backend(AttentionBackendEnum.TORCH_SDPA)


def _patch_rmsnorm_f_rms_norm() -> None:
    from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm

    def _forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if self.variance_size_override is not None:
            raise NotImplementedError("cosmos3_bitwise RMSNorm patch does not support variance_size_override")
        if residual is not None:
            # Same add the diffusers layer runs eagerly (single bf16 rounding).
            residual = x + residual
            out = F.rms_norm(residual, (self.hidden_size,), self.weight, self.variance_epsilon)
            return out, residual
        return F.rms_norm(x, (self.hidden_size,), self.weight, self.variance_epsilon)

    RMSNorm.forward_cuda = _forward
    RMSNorm.forward_native = _forward


def _patch_merged_column_linear_unfused() -> None:
    from sglang.multimodal_gen.runtime.layers.linear import MergedColumnParallelLinear, UnquantizedLinearMethod

    logged = False

    def _forward(self, x: torch.Tensor):
        # NOT self.output_partition_sizes: with tp=1, MergedColumnParallelLinear
        # assigns self.output_sizes only after super().__init__() has already
        # derived output_partition_sizes, so that attr collapses to
        # [sum(output_sizes)] and a "per-slice" loop over it degenerates into
        # the very fused GEMM this patch exists to avoid.
        sizes = getattr(self, "output_sizes", None)
        if not isinstance(self.quant_method, UnquantizedLinearMethod) or self.skip_bias_add or sizes is None:
            raise RuntimeError(
                "cosmos3_bitwise unfused-GEMM patch cannot handle this MergedColumnParallelLinear "
                f"(quant_method={type(self.quant_method).__name__}, skip_bias_add={self.skip_bias_add}, "
                f"output_sizes={sizes}); refusing to fall back to the fused GEMM silently."
            )
        sizes = [size // self.tp_size for size in sizes]
        nonlocal logged
        if not logged:
            logged = True
            print(f"[cosmos3_bitwise] unfused MergedColumnParallelLinear active: slices={sizes}", flush=True)
        outs = []
        offset = 0
        for size in sizes:
            bias = self.bias[offset : offset + size] if self.bias is not None else None
            outs.append(F.linear(x, self.weight[offset : offset + size], bias))
            offset += size
        return torch.cat(outs, dim=-1), None

    MergedColumnParallelLinear.forward = _forward


def _patch_silu_and_mul_eager() -> None:
    from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    SiluAndMul.forward_cuda = _forward
    SiluAndMul.forward_native = _forward


def _patch_qk_norm_rope_split_eager() -> None:
    from sglang.multimodal_gen.runtime.layers import layernorm
    from sglang.multimodal_gen.runtime.models.dits import cosmos3video

    # The split path's apply_qk_norm falls back to the (patched) RMSNorm
    # modules once the fused inplace JIT kernel is declared unavailable.
    layernorm.can_use_fused_inplace_qknorm = lambda *args, **kwargs: False

    def _delegate(q, k, q_norm, k_norm, head_dim, cos_sin_cache, rope_cache_positions):
        return cosmos3video._apply_qwen3_qk_norm_rope_split(q, k, q_norm, k_norm, head_dim, cos_sin_cache)

    cosmos3video._apply_qwen3_qk_norm_rope = _delegate


def _dumper_step_between_branches() -> None:
    # The denoising stage's Dumper instrumentation steps once per loop
    # iteration; sequential CFG puts two forwards in one iteration, which
    # would collide record names. Step between the branches so every dumper
    # step holds exactly one forward (uncond and cond land in adjacent steps;
    # the comparator pairs by bit-exact latent anchor, not by step index).
    try:
        from sglang.srt.debug_utils.dumper import dumper
    except ImportError:
        return
    if dumper.may_enable and dumper._non_intrusives:
        dumper.step()


def _patch_cfg_sequential() -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
        Cosmos3DenoisingStage,
    )

    def _predict_noise_cfg_batched(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        cond_text_ids: torch.Tensor,
        cond_text_mask: torch.Tensor,
        uncond_text_ids: torch.Tensor,
        uncond_text_mask: torch.Tensor,
        video_shape: tuple[int, int, int],
        fps: float,
        guidance_scale: float,
        noisy_frame_mask: torch.Tensor | None = None,
        max_text_seq_len: int | None = None,
        current_timestep: int | None = None,
    ) -> torch.Tensor:
        del max_text_seq_len  # per-branch true length, recomputed from each mask

        def run(text_ids, text_mask, cache_key):
            return self._run_transformer(
                latents=latents,
                timestep=timestep,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=video_shape,
                fps=fps,
                cache_key=cache_key,
                noisy_frame_mask=noisy_frame_mask,
                max_text_seq_len=None,
                current_timestep=current_timestep,
            )

        noise_pred_uncond = run(uncond_text_ids, uncond_text_mask, "uncond")
        _dumper_step_between_branches()
        noise_pred_cond = run(cond_text_ids, cond_text_mask, "cond")
        # CFG: uncond + g·(cond − uncond) — same op order as the train side's
        # cfg_combine and the original batched combine.
        return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    Cosmos3DenoisingStage._predict_noise_cfg_batched = _predict_noise_cfg_batched
