"""Cosmos3 bitwise-parity patches: the sgl-d rollout group (`apply`) and the train-side halves (`apply_train`).

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
  * LoRA runs as adapter GEMMs instead of a weight merge. The trainer's peft
    forward is ``base(x) + lora_B(lora_A(x))·s`` (three GEMMs); a merged
    ``GEMM(W + sBA)`` rounds differently, so merged sync caps parity at the
    first step (B starts at 0). The recipe keeps the wrappers unmerged via
    ``--sglang-lora-merge-mode dynamic``; with ``--lora-ipc-weight-sync`` the
    trainer ships lora_A/lora_B through the engine's native LoRA-IPC path,
    rounded to the train forward dtype before send (the rounding FSDP's
    mixed-precision gather applies before the train forward). The wrapper
    forwards replay peft's exact op sequence per target — fused targets
    (add_q/k/v -> to_qkv) arrive as one block-diagonal composed pair and each
    section's delta lands on its output slice.
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
    _patch_lora_peft_forwards()


def apply_train(model: torch.nn.Module) -> None:
    """Train-side halves of the parity contract; called from the cosmos3 config's postprocess."""
    _wrap_time_embedder_row_dedup(model.time_embedder)
    _patch_diffusers_rmsnorm_fp32_through_mul()
    _round_lora_ipc_sends_to_forward_dtype()


def _wrap_time_embedder_row_dedup(time_embedder: torch.nn.Module) -> None:
    """Dedupe identical per-token sinusoid rows to sglang-d's M=1 GEMM shape (cuBLAS is not M-invariant)."""
    orig_forward = time_embedder.forward

    def forward(x, *args, **kwargs):
        # Autocast off: bf16 autocast would undo the fp32 weight gather at the matmul boundary.
        with torch.autocast("cuda", enabled=False):
            if torch.is_tensor(x) and x.ndim == 2 and x.shape[0] > 1 and torch.equal(x, x[:1].expand_as(x)):
                out = orig_forward(x[:1], *args, **kwargs)
                return out.expand(x.shape[0], *out.shape[1:])
            return orig_forward(x, *args, **kwargs)

    time_embedder.forward = forward


def _patch_diffusers_rmsnorm_fp32_through_mul() -> None:
    """Route diffusers RMSNorm through F.rms_norm (fp32 through the weight mul), same op as _patch_rmsnorm_f_rms_norm."""
    from diffusers.models import normalization

    if getattr(normalization.RMSNorm, "_miles_fp32_through_mul", False):
        return

    orig_forward = normalization.RMSNorm.forward

    def forward(self, hidden_states):
        if self.weight is not None and self.bias is None:
            return torch.nn.functional.rms_norm(hidden_states, self.dim, self.weight, self.eps)
        return orig_forward(self, hidden_states)

    normalization.RMSNorm.forward = forward
    normalization.RMSNorm._miles_fp32_through_mul = True


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


def _lora_term(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, s: float) -> torch.Tensor:
    """peft vanilla LoRA (0.18.x), op for op: ``lora_B(lora_A(x)) * scaling``.

    The trainer's forward sees the fp32 adapter masters FSDP-gathered at the
    forward dtype; the sender ships A/B rounded to that same dtype (see
    ``_round_lora_ipc_sends_to_forward_dtype``), so both sides execute the
    same two bf16 GEMMs and the same elementwise multiply. The base output is
    added by the caller as ``base + term`` — same order as peft's
    ``result + ...``.
    """
    return F.linear(F.linear(x, A), B) * s


def _round_lora_ipc_sends_to_forward_dtype() -> None:
    from miles.backends.fsdp_utils.diffusion_update_weight_utils import DiffusionUpdateWeightFromTensorLoRAIPC
    from miles.backends.fsdp_utils.mixed_precision import parse_dtype_from_str

    if getattr(DiffusionUpdateWeightFromTensorLoRAIPC, "_miles_rounded_lora_send", False):
        return

    orig_prepare = DiffusionUpdateWeightFromTensorLoRAIPC._prepare_lora_param

    def _prepare(self, param: torch.Tensor) -> torch.Tensor:
        return orig_prepare(self, param).to(parse_dtype_from_str(self.args.diffusion_forward_dtype))

    DiffusionUpdateWeightFromTensorLoRAIPC._prepare_lora_param = _prepare
    DiffusionUpdateWeightFromTensorLoRAIPC._miles_rounded_lora_send = True


def _wrapper_scaling(layer) -> float:
    # peft's ``scaling = lora_alpha / r`` (exact python-float division); the
    # engine-side strength knob stays folded in for completeness (1.0 here).
    scaling = layer.lora_alpha / layer.lora_rank
    if layer.strength != 1.0:
        scaling = scaling * layer.strength
    return scaling


def _patch_lora_peft_forwards() -> None:
    """Replay peft's unmerged op order in the engine's LoRA wrappers.

    Stock wrapper forwards run under ``@torch.compile``, which re-fuses even
    the no-adapter base path — every wrapper forward must be replaced. Fused
    targets (add_q/k/v -> to_qkv) arrive as one block-diagonal composed pair
    (scale folded into B, exact for power-of-two alpha/rank); each section's
    delta lands on its output slice — elementwise identical to the train
    side's per-projection ``base + delta`` before concat. Bitwise parity is a
    tp_size==1 property; sharded layers fall back to the native TP-aware
    forwards.
    """
    from sglang.multimodal_gen.runtime.layers.lora.linear import (
        BaseLayerWithLoRA,
        ColumnParallelLinearWithLoRA,
        LinearWithLoRA,
        MergedColumnParallelLinearWithLoRA,
        RowParallelLinearWithLoRA,
    )

    orig_column_forward = ColumnParallelLinearWithLoRA.forward
    orig_row_forward = RowParallelLinearWithLoRA.forward
    orig_merged_forward = MergedColumnParallelLinearWithLoRA.forward

    def _tuple_lora_forward(self, x: torch.Tensor):
        out, output_bias = self.base_layer(x)
        if not self.merged and not self.disable_lora:
            # After the complete base output (bias included) — the position
            # peft adds the delta at.
            out = out + _lora_term(x, self.lora_A, self.lora_B, _wrapper_scaling(self))
        return out, output_bias

    def _nn_linear_lora_forward(self, x: torch.Tensor):
        out = self.base_layer(x)
        if not self.merged and not self.disable_lora:
            out = out + _lora_term(x, self.lora_A, self.lora_B, _wrapper_scaling(self))
        return out

    def _column_parallel_lora_forward(self, x: torch.Tensor):
        if self.base_layer.tp_size > 1:
            return orig_column_forward(self, x)
        return _tuple_lora_forward(self, x)

    def _row_parallel_lora_forward(self, x: torch.Tensor):
        if self.base_layer.tp_size > 1:
            return orig_row_forward(self, x)
        return _tuple_lora_forward(self, x)

    def _merged_column_lora_forward(self, x: torch.Tensor):
        if self.base_layer.tp_size > 1:
            return orig_merged_forward(self, x)
        out, output_bias = self.base_layer(x)
        if not self.merged and not self.disable_lora:
            sizes = self.base_layer.output_sizes
            rank = self.lora_A.shape[0] // len(sizes)
            scaling = _wrapper_scaling(self)
            row = col = 0
            for size in sizes:
                a = self.lora_A[col : col + rank]
                # Contiguous copy keeps the GEMM layout identical to the train
                # side's standalone per-projection GEMM.
                b = self.lora_B[row : row + size, col : col + rank].contiguous()
                out[..., row : row + size] += _lora_term(x, a, b, scaling)
                row += size
                col += rank
        return out, output_bias

    BaseLayerWithLoRA.forward = _tuple_lora_forward
    ColumnParallelLinearWithLoRA.forward = _column_parallel_lora_forward
    RowParallelLinearWithLoRA.forward = _row_parallel_lora_forward
    MergedColumnParallelLinearWithLoRA.forward = _merged_column_lora_forward
    LinearWithLoRA.forward = _nn_linear_lora_forward


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
    if hasattr(layernorm, "can_use_fused_inplace_qknorm_rope"):
        # Newer trees add a fused qknorm+rope kernel with its own guard; the
        # delegate below bypasses its only cosmos3 call site, this is belt and
        # braces should another path reach apply_qk_norm_rope.
        layernorm.can_use_fused_inplace_qknorm_rope = lambda *args, **kwargs: False

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
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import Cosmos3DenoisingStage

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
        **extra,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        del max_text_seq_len  # per-branch true length, recomputed from each mask
        # Omni-era conditioning (sound/action latents and friends) is identical
        # across CFG branches — the batched impl torch.cat's each with itself —
        # so it passes through per-branch unchanged. Drop the Nones so the same
        # code runs on trees whose _run_transformer predates these kwargs.
        extra = {key: value for key, value in extra.items() if value is not None}

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
                **extra,
            )

        noise_pred_uncond = run(uncond_text_ids, uncond_text_mask, "uncond")
        _dumper_step_between_branches()
        noise_pred_cond = run(cond_text_ids, cond_text_mask, "cond")

        # CFG: uncond + g·(cond − uncond) — same op order as the train side's
        # cfg_combine and the original batched combine.
        def combine(uncond, cond):
            return uncond + guidance_scale * (cond - uncond)

        if isinstance(noise_pred_cond, tuple):
            return tuple(combine(u, c) for u, c in zip(noise_pred_uncond, noise_pred_cond, strict=True))
        return combine(noise_pred_uncond, noise_pred_cond)

    Cosmos3DenoisingStage._predict_noise_cfg_batched = _predict_noise_cfg_batched
