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
  * LoRA runs as adapter GEMMs instead of a weight merge. The trainer's peft
    forward is ``base(x) + lora_B(lora_A(x))·s`` (three GEMMs); a merged
    ``GEMM(W + sBA)`` rounds differently, so merged sync caps parity at the
    first step (B starts at 0). With ``--lora-unmerged-weight-sync`` the
    trainer ships base weights untouched plus per-layer A/B/scaling tensors;
    this side intercepts them at the weight-sync loader and replays peft's
    exact op sequence per target (to_qkv slices for add_q/k/v, to_out for
    to_add_out).
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
    _patch_lora_adapter_intercept()


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
        lora = getattr(self, "_miles_lora", None)
        outs = []
        offset = 0
        for idx, size in enumerate(sizes):
            bias = self.bias[offset : offset + size] if self.bias is not None else None
            out = F.linear(x, self.weight[offset : offset + size], bias)
            if lora is not None and idx in lora:
                out = out + _lora_term(x, *lora[idx])
            outs.append(out)
            offset += size
        return torch.cat(outs, dim=-1), None

    MergedColumnParallelLinear.forward = _forward


def _lora_term(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, s: float) -> torch.Tensor:
    """peft vanilla LoRA (0.18.x), op for op: ``lora_B(lora_A(x)) * scaling``.

    The trainer runs it under bf16 autocast on FSDP-gathered bf16 adapters; the
    shipped A/B are pre-rounded to that dtype, so both sides execute the same
    two bf16 GEMMs and the same elementwise multiply. The base output is added
    by the caller as ``base + term`` — same order as peft's ``result + ...``.
    """
    return F.linear(F.linear(x, A), B) * s


def _attach_lora_adapters(module, adapters: dict[str, dict[str, torch.Tensor]]) -> None:
    """Store shipped adapter tensors on their target submodules.

    ``adapters`` maps a diffusers-style layer prefix (e.g.
    ``layers.5.self_attn.add_q_proj``) to its ``.lora_A.weight`` /
    ``.lora_B.weight`` / ``.lora_scaling`` tensors. The module's own
    param-name mapper resolves the prefix to the sgl-d parameter — including
    the merge index for slices of a fused param (add_q/k/v -> to_qkv slots
    0/1/2; to_add_out -> to_out, no index).
    """
    from sglang.multimodal_gen.runtime.post_training.weights_updater import (
        _build_module_weight_name_mapper,
    )

    map_name = _build_module_weight_name_mapper(module)
    for prefix, parts in adapters.items():
        missing = {".lora_A.weight", ".lora_B.weight", ".lora_scaling"} - set(parts)
        if missing:
            raise RuntimeError(f"cosmos3_bitwise LoRA sync: incomplete adapter for {prefix!r}: missing {missing}")
        mapped, slot = map_name(f"{prefix}.weight") if map_name is not None else (f"{prefix}.weight", None)
        target = module.get_submodule(mapped[: -len(".weight")])
        device = next(target.parameters()).device
        # clone(): the shipped tensors are views into the CUDA-IPC flattened
        # bucket, whose storage the sender reclaims after the update returns.
        A = parts[".lora_A.weight"].to(device).clone()
        B = parts[".lora_B.weight"].to(device).clone()
        s = float(parts[".lora_scaling"].item())
        registry = getattr(target, "_miles_lora", None)
        if registry is None:
            registry = {}
            target._miles_lora = registry
        registry[slot] = (A, B, s)
        if slot is None:
            _wrap_linear_instance_with_lora(target)


def _wrap_linear_instance_with_lora(target) -> None:
    """Instance-level wrap for non-fused targets (RowParallelLinear to_out):
    add the adapter term after the complete base output (bias included), the
    position peft adds it at."""
    if getattr(target, "_miles_lora_wrapped", False):
        return
    orig_forward = target.forward

    def forward(x):
        out, out_bias = orig_forward(x)
        A, B, s = target._miles_lora[None]
        return out + _lora_term(x, A, B, s), out_bias

    target.forward = forward
    target._miles_lora_wrapped = True


def _patch_lora_adapter_intercept() -> None:
    """Consume `<layer>.lora_A/lora_B/lora_scaling` tensors from weight sync.

    The trainer's --lora-unmerged-weight-sync ships base weights untouched
    plus per-layer adapter tensors (see DiffusionUpdateWeightFromTensorLoRA).
    sgl-d's loader would warn-and-drop these unknown names, so split them out
    before it runs and attach them to the resolved target modules. No-op when
    the trainer syncs merged weights.

    A layer's three parts arrive in separate calls — the sender flushes one
    flattened bucket per dtype (A/B at forward dtype, scaling fp64) and may
    also split across buffer-size flushes — so partial adapters are buffered
    until complete.
    """
    from sglang.multimodal_gen.runtime.post_training import weights_updater

    orig_load = weights_updater._load_weights_into_module
    pending: dict[str, dict[str, torch.Tensor]] = {}

    def _load(module, weights_iter):
        base_entries = []
        for name, weight in weights_iter:
            for suffix in (".lora_A.weight", ".lora_B.weight", ".lora_scaling"):
                if name.endswith(suffix):
                    pending.setdefault(name[: -len(suffix)], {})[suffix] = weight
                    break
            else:
                base_entries.append((name, weight))
        complete = {prefix: parts for prefix, parts in pending.items() if len(parts) == 3}
        if complete:
            _attach_lora_adapters(module, complete)
            for prefix in complete:
                del pending[prefix]
            print(f"[cosmos3_bitwise] attached {len(complete)} LoRA adapters (unmerged weight sync)", flush=True)
        return orig_load(module, iter(base_entries))

    weights_updater._load_weights_into_module = _load


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
