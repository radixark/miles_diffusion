"""Cosmos3 training pipeline config."""

from __future__ import annotations

import math

import torch
from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config

# Cosmos3 reuses the Wan2.2 VAE (4x temporal compression).
_VAE_TEMPORAL_FACTOR = 4

# GEN-tower parameter name fragments (diffusers Cosmos3OmniTransformer layout).
# Everything else — UND tower, lm_head, unused sound/action heads — stays frozen.
_GEN_PARAM_FRAGMENTS = (
    ".add_q_proj.",
    ".add_k_proj.",
    ".add_v_proj.",
    ".to_add_out.",
    ".norm_added_q.",
    ".norm_added_k.",
    ".mlp_moe_gen.",
    ".input_layernorm_moe_gen.",
    ".post_attention_layernorm_moe_gen.",
    ".norm_moe_gen.",
    ".proj_in.",
    ".proj_out.",
    ".time_embedder.",
)


def _is_gen_param(name: str) -> bool:
    dotted = f".{name}"
    return any(fragment in dotted for fragment in _GEN_PARAM_FRAGMENTS)


@register_train_pipeline_config("cosmos3")
class Cosmos3TrainPipelineConfig(TrainPipelineConfig):
    hf_ckpt_name_patterns = ("cosmos3", "cosmos-3")
    # process_timestep_as_input: base identity — the DiT takes the raw
    # trajectory timestep and scales by config.timestep_scale internally.
    # Timesteps stay fp32: the karras flow grid is non-integer and sgl-d
    # conditions on exact fp32 values (bf16 rounds 993.25 -> 992). Conds pass
    # through — the packed forward casts its own inputs (mRoPE position ids
    # sit at ~15000, where bf16 spacing is 128; a boundary cast scrambles
    # rotary phases).
    input_dtype_policy = {"latents": "default", "cond": None, "timestep": "fp32"}
    # The packed forward is single-sample by construction (compute_noise_pred
    # asserts it); never batch the CFG branches.
    cfg_batching = False
    lora_target_modules = ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
    # time_embedder gathers at fp32 via the family FSDPParallelPlan
    # (models/diffusers/cosmos3/parallel_plan.py); rollout parity patches ship
    # as the `cosmos3_bitwise` group, selected with --rollout-patch-group.

    @classmethod
    def validate_args(cls, args) -> None:
        if list(args.update_weight_target_modules) != ["transformer"]:
            raise ValueError("Cosmos3 requires --update-weight-target-module transformer.")

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None or cond.text_ids is None:
            return {}
        return {
            "text_ids": cond.text_ids.to(device),
            "text_mask": cond.text_mask.to(device),
            "fps": cond.fps,
        }

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
        pad_to_len: int | None = None,
    ) -> dict:
        # Packed single-sample forward: keep per-sample kwargs; no batched tensor to build.
        return {"per_sample": per_sample_cond_kwargs}

    def compute_noise_pred(
        self,
        *,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        pos_cond: dict | None,
        neg_cond: dict | None,
        joint_cond: dict | None,
        use_cfg: bool,
        cfg_batching: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
    ) -> torch.Tensor:
        assert not cfg_batching, "Cosmos3 packed forward is single-sample; cfg_batching unsupported"
        config = model.config
        preds = []
        for i, pos in enumerate(pos_cond["per_sample"]):
            latent = latents_input[i : i + 1]
            timestep = float(timesteps_input[i])
            pred = self._packed_forward(model, latent, timestep, pos, config)
            if use_cfg:
                pred_neg = self._packed_forward(model, latent, timestep, neg_cond["per_sample"][i], config)
                pred = self.cfg_combine(pred, pred_neg, guidance_scale, true_cfg_scale=true_cfg_scale)
            preds.append(pred)
        return torch.stack(preds, dim=0)

    def _packed_forward(
        self,
        model: torch.nn.Module,
        latent: torch.Tensor,
        timestep: float,
        cond: dict,
        config,
    ) -> torch.Tensor:
        """One (text, video) joint-sequence forward; mirrors the packing in
        diffusers' Cosmos3OmniDiffusersPipeline denoising loop (T2V/T2I: all
        frames noisy, no conditioning frames)."""
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
            get_3d_mrope_ids_text_tokens,
            get_3d_mrope_ids_vae_tokens,
        )

        device = latent.device
        und_len = int(cond["text_mask"].sum().item())
        input_ids = cond["text_ids"].reshape(-1)[:und_len]

        text_mrope_ids, next_offset = get_3d_mrope_ids_text_tokens(
            num_tokens=und_len,
            temporal_offset=0,
            use_float_positions=config.enable_fps_modulation,
        )
        vision_offset = next_offset + config.unified_3d_mrope_temporal_modality_margin

        p = config.latent_patch_size
        _, _, latent_t, latent_h, latent_w = latent.shape
        patch_h = math.ceil(latent_h / p)
        patch_w = math.ceil(latent_w / p)
        num_vision_tokens = latent_t * patch_h * patch_w

        vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=vision_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=cond["fps"] if config.enable_fps_modulation else None,
            base_fps=float(config.base_fps),
            temporal_compression_factor=_VAE_TEMPORAL_FACTOR,
        )

        sequence_length = und_len + num_vision_tokens
        vision_sequence_indexes = torch.arange(und_len, sequence_length, dtype=torch.long, device=device)
        preds_vision, _, _ = model(
            input_ids=input_ids,
            text_indexes=torch.arange(und_len, dtype=torch.long, device=device),
            position_ids=torch.cat([text_mrope_ids, vision_mrope_ids], dim=1).to(device),
            und_len=und_len,
            sequence_length=sequence_length,
            vision_tokens=[latent],
            vision_token_shapes=[(latent_t, patch_h, patch_w)],
            vision_sequence_indexes=vision_sequence_indexes,
            vision_mse_loss_indexes=vision_sequence_indexes,
            vision_timesteps=torch.full((num_vision_tokens,), timestep, device=device, dtype=torch.float32),
            vision_noisy_frame_indexes=[torch.arange(latent_t, dtype=torch.long, device=device)],
        )
        return preds_vision[0]

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        return noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)

    def postprocess_model_after_materialize(self, model: torch.nn.Module) -> None:
        # The UND tower sits inside the training forward graph (paired
        # attention), so gradients reach it unless explicitly frozen.
        for name, param in model.named_parameters():
            if "lora_" not in name and not _is_gen_param(name):
                param.requires_grad_(False)

        # sglang-d casts the fp32 timestep sinusoid to the MLP weight dtype
        # before linear_1 (`t_freq.to(w_dtype)`); mirror that exactly. With the
        # fp32 pattern in the family FSDPParallelPlan the weights gather at
        # fp32, so this keeps the sinusoid at fp32 like sglang-d's time_embedder.
        def _cast_to_weight_dtype(module, args):
            dtype = module.linear_1.weight.dtype
            return tuple(a.to(dtype) if torch.is_tensor(a) else a for a in args)

        model.time_embedder.register_forward_pre_hook(_cast_to_weight_dtype)
        _wrap_time_embedder_row_dedup(model.time_embedder)
        _patch_diffusers_rmsnorm_fp32_through_mul()


def _wrap_time_embedder_row_dedup(time_embedder: torch.nn.Module) -> None:
    """Collapse identical sinusoid rows before the timestep MLP, expand after.

    sglang-d runs the timestep MLP once per request (GEMM M=1) and broadcasts
    the embedding over tokens; diffusers expands the timestep per token first
    (M=390 for a 480x480 clip). cuBLAS fp32 GEMMs are not bitwise M-invariant
    (measured: linear_2 4096->4096 differs between M=1 and M=2), so per-token
    rows can never bit-match the rollout engine. Deduplicating is numerically
    exact — the rows are byte-identical copies — and reproduces sglang-d's
    compute shape. Gradients are unchanged up to the usual expand/sum autograd.
    """
    orig_forward = time_embedder.forward

    def forward(x, *args, **kwargs):
        # Autocast off: the trainer's bf16 autocast would cast the fp32-gathered
        # weights back to bf16 at the matmul boundary, undoing the parallel
        # plan's fp32 island. sgl-d runs this MLP at plain fp32 with no autocast.
        with torch.autocast("cuda", enabled=False):
            if torch.is_tensor(x) and x.ndim == 2 and x.shape[0] > 1 and torch.equal(x, x[:1].expand_as(x)):
                out = orig_forward(x[:1], *args, **kwargs)
                return out.expand(x.shape[0], *out.shape[1:])
            return orig_forward(x, *args, **kwargs)

    time_embedder.forward = forward


def _patch_diffusers_rmsnorm_fp32_through_mul() -> None:
    """Raise diffusers RMSNorm to fp32-through-the-weight-mul via F.rms_norm.

    diffusers' eager RMSNorm rounds the normalized activations to bf16 BEFORE
    multiplying the weight (two bf16 roundings); sglang-d keeps fp32 through
    the weight mul and rounds once. Following the "never downgrade" rule the
    train side comes up: route through torch's fused F.rms_norm (fp32
    accumulation, single rounding). The rollout patch group routes sglang-d's
    RMSNorm through the same op, so both sides run identical kernels.
    """
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
