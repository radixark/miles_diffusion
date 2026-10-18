"""Krea2 training pipeline config."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config


def _krea2_attention_with_gqa(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
    """diffusers 0.40's ``Krea2AttnProcessor.__call__``: SDPA takes the 12 K/V heads with ``enable_gqa``.

    0.41 repeats K/V to all 48 heads before SDPA, so every layer saves 4x the K/V for backward.
    """
    from diffusers.models.transformers import transformer_krea2

    query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
    key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
    value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
    gate = attn.to_gate(hidden_states)

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    if image_rotary_emb is not None:
        query = transformer_krea2.apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
        key = transformer_krea2.apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    # Looked up on the module so the sequence-parallel dispatch wrapper still applies.
    hidden_states = transformer_krea2.dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        enable_gqa=attn.num_heads != attn.num_kv_heads,
        backend=self._attention_backend,
        parallel_config=self._parallel_config,
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states * torch.sigmoid(gate)
    return attn.to_out[0](hidden_states)


@register_train_pipeline_config("krea2")
class Krea2TrainPipelineConfig(TrainPipelineConfig):
    tasks = ("t2i",)
    hf_ckpt_name_patterns = ("krea-2",)

    @classmethod
    def validate_args(cls, args) -> None:
        # The engine's krea2 pipeline has no per-request output expansion yet.
        if args.rollout_microgroup_size != 1:
            raise ValueError("krea2 rollout serves one sample per request; use --rollout-microgroup-size 1")

    def process_sigma_as_timesteps_input(self, sigmas: torch.Tensor) -> torch.Tensor:
        # The DiT takes flow time in [0, 1]; its sinusoidal embed applies the x1000 itself.
        return sigmas

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None:
            return {}
        kwargs = {}
        if cond.encoder_hidden_states:
            # (seq_len, num_text_layers, dim) -> (1, seq_len, num_text_layers, dim)
            kwargs["encoder_hidden_states"] = cond.encoder_hidden_states[0].to(device).unsqueeze(0)
        if cond.pos is not None:
            kwargs["position_ids"] = cond.pos.to(device)
        if cond.encoder_attention_mask:
            kwargs["encoder_attention_mask"] = (
                cond.encoder_attention_mask[0].to(device=device, dtype=torch.bool).unsqueeze(0)
            )
        return kwargs

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
        pad_to_len: int | None = None,
    ) -> dict:
        img_pos = None
        for kw in per_sample_cond_kwargs:
            # position_ids = zero rows for text, then the latent-grid rows; the grid is
            # resolution-only, so one batch shares it and text rows pad with more zeros.
            rows = kw["position_ids"][kw["encoder_hidden_states"].shape[1] :]
            if img_pos is None:
                img_pos = rows
            elif rows.shape != img_pos.shape:
                raise ValueError(
                    f"collate expects one image resolution per batch, "
                    f"got grids {tuple(rows.shape)} vs {tuple(img_pos.shape)}"
                )

        max_len = max(kw["encoder_hidden_states"].shape[1] for kw in per_sample_cond_kwargs)
        if pad_to_len is not None:
            max_len = max(max_len, int(pad_to_len))
        encs = []
        masks = []
        for kw in per_sample_cond_kwargs:
            enc = kw["encoder_hidden_states"]
            mask = kw["encoder_attention_mask"]
            pad = max_len - enc.shape[1]
            encs.append(F.pad(enc, (0, 0, 0, 0, 0, pad)) if pad else enc)
            masks.append(F.pad(mask, (0, pad)) if pad else mask)
        mask = torch.cat(masks, dim=0).to(device)
        return {
            "encoder_hidden_states": torch.cat(encs, dim=0).to(device),
            # All-valid text runs maskless, matching the rollout DiT's fast path.
            "encoder_attention_mask": None if bool(mask.all()) else mask,
            "position_ids": torch.cat([img_pos.new_zeros(max_len, 3), img_pos], dim=0).to(device),
        }

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
        from diffusers.models.transformers import transformer_krea2

        transformer_krea2.Krea2AttnProcessor.__call__ = _krea2_attention_with_gqa
