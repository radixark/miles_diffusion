"""Wan2.2 training pipeline config."""

from __future__ import annotations

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import _get_added_kv_projections, _get_qkv_projections
from miles.utils.types import CondKwargs

from ..sequence_parallel.attention import usp_attention
from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config


class WanUSPAttnProcessor:
    """Stock WanAttnProcessor with self-attention rerouted through usp_attention.

    Everything around the two attention calls is copied verbatim from
    diffusers 0.37.0 transformer_wan.WanAttnProcessor.__call__; cross-attention
    and the I2V image branch still go through dispatch_attention_fn, so the
    configured attention backend stays in charge there.
    """

    _attention_backend = None

    def __init__(self, parallel_state):
        self._parallel_state = parallel_state

    def _local_attention(self, query, key, value):
        return dispatch_attention_fn(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, backend=self._attention_backend
        )

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        is_self_attention = encoder_hidden_states is None

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        if is_self_attention:
            if attention_mask is not None:
                raise ValueError("USP self-attention does not support attention masks")
            hidden_states = usp_attention(
                query,
                key,
                value,
                self._parallel_state.ulysses_group,
                self._parallel_state.ring_group,
                local_attention_fn=self._local_attention,
                ring_backend=self._attention_backend,
            )
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


@register_train_pipeline_config("wan2_2")
class Wan2_2TrainPipelineConfig(TrainPipelineConfig):
    hf_ckpt_name_patterns = ("wan2.2", "wan-2.2")
    # High-noise expert ("transformer") handles t >= boundary, low-noise expert
    # ("transformer_2") the rest.
    boundary_ratio = 0.875
    # Wan DiT expects raw scheduler timesteps (0..num_train_timesteps), no /1000 scaling.
    needs_timestep_scaling = False

    def component_for_timestep(self, timestep: float, num_train_timesteps: int) -> str:
        if timestep >= self.boundary_ratio * num_train_timesteps:
            return "transformer"
        return "transformer_2"

    def select_guidance_scale(
        self,
        timestep: float,
        num_train_timesteps: int,
        guidance_scale: float,
        guidance_scale_2: float | None,
    ) -> float:
        if timestep >= self.boundary_ratio * num_train_timesteps:
            return guidance_scale
        # Rollout backend (sglang-diffusion) uses batch.guidance_scale_2 for low-noise steps with NO fallback;
        # While high-noise and low-noise can be different;
        # A misalignment of guidance_scale_2 between training and rollout would hurt training significantly, so we require it to be set explicitly.
        assert guidance_scale_2 is not None, (
            "Wan2.2 low-noise steps require --diffusion-guidance-scale-2 "
            "(rollout already denoises them with guidance_scale_2)."
        )
        return guidance_scale_2

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None or not cond.encoder_hidden_states:
            return {}
        enc = torch.cat(cond.encoder_hidden_states).to(device)
        if enc.ndim == 2:
            enc = enc.unsqueeze(0)
        return {"encoder_hidden_states": enc}

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
        pad_to_len: int | None = None,  # accepted for interface parity (PR #10); Wan2.2 concats fixed-length T5 embeds
    ) -> dict:
        encs = [kw["encoder_hidden_states"] for kw in per_sample_cond_kwargs]
        return {"encoder_hidden_states": torch.cat(encs, dim=0).to(device)}

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        return noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)

    def apply_sp_attention(self, transformer, parallel_state) -> None:
        base = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
        processor = WanUSPAttnProcessor(parallel_state)
        # set_attention_backend ran before SP install; carry its choice over.
        processor._attention_backend = next(iter(base.attn_processors.values()))._attention_backend
        base.set_attn_processor(processor)
