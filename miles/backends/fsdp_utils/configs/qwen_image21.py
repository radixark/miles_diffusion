"""Qwen-Image 2.1 training pipeline config.

T2I Flow-GRPO. The rollout engine (sglang PR 39983) emits Qwen3-VL text features;
the trainer loads ``QwenImage21Transformer2DModel`` from diffusers PR 14804 and
rebuilds the joint ``img_mask`` / ``img_shapes`` that the single-stream DiT needs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config

_VAE_SCALE = 16
_IMG_TOKENS_PER_SLOT = 4


def _rebuild_qwen21_rope_on_cuda(model) -> None:
    """Rebuild ``QwenImage21Rope.freqs`` on CUDA.

    diffusers builds the 3-axis tables with CPU ``torch.pow``; sglang-d builds
    the equivalent angles on device. The fp32 ULP gap drifts every block.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    for submod in model.modules():
        if not (
            hasattr(submod, "freqs")
            and hasattr(submod, "rope_params")
            and hasattr(submod, "axes_dim")
            and hasattr(submod, "theta")
        ):
            continue
        theta = float(submod.theta)

        def _params(index: torch.Tensor, dim: int, theta: float = theta) -> torch.Tensor:
            inv = 1.0 / torch.pow(
                theta,
                torch.arange(0, dim, 2, device=device).to(torch.float32).div(dim),
            )
            freqs = torch.outer(index, inv)
            return torch.polar(torch.ones_like(freqs), freqs)

        pos_index = torch.arange(8192, device=device)
        neg_index = torch.arange(1024, device=device).flip(0) * -1 - 1
        submod.freqs = [
            torch.cat([_params(pos_index, dim), _params(neg_index, dim)], dim=0) for dim in submod.axes_dim
        ]


@register_train_pipeline_config("qwen_image21")
class QwenImage21TrainPipelineConfig(TrainPipelineConfig):
    hf_ckpt_name_patterns = (
        "qwen-image-2.1",
        "qwen-image-21",
        "qwenimage21",
        "qwen_image_21",
        "qwen_image21",
    )
    cfg_batching = False

    lora_target_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "img_mlp.proj",
        "img_mlp.out",
        "img_mlp.gate_layer",
    ]

    def __init__(self):
        self.height = 1024
        self.width = 1024

    def configure(self, args) -> None:
        self.height = int(args.diffusion_height)
        self.width = int(args.diffusion_width)

    def process_timestep_as_input(self, timesteps):
        # diffusers' 2.1 pipeline passes ``t / 1000``; sglang-d divides inside the DiT.
        return timesteps / 1000.0

    def process_sigma_as_timesteps_input(self, sigmas, *, num_train_timesteps):
        return sigmas

    def _latent_hw(self) -> tuple[int, int]:
        if self.height % _VAE_SCALE or self.width % _VAE_SCALE:
            raise ValueError(
                f"Qwen-Image 2.1 height/width must be divisible by {_VAE_SCALE}, got {self.height}x{self.width}"
            )
        return self.height // _VAE_SCALE, self.width // _VAE_SCALE

    def _target_slots(self) -> int:
        lh, lw = self._latent_hw()
        tokens = lh * lw
        if tokens % _IMG_TOKENS_PER_SLOT:
            raise ValueError(f"target latent tokens {tokens} must be divisible by {_IMG_TOKENS_PER_SLOT}")
        return tokens // _IMG_TOKENS_PER_SLOT

    def _img_shapes(self) -> list[tuple[int, int, int]]:
        lh, lw = self._latent_hw()
        return [(1, lh, lw)]

    def _append_target_img_mask(self, text_len: int, device: torch.device) -> torch.Tensor:
        # T2I: VLM sequence has no vision slots; the pipeline appends one slot per 2x2 target group.
        return torch.cat(
            [
                torch.zeros(1, text_len, dtype=torch.bool, device=device),
                torch.ones(1, self._target_slots(), dtype=torch.bool, device=device),
            ],
            dim=1,
        )

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None:
            return {}
        kwargs = {}
        if cond.encoder_hidden_states:
            enc = torch.cat(cond.encoder_hidden_states).to(device)
            if enc.ndim == 2:
                enc = enc.unsqueeze(0)
            kwargs["encoder_hidden_states"] = enc
            kwargs["img_mask"] = self._append_target_img_mask(enc.shape[1], device)
        if cond.encoder_attention_mask:
            mask = torch.cat(cond.encoder_attention_mask).to(device=device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            kwargs["encoder_hidden_states_mask"] = mask
        elif cond.txt_seq_lens and "encoder_hidden_states" in kwargs:
            text_len = kwargs["encoder_hidden_states"].shape[1]
            L = int(cond.txt_seq_lens[0])
            kwargs["encoder_hidden_states_mask"] = torch.arange(text_len, device=device).unsqueeze(0) < L
        kwargs["img_shapes"] = [self._img_shapes()]
        return kwargs

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
        pad_to_len: int | None = None,
    ) -> dict:
        """Pad text embeds, then append the same target-slot img_mask to every row."""
        encs: list[torch.Tensor] = []
        text_masks: list[torch.Tensor] = []
        seq_lens: list[int] = []
        for kw in per_sample_cond_kwargs:
            enc = kw["encoder_hidden_states"]
            assert enc.shape[0] == 1, f"collate expects batch=1 encoder_hidden_states, got {tuple(enc.shape)}"
            encs.append(enc)
            text_len = enc.shape[1]
            mask = kw.get("encoder_hidden_states_mask")
            if mask is None:
                mask = torch.ones(1, text_len, dtype=torch.bool, device=enc.device)
            text_masks.append(mask.to(dtype=torch.bool))
            seq_lens.append(int(mask.sum().item()) if mask.dtype == torch.bool else text_len)

        max_len = max(enc.shape[1] for enc in encs)
        if pad_to_len is not None:
            max_len = max(max_len, int(pad_to_len))

        padded_encs = []
        padded_masks = []
        for enc, mask in zip(encs, text_masks, strict=True):
            cur = enc.shape[1]
            if cur < max_len:
                enc = F.pad(enc, (0, 0, 0, max_len - cur))
                mask = F.pad(mask, (0, max_len - cur))
            elif cur > max_len:
                enc = enc[:, :max_len, :]
                mask = mask[:, :max_len]
            padded_encs.append(enc)
            padded_masks.append(mask)

        target_slots = self._target_slots()
        img_mask = torch.cat(
            [
                torch.zeros(len(padded_encs), max_len, dtype=torch.bool, device=device),
                torch.ones(len(padded_encs), target_slots, dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        return {
            "encoder_hidden_states": torch.cat(padded_encs, dim=0).to(device),
            "encoder_hidden_states_mask": torch.cat(padded_masks, dim=0).to(device),
            "img_mask": img_mask,
            "img_shapes": [self._img_shapes()] * len(padded_encs),
        }

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
        # The 2.1 DiT projects the joint text+image sequence; keep the target image tail.
        pred = super().compute_noise_pred(
            model=model,
            latents_input=latents_input,
            timesteps_input=timesteps_input,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            joint_cond=joint_cond,
            use_cfg=use_cfg,
            cfg_batching=cfg_batching,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
        )
        target_tokens = latents_input.shape[1]
        return pred[:, -target_tokens:]

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        """CFG matching diffusers' Qwen-Image 2.1 pipeline (no cond-norm rescale)."""
        scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        return noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)

    def postprocess_model_after_materialize(self, model: torch.nn.Module) -> None:
        _rebuild_qwen21_rope_on_cuda(model)
