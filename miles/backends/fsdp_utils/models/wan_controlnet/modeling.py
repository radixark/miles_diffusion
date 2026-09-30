"""A small, independently checkpointed ControlNet for Diffusers Wan.

This is a new control architecture, not a loader for pretrained pose weights.
Selected pretrained Wan blocks initialize the branch; every residual output
projection starts at zero. The frozen backbone therefore produces exactly its
original output at initialization. We explicitly reuse Wan's modules and forward
algebra instead of installing hooks that can leak across calls or recomputations.
"""

from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class WanPoseControlNet(nn.Module):
    """Pose patch embedding, copied Wan blocks, and zero output projections."""

    def __init__(self, backbone, block_indices: Sequence[int]):
        super().__init__()
        self.block_indices = tuple(int(index) for index in block_indices)
        if not self.block_indices or tuple(sorted(set(self.block_indices))) != self.block_indices:
            raise ValueError("control block_indices must be a nonempty, strictly increasing sequence")
        if self.block_indices[0] < 0 or self.block_indices[-1] >= len(backbone.blocks):
            raise ValueError("control block index is outside the Wan backbone")
        self.patch_embedding = copy.deepcopy(backbone.patch_embedding)
        self.blocks = nn.ModuleList(copy.deepcopy(backbone.blocks[index]) for index in self.block_indices)
        dim = backbone.config.num_attention_heads * backbone.config.attention_head_dim
        prototype = backbone.patch_embedding.weight
        self.output_projections = nn.ModuleList(
            nn.Linear(dim, dim, device=prototype.device, dtype=prototype.dtype) for _ in self.block_indices
        )
        for projection in self.output_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.requires_grad_(True)

    def forward(
        self,
        noisy_tokens,
        control_latents,
        encoder_hidden_states,
        timestep_proj,
        rotary_emb,
        *,
        gradient_checkpointing=False,
    ):
        pose_tokens = self.patch_embedding(control_latents).flatten(2).transpose(1, 2)
        hidden_states = noisy_tokens + pose_tokens
        residuals = []
        for block, projection in zip(self.blocks, self.output_projections, strict=True):
            if torch.is_grad_enabled() and gradient_checkpointing:
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            residuals.append(projection(hidden_states))
        return residuals


class WanPoseControlTransformer(nn.Module):
    """Wan-compatible transformer accepting aligned ``control_latents``.

    ``control_latents`` has the same B,C,T,H,W shape as ``hidden_states`` and
    contains clean VAE latents of rendered skeleton video. It is not the RGB
    target video, a reference-frame latent, or an independently noised input.
    """

    _no_split_modules = ["WanTransformerBlock"]

    def __init__(self, backbone, *, num_control_blocks: int = 4, block_indices: Sequence[int] | None = None):
        super().__init__()
        if block_indices is None:
            if not 1 <= num_control_blocks <= len(backbone.blocks):
                raise ValueError("num_control_blocks must be between one and the backbone layer count")
            block_indices = tuple(i * len(backbone.blocks) // num_control_blocks for i in range(num_control_blocks))
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.config = backbone.config
        self.controlnet = WanPoseControlNet(backbone, block_indices)
        self.gradient_checkpointing = False

    @property
    def dtype(self):
        return self.backbone.dtype

    @property
    def device(self):
        return self.backbone.device

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def set_attention_backend(self, backend: str):
        # Diffusers walks self.modules(), including both backbone and branch.
        type(self.backbone).set_attention_backend(self, backend)

    def cache_context(self, name):
        return self.backbone.cache_context(name)

    @contextmanager
    def control_condition(self, control_latents, *, control_scale=1.0):
        """Bind inference-only conditions while an unmodified WanPipeline runs.

        The context is local to this wrapper instance and restores prior state
        even if generation raises. Concurrent requests need separate wrappers.
        """
        previous = getattr(self, "_inference_control", None)
        self._inference_control = (control_latents, control_scale)
        try:
            yield self
        finally:
            self._inference_control = previous

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict=True,
        attention_kwargs=None,
        *,
        control_latents=None,
        control_scale=1.0,
    ):
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        inference_control = getattr(self, "_inference_control", None)
        if control_latents is None and inference_control is not None:
            control_latents, control_scale = inference_control
        if control_latents is None or control_scale == 0:
            return self.backbone(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
            )
        if attention_kwargs:
            raise ValueError("Wan pose-control does not implement nonempty attention_kwargs or LoRA scaling")
        if control_latents.shape != hidden_states.shape:
            raise ValueError(
                f"control_latents must align exactly with noisy target latents: "
                f"{tuple(control_latents.shape)} != {tuple(hidden_states.shape)}"
            )
        if control_latents.device != hidden_states.device or control_latents.dtype != hidden_states.dtype:
            raise ValueError("control_latents must share the noisy target device and dtype")

        base = self.backbone
        batch, _, frames, height, width = hidden_states.shape
        pt, ph, pw = base.config.patch_size
        if frames % pt or height % ph or width % pw:
            raise ValueError("latent spatial/temporal dimensions must be divisible by Wan patch_size")
        rotary_emb = base.rope(hidden_states)
        noisy_tokens = base.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        ts_seq_len = timestep.shape[1] if timestep.ndim == 2 else None
        if ts_seq_len is not None:
            timestep = timestep.flatten()
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = base.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        timestep_proj = timestep_proj.unflatten(2 if ts_seq_len is not None else 1, (6, -1))
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.cat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        residuals = self.controlnet(
            noisy_tokens,
            control_latents,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        residual_at = dict(zip(self.controlnet.block_indices, residuals, strict=True))
        hidden_states = noisy_tokens
        for index, block in enumerate(base.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            if index in residual_at:
                hidden_states = hidden_states + residual_at[index] * control_scale

        if temb.ndim == 3:
            shift, scale = (base.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift, scale = shift.squeeze(2), scale.squeeze(2)
        else:
            shift, scale = (base.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        shift, scale = shift.to(hidden_states.device), scale.to(hidden_states.device)
        hidden_states = (base.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = base.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, frames // pt, height // ph, width // pw, pt, ph, pw, -1)
        output = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return Transformer2DModelOutput(sample=output) if return_dict else (output,)

    def controlnet_config(self):
        keys = ("patch_size", "in_channels", "num_layers", "num_attention_heads", "attention_head_dim", "text_dim")
        return {
            "format_version": 1,
            "architecture": "WanPoseControlNet",
            "base_model_name_or_path": self.config.get("_name_or_path", ""),
            "block_indices": list(self.controlnet.block_indices),
            "backbone_config": {key: self.config[key] for key in keys},
            "backbone_config_full": {key: value for key, value in self.config.items() if not key.startswith("_")},
        }

    def save_controlnet(self, directory, *, state_dict: Mapping[str, torch.Tensor] | None = None):
        """Export branch only. Under FSDP supply an already gathered full state.

        ``state_dict`` may be either the full wrapper state or branch-only state.
        Gathering FSDP/DTensor states is collective and belongs to the caller.
        """
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = self.controlnet.state_dict() if state_dict is None else dict(state_dict)
        if any(key.startswith("controlnet.") for key in state):
            state = {key[len("controlnet.") :]: value for key, value in state.items() if key.startswith("controlnet.")}
        expected = set(self.controlnet.state_dict())
        if set(state) != expected:
            raise ValueError("control checkpoint keys do not match this branch")
        if any(hasattr(value, "full_tensor") for value in state.values()):
            raise ValueError("gather FSDP/DTensor state collectively before exporting ControlNet")
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in state.items()},
            str(directory / "controlnet.safetensors"),
        )
        (directory / "controlnet_config.json").write_text(json.dumps(self.controlnet_config(), indent=2) + "\n")

    def load_controlnet(self, directory):
        from safetensors.torch import load_file

        directory = Path(directory)
        config = json.loads((directory / "controlnet_config.json").read_text())
        # Paths are provenance, not architecture: Diffusers may record either
        # the snapshot root or its transformer subfolder for identical weights.
        # JSON normalizes tuple patch sizes into lists.
        expected = json.loads(json.dumps(self.controlnet_config()))
        architecture_keys = ("format_version", "architecture", "block_indices", "backbone_config")
        if any(config.get(key) != expected[key] for key in architecture_keys):
            raise ValueError("control checkpoint architecture does not match this Wan wrapper")
        if "backbone_config_full" in config:
            if config["backbone_config_full"] != expected["backbone_config_full"]:
                raise ValueError("control checkpoint full backbone configuration does not match this Wan wrapper")
        else:
            warnings.warn(
                "Legacy ControlNet checkpoint lacks full backbone configuration; only dimensions can be checked. "
                "Use the original base weights and re-export the checkpoint for complete configuration validation.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.controlnet.load_state_dict(load_file(str(directory / "controlnet.safetensors")), strict=True)
        return self

    @classmethod
    def from_controlnet(cls, backbone, directory):
        config = json.loads((Path(directory) / "controlnet_config.json").read_text())
        return cls(backbone, block_indices=config["block_indices"]).load_controlnet(directory)


def enable_gradient_checkpointing(model):
    model.enable_gradient_checkpointing()


def load_scheduler(args):
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_pretrained(args.hf_checkpoint, subfolder="scheduler")
