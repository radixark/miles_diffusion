"""FLUX.3 image DiT for training: the text-to-image path of sglang-d's ``Flux3ImageDiT``.

The module tree keeps the checkpoint names, so weights load 1:1; on sync the engine fuses
q/k/v into ``to_qkv`` and the weight bridge renames the QK-norm ``scale``. The math follows
the engine's eager path: per-token AdaLN modulation, q/k RMSNorm and interleaved 4-axis RoPE
in fp32, parallel attention + SwiGLU, and a gated residual.
Image tokens come first in the joint sequence, as in the engine's packed layout.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import flash_attn_interface
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

IMAGE = "image"
TEXT = "txt"


def enable_gradient_checkpointing(model: nn.Module) -> None:
    model.gradient_checkpointing = True


@functools.cache
def flash_attn_module() -> Any:
    """The flash kernel the DiT calls: FA3 on Hopper, the pip FA4 on SM10x Blackwell."""
    major = torch.cuda.get_device_capability()[0]
    if major == 10:
        import flash_attn.cute

        return flash_attn.cute
    if major == 9:
        return flash_attn_interface
    raise NotImplementedError(f"FLUX.3 image trains on Hopper (FA3) or SM10x Blackwell (FA4), got sm_{major}x")


def is_blackwell() -> bool:
    return torch.cuda.get_device_capability()[0] == 10


def flash_attention_entrypoints(backend: str) -> list[tuple[str, Any, str]]:
    """The kernel the blocks call, which MilesModelBackend patches with ``deterministic=True``."""
    return [(required_flash_kernel_label(backend), flash_attn_module(), "flash_attn_func")]


def required_flash_kernel_label(backend: str) -> str:
    return "flash_attention_4" if is_blackwell() else "flash_attention_3"


def rope_cos_sin(ids: torch.Tensor, axes_dim: tuple[int, ...], theta: int) -> torch.Tensor:
    """Position ids (B, L, n_axes) -> fp32 (B, L, head_dim) rows [cos | sin]; pair i rotates by angle i."""
    angles = []
    for axis, dim in enumerate(axes_dim):
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=ids.device) / dim
        omega = 1.0 / (theta**scale)
        angles.append(torch.einsum("...n,d->...nd", ids[..., axis], omega))
    angles = torch.cat(angles, dim=-1)
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1).float()


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos_sin: torch.Tensor, out_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate adjacent channel pairs of q/k (B, L, H, D) in fp32, rounding once to out_dtype."""
    cos, sin = cos_sin[:, :, None].chunk(2, dim=-1)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        even = cos * pairs[..., 0] + (-sin) * pairs[..., 1]
        odd = sin * pairs[..., 0] + cos * pairs[..., 1]
        return torch.stack((even, odd), dim=-1).reshape_as(x).to(out_dtype)

    return rotate(q), rotate(k)


def timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    """Sinusoidal embedding of t in [0, 1] (scaled by 1000), fp32."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (1000.0 * t)[..., None].float() * freqs
    return torch.cat((torch.cos(args), torch.sin(args)), dim=-1)


class Flux3MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class Flux3RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """fp32 out: the engine's fused q/k norm + RoPE kernel rounds only after RoPE."""
        x_float = x.float()
        rrms = torch.rsqrt(torch.mean(x_float**2, dim=-1, keepdim=True) + 1e-6)
        return x_float * rrms * self.scale.float()


class Flux3ImageQKNorm(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.query_norm = Flux3RMSNorm(head_dim)
        self.key_norm = Flux3RMSNorm(head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query_norm(q), self.key_norm(k)


class Flux3ImageModulation(nn.Module):
    """Per-token shift / scale / gate from the timestep vector."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.lin = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.lin(F.silu(vec)).chunk(3, dim=-1)
        return shift, scale, gate


class Flux3ImageBlock(nn.Module):
    """Parallel attention + SwiGLU block with q/k RMSNorm and RoPE; a mode block and a joint block alike."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_in = nn.Linear(hidden_size, 2 * mlp_hidden_dim, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden_dim, hidden_size, bias=False)
        self.norm = Flux3ImageQKNorm(self.head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: torch.Tensor,
        mod: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        shift, scale, gate = mod
        # The engine's fused LayerNorm + modulate and SwiGLU compute in fp32 and round once.
        x_mod = (self.pre_norm(x.float()) * (1 + scale.float()) + shift.float()).to(x.dtype)
        batch, length, _ = x_mod.shape
        q, k, v = (
            proj(x_mod).view(batch, length, self.num_heads, self.head_dim)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        q, k = apply_rope(*self.norm(q, k), cos_sin, out_dtype=v.dtype)
        # Looked up per call so the deterministic patch applies.
        attn = flash_attn_module().flash_attn_func(q, k, v, softmax_scale=self.head_dim**-0.5)
        attn = (attn[0] if isinstance(attn, tuple) else attn).flatten(2)
        gate_half, value_half = self.mlp_in(x_mod).float().chunk(2, dim=-1)
        update = self.attn_out(attn) + self.mlp_out((F.silu(gate_half) * value_half).to(x_mod.dtype))
        return torch.addcmul(x, gate, update)


class Flux3ImageLastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        activated = self.adaLN_modulation[0](vec)
        shift_w, scale_w = self.adaLN_modulation[1].weight.chunk(2)
        x = self.norm_final(x) * (F.linear(activated, scale_w) + 1)
        return self.linear(x + F.linear(activated, shift_w))


class Flux3ImageTransformer(nn.Module):
    """Text-to-image FLUX.3 DiT: per-stream mode blocks, then joint single blocks over [image | text]."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        depth: int,
        depth_single_blocks: int,
        mlp_ratio: float,
        axes_dim: tuple[int, ...],
        theta: int,
        image_channels: int,
        context_in_dim: int,
    ):
        super().__init__()
        self.axes_dim = tuple(axes_dim)
        self.theta = theta
        self.gradient_checkpointing = False

        def block() -> Flux3ImageBlock:
            return Flux3ImageBlock(hidden_size, num_heads, mlp_ratio)

        def modulations() -> nn.ModuleDict:
            return nn.ModuleDict({stream: Flux3ImageModulation(hidden_size) for stream in (IMAGE, TEXT)})

        self.emb_in = nn.ModuleDict({IMAGE: nn.Linear(image_channels, hidden_size, bias=False)})
        self.txt_in = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.time_in = Flux3MLPEmbedder(256, hidden_size)
        self.early_stream_modulations = modulations()
        self.single_stream_modulations = modulations()
        self.content_mode_blocks = nn.ModuleDict({IMAGE: nn.ModuleList(block() for _ in range(depth))})
        self.txt_mode_blocks = nn.ModuleList(block() for _ in range(depth))
        self.single_blocks = nn.ModuleList(block() for _ in range(depth_single_blocks))
        self.final_layer = nn.ModuleDict({IMAGE: Flux3ImageLastLayer(hidden_size, image_channels)})

    def _run(self, block: nn.Module, *args) -> torch.Tensor:
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def _timestep_vector(self, timesteps: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Per-token timesteps (B, L) in [0, 1] -> (B, L, hidden); the embedding MLP runs outside autocast."""
        with torch.autocast(device_type=timesteps.device.type, enabled=False):
            vec = self.time_in(timestep_embedding(timesteps).to(self.time_in.in_layer.weight.dtype))
        return vec.to(dtype)

    def forward(
        self,
        image: torch.Tensor,
        sigma: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        flux3_txt_4d_rope_pos_ids: torch.Tensor,
        flux3_img_4d_rope_pos_ids: torch.Tensor,
    ) -> torch.Tensor:
        """image (B, L_img, C) at noise level sigma (B,), text (B, L_txt, context_in_dim) -> velocity (B, L_img, C)."""
        dtype = self.txt_in.weight.dtype
        num_image_tokens = image.shape[1]
        # The engine stores per-token timesteps in bf16, so the DiT sees sigma rounded to bf16; text runs at t=0.
        image_t = sigma.to(torch.bfloat16)[:, None].expand(-1, num_image_tokens)
        text_t = sigma.new_zeros(encoder_hidden_states.shape[:2], dtype=torch.bfloat16)
        vec = {IMAGE: self._timestep_vector(image_t, dtype), TEXT: self._timestep_vector(text_t, dtype)}
        early = {stream: self.early_stream_modulations[stream](vec[stream]) for stream in (IMAGE, TEXT)}
        single = {stream: self.single_stream_modulations[stream](vec[stream]) for stream in (IMAGE, TEXT)}
        rope = {
            IMAGE: rope_cos_sin(flux3_img_4d_rope_pos_ids, self.axes_dim, self.theta),
            TEXT: rope_cos_sin(flux3_txt_4d_rope_pos_ids, self.axes_dim, self.theta),
        }

        x_image = self.emb_in[IMAGE](image.to(dtype))
        x_text = self.txt_in(encoder_hidden_states.to(dtype))
        for image_block, text_block in zip(self.content_mode_blocks[IMAGE], self.txt_mode_blocks, strict=True):
            x_image = self._run(image_block, x_image, rope[IMAGE], early[IMAGE])
            x_text = self._run(text_block, x_text, rope[TEXT], early[TEXT])

        hidden = torch.cat((x_image, x_text), dim=1)
        joint_rope = torch.cat((rope[IMAGE], rope[TEXT]), dim=1)
        joint_mod = tuple(torch.cat((single[IMAGE][i], single[TEXT][i]), dim=1) for i in range(3))
        for block in self.single_blocks:
            hidden = self._run(block, hidden, joint_rope, joint_mod)
        return self.final_layer[IMAGE](hidden[:, :num_image_tokens], vec[IMAGE])
