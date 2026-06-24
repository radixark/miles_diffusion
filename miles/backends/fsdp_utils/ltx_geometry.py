"""Deterministic LTX-2.3 video geometry for miles train forward.

sglang rollout returns text embeds via ``denoising_env`` only; RoPE coords and
TI2V masks are rebuilt here from the same request-level constants sglang uses
(see ``MILES_ROLLOUT_HANDOFF.md``).
"""

from __future__ import annotations

import torch

# LTX-2.3 defaults (match sglang ``LTXVideoRotaryPositionalEmbeddings`` / VAE).
_LTX_VAE_SPATIAL_COMPRESSION = 32
_LTX_VAE_TEMPORAL_COMPRESSION = 8
_LTX_PATCH_SIZE = 1
_LTX_PATCH_SIZE_T = 1
_LTX_SCALE_FACTORS = (8, 32, 32)
_LTX_CAUSAL_OFFSET = 1


def latent_grid_shape(
    *,
    height: int,
    width: int,
    num_frames: int,
) -> tuple[int, int, int]:
    """Return ``(latent_frames, latent_height, latent_width)``."""
    latent_height = height // _LTX_VAE_SPATIAL_COMPRESSION
    latent_width = width // _LTX_VAE_SPATIAL_COMPRESSION
    latent_frames = (num_frames - 1) // _LTX_VAE_TEMPORAL_COMPRESSION + 1
    return latent_frames, latent_height, latent_width


def prepare_ltx_video_positions(
    *,
    batch_size: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    fps: float,
    device: torch.device,
    start_frame: int = 0,
) -> torch.Tensor:
    """Build video position grid ``[B, 3, T, 2]`` for ltx_core (middle-index RoPE)."""
    grid_f = torch.arange(
        start=int(start_frame),
        end=int(num_latent_frames) + int(start_frame),
        step=_LTX_PATCH_SIZE_T,
        dtype=torch.float32,
        device=device,
    )
    grid_h = torch.arange(
        start=0,
        end=latent_height,
        step=_LTX_PATCH_SIZE,
        dtype=torch.float32,
        device=device,
    )
    grid_w = torch.arange(
        start=0,
        end=latent_width,
        step=_LTX_PATCH_SIZE,
        dtype=torch.float32,
        device=device,
    )
    grid = torch.stack(torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij"), dim=0)

    patch_size = (_LTX_PATCH_SIZE_T, _LTX_PATCH_SIZE, _LTX_PATCH_SIZE)
    patch_ends = grid + torch.tensor(patch_size, dtype=grid.dtype, device=grid.device).view(3, 1, 1, 1)
    latent_coords = torch.stack([grid, patch_ends], dim=-1)
    latent_coords = latent_coords.flatten(1, 3).unsqueeze(0).expand(batch_size, -1, -1, -1)

    scale_tensor = torch.tensor(_LTX_SCALE_FACTORS, device=latent_coords.device)
    broadcast_shape = [1] * latent_coords.ndim
    broadcast_shape[1] = -1
    pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)
    pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + _LTX_CAUSAL_OFFSET - _LTX_SCALE_FACTORS[0]).clamp(min=0)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / float(fps)
    return pixel_coords


def build_ltx_t2v_geometry(
    *,
    batch_size: int,
    num_tokens: int,
    latent_dim: int,
    height: int,
    width: int,
    num_frames: int,
    fps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Pure text-to-video geometry: all tokens denoise, clean latent is zero."""
    latent_frames, latent_height, latent_width = latent_grid_shape(height=height, width=width, num_frames=num_frames)
    expected_tokens = latent_frames * latent_height * latent_width
    if expected_tokens != num_tokens:
        raise ValueError(
            f"LTX latent token count mismatch: trajectory has T={num_tokens} but "
            f"geometry from {height}x{width}x{num_frames}@{fps}fps expects "
            f"T={expected_tokens} ({latent_frames=}, {latent_height=}, {latent_width=})"
        )

    positions = prepare_ltx_video_positions(
        batch_size=batch_size,
        num_latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        fps=fps,
        device=device,
    )
    denoise_mask = torch.ones(batch_size, num_tokens, device=device, dtype=torch.float32)
    clean_latent = torch.zeros(batch_size, num_tokens, latent_dim, device=device, dtype=dtype)
    return {
        "positions": positions.to(dtype),
        "denoise_mask": denoise_mask,
        "clean_latent": clean_latent,
    }
