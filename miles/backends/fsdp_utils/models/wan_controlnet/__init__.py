"""Frozen Wan backbone with a separately trained, pose-conditioned control branch."""

from .modeling import WanPoseControlNet, WanPoseControlTransformer

__all__ = ["WanPoseControlNet", "WanPoseControlTransformer"]
