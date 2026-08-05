from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.backends.fsdp_utils.configs.train_pipeline_config import TrainPipelineConfig
from miles.backends.fsdp_utils.precision import apply_input_dtype_policy

DEFAULT_POLICY = TrainPipelineConfig.input_dtype_policy


def _inputs():
    latents = torch.zeros(1, 4, 8, dtype=torch.float32)
    timesteps = torch.tensor([857.69], dtype=torch.float32)
    pos_cond = {
        "context": torch.zeros(1, 3, 8, dtype=torch.float32),
        "context_mask": torch.ones(1, 3, dtype=torch.int64),
    }
    return latents, timesteps, pos_cond


def test_default_policy_is_passthrough():
    latents, timesteps, pos_cond = _inputs()
    out_latents, out_timesteps, (out_pos, out_neg, out_joint) = apply_input_dtype_policy(
        DEFAULT_POLICY,
        latents=latents,
        timesteps=timesteps,
        conds=(pos_cond, None, None),
        default_dtype=torch.bfloat16,
    )
    assert out_latents.dtype == torch.float32
    assert out_timesteps.dtype == torch.float32
    assert out_pos["context"].dtype == torch.float32
    assert out_pos["context_mask"].dtype == torch.int64
    assert out_neg is None and out_joint is None


def test_family_override_timestep_default():
    latents, timesteps, pos_cond = _inputs()
    policy = {**DEFAULT_POLICY, "timestep": "default"}
    _, out_timesteps, _ = apply_input_dtype_policy(
        policy,
        latents=latents,
        timesteps=timesteps,
        conds=(pos_cond, None, None),
        default_dtype=torch.bfloat16,
    )
    assert out_timesteps.dtype == torch.bfloat16


def test_cast_policy_casts_floats_only():
    latents, timesteps, pos_cond = _inputs()
    out_latents, _, (out_pos, _, _) = apply_input_dtype_policy(
        {"latents": "default", "cond": "default", "timestep": "fp32"},
        latents=latents,
        timesteps=timesteps,
        conds=(pos_cond, None, None),
        default_dtype=torch.bfloat16,
    )
    assert out_latents.dtype == torch.bfloat16
    assert out_pos["context"].dtype == torch.bfloat16
    assert out_pos["context_mask"].dtype == torch.int64


def test_unknown_key_rejected():
    latents, timesteps, pos_cond = _inputs()
    with pytest.raises(ValueError, match="unknown keys"):
        apply_input_dtype_policy(
            {"latnets": "default"},
            latents=latents,
            timesteps=timesteps,
            conds=(pos_cond, None, None),
            default_dtype=torch.bfloat16,
        )
