from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.backends.fsdp_utils.configs.qwen_image import QwenImageTrainPipelineConfig
from miles.backends.fsdp_utils.configs.train_pipeline_config import TrainPipelineConfig, resolve_diffusion_model_family
from miles.backends.fsdp_utils.configs.wan2_2 import Wan2_2TrainPipelineConfig


class TestFamilyResolution:
    # Checkpoint ref -> family key: declared patterns match case-insensitively
    # (HF ids and local paths alike); unknown refs fail loud; env var overrides.
    @pytest.mark.parametrize(
        "ref,family",
        [
            ("Qwen/Qwen-Image", "qwen_image"),
            ("Wan-AI/Wan2.2-T2V-A14B", "wan2_2"),
            ("/data/ckpts/SD3.5-Medium-Finetune", "sd3"),
        ],
    )
    def test_known_patterns(self, ref, family):
        assert resolve_diffusion_model_family(ref) == family

    def test_unknown_ref_raises(self):
        with pytest.raises(ValueError, match="Cannot resolve"):
            resolve_diffusion_model_family("mystery-lab/unknown-model")

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("MILES_DIFFUSION_MODEL_FAMILY", "SD3")
        assert resolve_diffusion_model_family("mystery-lab/unknown-model") == "sd3"


class _MinimalConfig(TrainPipelineConfig):
    def prepare_cond_kwargs(self, cond, device):
        return {}

    def cfg_combine(self, noise_pred_pos, noise_pred_neg, guidance_scale, true_cfg_scale=None):
        scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        return noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)

    def preprocess_model_before_fsdp(self, model):
        return None


class _CondBiasDiT(torch.nn.Module):
    """Linear fake DiT: output = hidden*2 + bias, so every path is exactly checkable."""

    def forward(self, hidden_states, timestep, return_dict=False, bias=None):
        return (hidden_states * 2.0 + (bias if bias is not None else 0.0),)


class TestComputeNoisePred:
    # The forward hoisted from the actor: no-CFG = one pos pass; CFG joint-batch
    # (cat->chunk) must be numerically identical to the two-pass path.
    def setup_method(self):
        self.cfg = _MinimalConfig()
        self.h = torch.arange(12.0).reshape(2, 6)
        self.pos = {"bias": torch.full((2, 1), 1.0)}
        self.neg = {"bias": torch.full((2, 1), -1.0)}

    def _call(self, **overrides):
        kwargs = dict(
            model=_CondBiasDiT(),
            latents_input=self.h,
            timesteps_input=torch.tensor([3.0, 5.0]),
            pos_cond=self.pos,
            neg_cond=self.neg,
            joint_cond=None,
            use_cfg=True,
            cfg_batching=False,
            guidance_scale=2.0,
            true_cfg_scale=None,
        )
        kwargs.update(overrides)
        return self.cfg.compute_noise_pred(**kwargs)

    def test_no_cfg_is_single_pos_pass(self):
        torch.testing.assert_close(self._call(use_cfg=False), self.h * 2.0 + 1.0)

    def test_two_pass_applies_cfg_combine(self):
        # neg + s*(pos - neg) with pos = 2h+1, neg = 2h-1, s = 2 -> 2h+3
        torch.testing.assert_close(self._call(), self.h * 2.0 + 3.0)

    def test_joint_batch_matches_two_pass(self):
        joint = {"bias": torch.cat([self.pos["bias"], self.neg["bias"]], dim=0)}
        torch.testing.assert_close(self._call(cfg_batching=True, joint_cond=joint), self._call())


def test_timestep_scaling_is_qwen_image_specific():
    assert QwenImageTrainPipelineConfig.needs_timestep_scaling
    assert not Wan2_2TrainPipelineConfig.needs_timestep_scaling
    assert not _MinimalConfig.needs_timestep_scaling
