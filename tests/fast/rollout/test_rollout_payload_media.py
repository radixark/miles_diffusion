"""Which media the engine decodes and ships: training rollouts skip the decode when no reward reads it."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.rollout.sglang_diffusion_rollout import build_rollout_sampling_params


def _args(**overrides):
    values = dict(
        diffusion_negative_prompt=None,
        diffusion_generator_device=None,
        diffusion_quality="exact",
        diffusion_eval_num_steps=None,
        diffusion_num_steps=4,
        diffusion_width=832,
        diffusion_height=480,
        diffusion_guidance_scale=1.0,
        diffusion_guidance_scale_2=None,
        diffusion_true_cfg_scale=None,
        diffusion_output_num_frames=None,
        diffusion_fps=None,
        diffusion_sde_type="ode",
        diffusion_noise_level=0.7,
        diffusion_log_prob_no_const=False,
        diffusion_debug_mode=False,
        rollout_video_dtype="uint8",
        train_pipeline_config_path=None,
        rm_type="none",
        custom_rm_path=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_training_rollouts_skip_the_decode_when_no_reward_reads_it():
    assert build_rollout_sampling_params(_args())["rollout_return_generated_output"] is False
    # Eval renders are what gets looked at, so they always decode.
    assert "rollout_return_generated_output" not in build_rollout_sampling_params(_args(), evaluation=True)


@pytest.mark.parametrize("overrides", [{"rm_type": "pickscore"}, {"custom_rm_path": "my.rm"}])
def test_a_reward_keeps_the_decode(overrides):
    assert "rollout_return_generated_output" not in build_rollout_sampling_params(_args(**overrides))


def test_video_dtype_applies_to_eval_too():
    for evaluation in (False, True):
        assert build_rollout_sampling_params(_args(), evaluation=evaluation)["rollout_video_dtype"] == "uint8"
