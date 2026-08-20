from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

# The negative_prompt contract of the rollout payload:
#
#   --diffusion-negative-prompt <unset>  ─►  key absent from the payload
#                                            ─► engine strips nothing, per-model default applies
#   --diffusion-negative-prompt " "      ─►  " " passes through verbatim (FlowGRPO recipes)

from argparse import Namespace

from miles.rollout.sglang_diffusion_rollout import build_rollout_generate_payload, build_rollout_sampling_params


def _args(**overrides):
    values = dict(
        diffusion_negative_prompt=None,
        diffusion_generator_device=None,
        diffusion_eval_num_steps=None,
        diffusion_num_steps=10,
        diffusion_width=832,
        diffusion_height=480,
        diffusion_guidance_scale=4.0,
        diffusion_guidance_scale_2=None,
        diffusion_true_cfg_scale=None,
        diffusion_output_num_frames=None,
        diffusion_fps=None,
        diffusion_sde_type="sde",
        diffusion_noise_level=0.7,
        diffusion_log_prob_no_const=False,
        diffusion_debug_mode=False,
        train_pipeline_config_path=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_unset_negative_prompt_is_omitted():
    payload = build_rollout_generate_payload(build_rollout_sampling_params(_args()), "p")
    assert "negative_prompt" not in payload


def test_explicit_negative_prompt_passes_through():
    payload = build_rollout_generate_payload(build_rollout_sampling_params(_args(diffusion_negative_prompt=" ")), "p")
    assert payload["negative_prompt"] == " "
