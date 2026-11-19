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
        diffusion_quality="exact",
        diffusion_eval_num_steps=None,
        diffusion_num_steps=10,
        diffusion_width=832,
        diffusion_height=480,
        diffusion_guidance_scale=4.0,
        diffusion_guidance_scale_2=None,
        diffusion_true_cfg_scale=None,
        diffusion_output_num_frames=None,
        diffusion_fps=None,
        primary_stream="visual",
        diffusion_sde_type={"visual": "sde"},
        diffusion_noise_level={"visual": 0.7},
        diffusion_log_prob_no_const={"visual": False},
        diffusion_debug_mode=False,
        rollout_video_dtype="keep",
        rollout_extra_streams=[],
        rollout_extra_stream_sde_args={},
        train_pipeline_config_path=None,
        use_output_store=False,
        rm_type="pickscore",
        custom_rm_path=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_extra_streams_are_requested_only_for_training_rollouts_that_name_them():
    """The engine rejects the field on an eval request, and an unasked-for key would change every existing request."""
    assert "rollout_extra_streams" not in build_rollout_sampling_params(_args())
    asked = _args(rollout_extra_streams=["audio"])
    assert build_rollout_sampling_params(asked)["rollout_extra_streams"] == ["audio"]
    assert "rollout_extra_streams" not in build_rollout_sampling_params(asked, evaluation=True)


def test_the_primary_streams_settings_are_the_top_level_and_others_are_overrides():
    """The engine reads the primary stream's settings from the top level; a dict there, or an override sent for a
    stream that inherits them, breaks the request or every multi-stream run on an engine without overrides."""
    overrides = {"audio": {"rollout_sde_type": "ode", "rollout_noise_level": 0.7, "rollout_log_prob_no_const": True}}
    args = _args(
        diffusion_sde_type={"visual": "cps", "audio": "ode"},
        diffusion_log_prob_no_const={"visual": True, "audio": True},
        rollout_extra_streams=["audio"],
        rollout_extra_stream_sde_args=overrides,
    )

    params = build_rollout_sampling_params(args)

    assert (params["rollout_sde_type"], params["rollout_noise_level"], params["rollout_log_prob_no_const"]) == (
        "cps",
        0.7,
        True,
    )
    assert params["rollout_extra_stream_sde_args"] == overrides
    assert "rollout_extra_stream_sde_args" not in build_rollout_sampling_params(args, evaluation=True)
    assert "rollout_extra_stream_sde_args" not in build_rollout_sampling_params(_args(rollout_extra_streams=["audio"]))


def test_unset_negative_prompt_is_omitted():
    payload = build_rollout_generate_payload(build_rollout_sampling_params(_args()), "p")
    assert "negative_prompt" not in payload


def test_explicit_negative_prompt_passes_through():
    payload = build_rollout_generate_payload(build_rollout_sampling_params(_args(diffusion_negative_prompt=" ")), "p")
    assert payload["negative_prompt"] == " "


def test_output_store_asks_the_engine_to_store_the_outputs():
    assert "rollout_return_outputs_via_store" not in build_rollout_sampling_params(_args())
    assert build_rollout_sampling_params(_args(use_output_store=True))["rollout_return_outputs_via_store"] is True
    assert "rollout_return_outputs_via_store" not in build_rollout_sampling_params(
        _args(use_output_store=True), evaluation=True
    )
