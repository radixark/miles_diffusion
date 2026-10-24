"""--loss-type dmd2 defaults and validation; validation rejects what DMD2 cannot run and never fills args.

    defaults      ──► converter / generator prepare / generator loss / train-input paths,
                      --critic-load = --teacher-load, --critic-lr = --lr, unless the user set them
    critic group  ──► scores with --teacher-load (no adapters), learns on a visual --fsdp-flow-shift grid,
                      takes --dmd2-critic-steps-per-rollout steps over each DP shard
    other losses  ──► --critic-load / --critic-lr have no consumer and are rejected
    teacher score ──► CFG --dmd2-teacher-guidance-scale >= 1
    actor         ──► unguided rollouts, one step per rollout
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.utils.arguments import set_default_diffusion_args, validate_dmd2_args


def _args(**overrides):
    values = dict(
        loss_type="dmd2",
        critic_load="teacher",
        critic_lr=1e-6,
        teacher_load="teacher",
        teacher_lora_adapter_paths=None,
        dmd2_teacher_guidance_scale=4.5,
        dmd2_critic_steps_per_rollout=5,
        fsdp_flow_shift={"visual": 3.0},
        diffusion_guidance_scale=1.0,
        diffusion_true_cfg_scale=None,
        use_lora=False,
        ref_mode="none",
        num_steps_per_rollout=1,
        global_batch_size=40,
        actor_num_gpus_per_node=4,
        actor_num_nodes=1,
        sequence_parallel_size=2,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("user", "expected"),
    [({}, ("teacher", 2e-6)), ({"critic_load": "critic-init", "critic_lr": 8e-6}, ("critic-init", 8e-6))],
)
def test_dmd2_critic_defaults_to_the_teacher_and_the_actor_lr(user, expected):
    args = Namespace(
        sglang_tp_size=1,
        sglang_sp_degree=None,
        sglang_enable_cfg_parallel=False,
        rollout_num_gpus_per_engine=1,
        loss_type="dmd2",
        custom_expand_samples_to_train_pairs_path=None,
        custom_prepare_train_batch_path=None,
        custom_loss_function_path=None,
        custom_train_input_function_path=None,
        ref_mode=None,
        diffusion_kl_beta=0.0,
        teacher_load="teacher",
        lr=2e-6,
        **{"critic_load": None, "critic_lr": None, **user},
    )
    set_default_diffusion_args(args)
    assert (args.critic_load, args.critic_lr) == expected


@pytest.mark.parametrize("flag", ["critic_load", "critic_lr"])
def test_critic_flags_without_dmd2_are_rejected(flag):
    with pytest.raises(ValueError, match="configure the critic group of --loss-type dmd2"):
        validate_dmd2_args(Namespace(**{"loss_type": "nft", "critic_load": None, "critic_lr": None, flag: "set"}))
    validate_dmd2_args(Namespace(loss_type="nft", critic_load=None, critic_lr=None))


@pytest.mark.parametrize("custom_loss", [None, "my.generator_loss"])
def test_dmd2_fills_only_unset_hook_paths(custom_loss):
    args = Namespace(
        sglang_tp_size=1,
        sglang_sp_degree=None,
        sglang_enable_cfg_parallel=False,
        rollout_num_gpus_per_engine=1,
        loss_type="dmd2",
        custom_expand_samples_to_train_pairs_path=None,
        custom_prepare_train_batch_path=None,
        custom_loss_function_path=custom_loss,
        custom_train_input_function_path=None,
        ref_mode=None,
        diffusion_kl_beta=0.0,
        teacher_load="teacher",
        lr=2e-6,
        critic_load=None,
        critic_lr=None,
    )
    set_default_diffusion_args(args)
    assert args.custom_expand_samples_to_train_pairs_path.endswith(
        "data_conversion_hub.dmd2.expand_samples_to_train_pairs"
    )
    assert args.custom_prepare_train_batch_path.endswith("loss_hub.dmd2.prepare_generator_batch")
    assert args.custom_loss_function_path == (
        custom_loss or "miles.backends.fsdp_utils.loss_hub.dmd2.dmd2_loss_formula"
    )
    assert args.custom_train_input_function_path.endswith("loss_hub.dmd2.select_scored_pairs")
    assert args.ref_mode == "none"


def test_valid_dmd2_args_pass_unchanged():
    args = _args()
    original = vars(args).copy()
    validate_dmd2_args(args)
    assert vars(args) == original


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"teacher_load": None}, "requires --teacher-load"),
        ({"dmd2_teacher_guidance_scale": None}, "--dmd2-teacher-guidance-scale >= 1"),
        ({"dmd2_teacher_guidance_scale": 0.5}, "--dmd2-teacher-guidance-scale >= 1"),
        ({"fsdp_flow_shift": None}, "visual --fsdp-flow-shift"),
        ({"fsdp_flow_shift": {"audio": 3.0}}, "visual --fsdp-flow-shift"),
        ({"diffusion_guidance_scale": 4.5}, "samples the student unguided"),
        ({"diffusion_true_cfg_scale": 4.5}, "samples the student unguided"),
        ({"use_lora": True}, "drop --use-lora"),
        ({"teacher_lora_adapter_paths": ["teacher-adapter"]}, "drop --use-lora and --teacher-lora-adapter-path"),
        ({"ref_mode": "ref"}, "uses no reference model"),
        ({"num_steps_per_rollout": 2}, "one actor step per rollout"),
        ({"dmd2_critic_steps_per_rollout": 0}, "must be >= 1"),
        # 40 samples over 2 DP shards = 20 per shard, which 3 critic steps cannot split.
        ({"dmd2_critic_steps_per_rollout": 3}, "split each of the 2 DP shards"),
    ],
)
def test_unsupported_dmd2_args_are_rejected(overrides, message):
    args = _args(**overrides)
    original = vars(args).copy()
    with pytest.raises(ValueError, match=message):
        validate_dmd2_args(args)
    assert vars(args) == original
