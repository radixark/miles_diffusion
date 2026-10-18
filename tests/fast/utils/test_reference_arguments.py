"""Model roles are independent of the actor's full-parameter or LoRA training mode.

    actor: full / LoRA ---> ref: base / base + LoRA, teacher: base / base + LoRA
    actor: LoRA        ---> lora_base: reuse actor with its adapter disabled
    teacher            ---> read only by custom prepare/loss hooks, so it needs one
    actor LoRA enabled --> adapter config --> checkpoint A/B support + uniform IPC scaling + CLI r/alpha match
    adapter paths      ---> one plain PEFT directory per component, in order; any other count is rejected

Validation rejects incomplete role sources and unused options without changing args.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=["fsdp"])

from argparse import Namespace

import pytest
from peft import LoraConfig

from miles.utils.arguments import validate_actor_lora_adapter, validate_reference_model_args


def _args(**overrides):
    values = dict(
        ref_mode="none",
        ref_load=None,
        teacher_load=None,
        lora_adapter_paths=None,
        lora_ipc_weight_sync=False,
        colocate=True,
        train_only=False,
        lora_rank=2,
        lora_alpha=4,
        update_weight_target_modules=["transformer", "transformer_2"],
        ref_lora_adapter_paths=None,
        teacher_lora_adapter_paths=None,
        ref_cpu_offload=False,
        teacher_cpu_offload=False,
        use_lora=False,
        use_ema=False,
        sglang_tp_size=1,
        sglang_sp_degree=None,
        sglang_enable_cfg_parallel=False,
        loss_type="policy_loss",
        diffusion_kl_beta=0.0,
        custom_prepare_train_batch_path=None,
        custom_loss_function_path=None,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("role_has_adapter", [False, True])
def test_frozen_models_do_not_inherit_actor_training_mode(use_lora, role_has_adapter):
    args = _args(
        use_lora=use_lora,
        lora_adapter_paths=["actor-high", "actor-low"] if use_lora else None,
        ref_mode="ref",
        ref_load="reference-base",
        teacher_load="teacher-base",
        ref_lora_adapter_paths=["reference-high", "reference-low"] if role_has_adapter else None,
        teacher_lora_adapter_paths=["teacher-high", "teacher-low"] if role_has_adapter else None,
        ref_cpu_offload=True,
        teacher_cpu_offload=True,
        custom_loss_function_path="my_loss.teacher_loss",
    )
    original = vars(args).copy()
    validate_reference_model_args(args)
    assert vars(args) == original


def test_lora_base_needs_no_separate_reference_checkpoint():
    args = _args(use_lora=True, lora_adapter_paths=["actor-high", "actor-low"], ref_mode="lora_base")
    validate_reference_model_args(args)
    assert args.ref_load is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ref_mode": "ref"}, "--ref-mode ref requires --ref-load"),
        ({"ref_load": "reference"}, "--ref-load requires --ref-mode ref"),
        ({"ref_lora_adapter_paths": ["high", "low"]}, "--ref-lora-adapter-path requires --ref-load"),
        ({"teacher_lora_adapter_paths": ["high", "low"]}, "--teacher-lora-adapter-path requires --teacher-load"),
        # two components, one adapter --> the adapter/component pairing is ambiguous
        (
            {"ref_mode": "ref", "ref_load": "reference", "ref_lora_adapter_paths": ["only-one"]},
            "needs one adapter per",
        ),
        ({"ref_cpu_offload": True}, "--ref-cpu-offload requires --ref-load"),
        ({"teacher_cpu_offload": True}, "--teacher-cpu-offload requires --teacher-load"),
        ({"ref_mode": "lora_base"}, "--ref-mode lora_base requires --use-lora"),
        ({"teacher_load": "teacher"}, "--teacher-load has no consumer"),
    ],
)
def test_incomplete_model_roles_are_rejected(overrides, message):
    args = _args(**overrides)
    original = vars(args).copy()
    with pytest.raises(ValueError, match=message):
        validate_reference_model_args(args)
    assert vars(args) == original


def test_loaded_adapter_matching_actor_cli_passes_without_changing_args(tmp_path):
    # adapter r=2 alpha=4 == --lora-rank 2 --lora-alpha 4 -> accepted, args untouched
    adapter_paths = [str(tmp_path / "high"), str(tmp_path / "low")]
    for adapter_path in adapter_paths:
        LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]).save_pretrained(adapter_path)
    args = _args(use_lora=True, lora_adapter_paths=adapter_paths, lora_ipc_weight_sync=True)
    original = vars(args).copy()
    validate_actor_lora_adapter(args)
    assert vars(args) == original


@pytest.mark.parametrize("cli", [{"lora_rank": 64}, {"lora_alpha": 64}])
def test_loaded_adapter_rejects_mismatched_actor_cli(tmp_path, cli):
    # adapter r=2 alpha=4 vs CLI 64 -> rejected instead of silently ignoring the CLI values
    LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]).save_pretrained(tmp_path)
    args = _args(
        use_lora=True, lora_adapter_paths=[str(tmp_path)], update_weight_target_modules=["transformer"], **cli
    )
    with pytest.raises(ValueError, match="set --lora-rank 2 --lora-alpha 4"):
        validate_actor_lora_adapter(args)


def test_actor_adapter_count_must_match_the_components():
    # components [transformer, transformer_2], one adapter --> rejected before any config is read
    args = _args(use_lora=True, lora_adapter_paths=["unread-adapter-checkpoint"])
    with pytest.raises(ValueError, match="--lora-adapter-path needs one adapter per"):
        validate_actor_lora_adapter(args)


def test_actor_adapter_requires_lora_training_before_loading_config():
    args = _args(lora_adapter_paths=["unread-adapter-checkpoint"])
    with pytest.raises(ValueError, match="--lora-adapter-path requires --use-lora"):
        validate_actor_lora_adapter(args)


@pytest.mark.parametrize(
    ("config_options", "ipc", "colocate", "message"),
    [
        ({"use_dora": True}, False, True, "must contain only LoRA A/B matrices"),
        ({"use_rslora": True}, True, True, "requires uniform standard LoRA"),
        ({"use_rslora": True}, False, False, "requires uniform standard LoRA"),  # adapters over NCCL
    ],
)
def test_loaded_adapter_rejects_unsupported_export_state(tmp_path, config_options, ipc, colocate, message):
    LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], **config_options).save_pretrained(tmp_path)
    args = _args(
        use_lora=True,
        lora_adapter_paths=[str(tmp_path)],
        lora_ipc_weight_sync=ipc,
        colocate=colocate,
        update_weight_target_modules=["transformer"],
    )
    with pytest.raises(ValueError, match=message):
        validate_actor_lora_adapter(args)


def test_layer_specific_scaling_is_allowed_without_ipc(tmp_path):
    LoraConfig(
        r=2, lora_alpha=4, use_rslora=True, rank_pattern={"proj": 4}, alpha_pattern={"proj": 8}
    ).save_pretrained(tmp_path)
    validate_actor_lora_adapter(
        _args(use_lora=True, lora_adapter_paths=[str(tmp_path)], update_weight_target_modules=["transformer"])
    )
