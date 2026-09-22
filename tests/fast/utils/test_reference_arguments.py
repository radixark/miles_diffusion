"""Pretrained actor LoRA must match checkpoint and weight-publication support.

    --use-lora --> saved adapter config --> A/B-only checkpoint + uniform IPC scaling + CLI r/alpha match
    unsupported config -----------------> reject before model construction

Validation requires the CLI rank/alpha to match the loaded adapter and leaves args unchanged.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=["fsdp"])

from argparse import Namespace

import pytest
from peft import LoraConfig

from miles.utils.arguments import validate_actor_lora_adapter


def _args(**overrides):
    values = dict(
        ref_mode="none",
        ref_load=None,
        teacher_load=None,
        lora_adapter_path=None,
        lora_ipc_weight_sync=False,
        lora_rank=2,
        lora_alpha=4,
        update_weight_target_modules=["transformer", "transformer_2"],
        ref_lora_adapter_path=None,
        teacher_lora_adapter_path=None,
        ref_cpu_offload=False,
        teacher_cpu_offload=False,
        use_lora=False,
        use_ema=False,
        sglang_tp_size=1,
        sglang_sp_degree=None,
        sglang_enable_cfg_parallel=False,
        loss_type="policy_loss",
        diffusion_kl_beta=0.0,
    )
    values.update(overrides)
    return Namespace(**values)


def test_loaded_adapter_matching_actor_cli_passes_without_changing_args(tmp_path):
    # adapter r=2 alpha=4 == --lora-rank 2 --lora-alpha 4 -> accepted, args untouched
    args = _args(use_lora=True, lora_adapter_path=str(tmp_path), lora_ipc_weight_sync=True)
    for component in args.update_weight_target_modules:
        LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]).save_pretrained(tmp_path / component)
    original = vars(args).copy()
    validate_actor_lora_adapter(args)
    assert vars(args) == original


@pytest.mark.parametrize("cli", [{"lora_rank": 64}, {"lora_alpha": 64}])
def test_loaded_adapter_rejects_mismatched_actor_cli(tmp_path, cli):
    # adapter r=2 alpha=4 vs CLI 64 -> rejected instead of silently ignoring the CLI values
    LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]).save_pretrained(tmp_path)
    args = _args(use_lora=True, lora_adapter_path=str(tmp_path), update_weight_target_modules=["transformer"], **cli)
    with pytest.raises(ValueError, match="set --lora-rank 2 --lora-alpha 4"):
        validate_actor_lora_adapter(args)


def test_actor_adapter_requires_lora_training_before_loading_config():
    args = _args(lora_adapter_path="unread-adapter-checkpoint")
    with pytest.raises(ValueError, match="--lora-adapter-path requires --use-lora"):
        validate_actor_lora_adapter(args)


@pytest.mark.parametrize(
    ("config_options", "ipc", "message"),
    [
        ({"use_dora": True}, False, "must contain only LoRA A/B matrices"),
        ({"use_rslora": True}, True, "requires uniform standard LoRA"),
    ],
)
def test_loaded_adapter_rejects_unsupported_export_state(tmp_path, config_options, ipc, message):
    LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], **config_options).save_pretrained(tmp_path)
    args = _args(
        use_lora=True,
        lora_adapter_path=str(tmp_path),
        lora_ipc_weight_sync=ipc,
        update_weight_target_modules=["transformer"],
    )
    with pytest.raises(ValueError, match=message):
        validate_actor_lora_adapter(args)


def test_layer_specific_scaling_is_allowed_without_ipc(tmp_path):
    LoraConfig(
        r=2, lora_alpha=4, use_rslora=True, rank_pattern={"proj": 4}, alpha_pattern={"proj": 8}
    ).save_pretrained(tmp_path)
    validate_actor_lora_adapter(
        _args(use_lora=True, lora_adapter_path=str(tmp_path), update_weight_target_modules=["transformer"])
    )
