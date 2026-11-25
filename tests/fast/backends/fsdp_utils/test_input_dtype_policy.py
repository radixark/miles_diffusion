"""The family's input_dtype_policy, resolved once per run, casts each stage's inputs as the trainer sends them.

    policy {"latents"/"timestep"/"cond": "default" | "fp32"/"bf16"/"fp16" | None}  --resolve(forward dtype)-->  InputDtypes
    PreparedBatch.model_inputs(InputDtypes)  -->  compute_noise_pred kwargs, floats cast, int masks and the raw batch untouched
    a batch derived by dataclasses.replace (DMD2's teacher) casts whatever it holds when it is read
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import dataclasses

import pytest
import torch

from miles.backends.fsdp_utils.configs.train_pipeline_config import TrainPipelineConfig
from miles.backends.fsdp_utils.input_dtype_policy import InputDtypes
from miles.backends.fsdp_utils.loss_hub.types import PreparedBatch, PreparedBatches

DEFAULT_POLICY = TrainPipelineConfig.input_dtype_policy
CAST_POLICY = {"latents": "default", "cond": None, "timestep": "fp32"}


def _prepared(**overrides):
    pos_cond = {
        "context": torch.zeros(1, 3, 8, dtype=torch.float32),
        "context_mask": torch.ones(1, 3, dtype=torch.int64),
    }
    timesteps = {"visual": torch.tensor([857.69], dtype=torch.float32)}
    fields = dict(
        latents={"visual": torch.zeros(1, 4, 8, dtype=torch.float32)},
        timesteps=timesteps,
        timesteps_for_model=timesteps,
        model=torch.nn.Identity(),
        component_name="transformer",
        guidance_scale=1.0,
        use_cfg=False,
        cfg_batching=False,
        true_cfg_scale=None,
        pos_cond=pos_cond,
        neg_cond=None,
        joint_cond=None,
        advantage=torch.zeros(1),
    )
    return PreparedBatch(**{**fields, **overrides})


def test_the_default_policy_is_passthrough():
    assert InputDtypes.resolve(DEFAULT_POLICY, torch.bfloat16) == InputDtypes(None, None, None)
    sent = _prepared().model_inputs(InputDtypes.resolve(DEFAULT_POLICY, torch.bfloat16))
    assert sent["latents_input"]["visual"].dtype == torch.float32
    assert sent["timesteps_input"]["visual"].dtype == torch.float32
    assert sent["pos_cond"]["context"].dtype == torch.float32
    assert sent["neg_cond"] is None and sent["joint_cond"] is None


def test_default_resolves_to_the_forward_dtype_and_names_to_their_dtype():
    assert InputDtypes.resolve({**DEFAULT_POLICY, "timestep": "default"}, torch.bfloat16).timestep == torch.bfloat16
    assert InputDtypes.resolve(CAST_POLICY, torch.bfloat16) == InputDtypes(torch.bfloat16, torch.float32, None)


def test_the_cast_touches_floats_only_and_leaves_the_batch_as_it_was():
    prepared = _prepared()
    sent = prepared.model_inputs(
        InputDtypes.resolve({"latents": "default", "cond": "default", "timestep": "fp32"}, torch.bfloat16)
    )
    assert sent["latents_input"]["visual"].dtype == torch.bfloat16
    assert sent["timesteps_input"]["visual"].dtype == torch.float32
    assert sent["pos_cond"]["context"].dtype == torch.bfloat16
    assert sent["pos_cond"]["context_mask"].dtype == torch.int64
    # The formulas read the batch: it keeps the rollout dtypes.
    assert prepared.latents["visual"].dtype == torch.float32 and prepared.pos_cond["context"].dtype == torch.float32


def test_a_batch_derived_by_replace_casts_what_it_holds_when_read():
    # DMD2 derives the teacher's batch from the scoring batch by swapping conds; the cast happens on send, so the
    # new conds are what gets cast.
    teacher = dataclasses.replace(
        _prepared(), use_cfg=True, neg_cond={"context": torch.zeros(1, 3, 8, dtype=torch.float32)}
    )
    sent = teacher.model_inputs(
        InputDtypes.resolve({"latents": None, "cond": "default", "timestep": None}, torch.bfloat16)
    )
    assert sent["neg_cond"]["context"].dtype == torch.bfloat16 and sent["pos_cond"]["context"].dtype == torch.bfloat16
    assert sent["latents_input"]["visual"].dtype == torch.float32


def test_every_stage_is_cast_the_same_way():
    stages = PreparedBatches(actor=_prepared(), teacher=_prepared(use_cfg=True), actor_forward=_prepared())
    input_dtypes = InputDtypes.resolve(CAST_POLICY, torch.bfloat16)
    for batch in (stages.actor, stages.teacher, stages.actor_forward):
        sent = batch.model_inputs(input_dtypes)
        assert sent["latents_input"]["visual"].dtype == torch.bfloat16
        assert sent["timesteps_input"]["visual"].dtype == torch.float32
        assert sent["pos_cond"]["context"].dtype == torch.float32


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        InputDtypes.resolve({"latnets": "default"}, torch.bfloat16)


def test_unknown_dtype_name_rejected():
    with pytest.raises(ValueError, match="unknown dtype"):
        InputDtypes.resolve({"latents": "fp8"}, torch.bfloat16)
