"""MiniMax H3 Image train config: the forward reaches the output head of the pair's rollout step.

    rollout pair (step k, sigma_k) + sglang's packed t2i layout
        -> MiniMaxH3ImageTransformer2DModel(..., timestep=1 - sigma_k, pdd_step_index=k)
        -> negated: H3 predicts clean - noise, the flow objective noise - clean
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import shlex
import sys
from argparse import Namespace

import diffusers
import pytest
import torch
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence_ref2va_blocks,
)

from miles.backends.fsdp_utils.arguments import load_fsdp_args
from miles.backends.fsdp_utils.configs.h3_image import H3ImageTrainPipelineConfig
from miles.backends.fsdp_utils.mixed_precision import compile_param_dtype_maps
from miles.backends.fsdp_utils.models.diffusers import load_fsdp_parallel_plan
from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args, set_default_diffusion_args

TEXT_LEN, LATENT_H, LATENT_W = 5, 8, 12
NUM_IMAGE_ROWS = LATENT_H * LATENT_W // 4
SIGMA = 0.75


def _tiny_h3_image(forward_dtype=torch.float32):
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = diffusers.MiniMaxH3ImageTransformer2DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            hidden_size=24,
            num_layers=1,
            num_refiner_layers=1,
            ffn_dim=32,
            in_channels=24,
            patch_size=(1, 2, 2),
            text_dim=8,
            freq_dim=8,
            time_embed_hidden_dim=24,
            time_embed_dim=16,
            rope_freq_dim=2,
            num_pdd_heads=3,
        )
    if forward_dtype == torch.bfloat16:
        model.to(dtype=forward_dtype)
        # the released checkpoint is mixed: these modules stay fp32
        for name, layer in model.named_modules():
            if any(fragment in name for fragment in model._keep_in_fp32_modules):
                layer.float()
    return model


def _t2i_layout(ref_blocks=()):
    """The packed sequence the engine builds for one image: text rows, image rows, then a padding tail."""
    used_rows = TEXT_LEN + NUM_IMAGE_ROWS + sum(block["latent_h"] * block["latent_w"] // 4 for block in ref_blocks)
    return minimax_h3_packed_sequence_ref2va_blocks(
        text_len=TEXT_LEN,
        latent_t=1,
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        audio_t=0,
        ref_blocks=list(ref_blocks),
        seq_len=((used_rows + 15) // 16) * 16 + 16,
    )


def _rows_and_cond(layout):
    with torch.random.fork_rng():
        torch.manual_seed(11)
        visual = torch.randn(1, NUM_IMAGE_ROWS, 96)
        encoder_hidden_states = torch.randn(1, TEXT_LEN, 8)
    cond = {
        "encoder_hidden_states": encoder_hidden_states,
        "h3_packed_layout": layout,
        "h3_token_tags": layout["token_tags"],
    }
    return visual, cond


def _forward(model, visual, cond, denoising_step_indices):
    return H3ImageTrainPipelineConfig().compute_noise_pred(
        model=model,
        latents_input={"visual": visual},
        timesteps_input={"visual": torch.tensor([SIGMA * 1000.0])},
        denoising_step_indices=denoising_step_indices,
        pos_cond=cond,
        neg_cond=None,
        joint_cond=None,
        use_cfg=False,
        cfg_batching=False,
        guidance_scale=1.0,
        true_cfg_scale=None,
    )["visual"]


@pytest.mark.parametrize("forward_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_forward_runs_the_output_head_of_the_pairs_step(forward_dtype):
    """Every head accepts every input, so a wrong head trains without an error."""
    model = _tiny_h3_image(forward_dtype)
    layout = _t2i_layout()
    visual, cond = _rows_and_cond(layout)
    used_rows = int(layout["cu_seqlens"][1])

    predictions = [_forward(model, visual, cond, torch.tensor([step])) for step in range(3)]

    for step, prediction in enumerate(predictions):
        velocity = model(
            hidden_states=visual,
            encoder_hidden_states=cond["encoder_hidden_states"],
            timestep=torch.tensor([1.0 - SIGMA]),
            timestep_indices=torch.zeros(used_rows, dtype=torch.long),
            token_tags=layout["token_tags"][:used_rows],
            position_ids=layout["img_position_ids"][:used_rows].float(),
            image_indices=layout["img_pos"],
            text_indices=layout["text_pos"],
            pdd_step_index=step,
            return_dict=False,
        )[0]
        assert prediction.shape == (1, NUM_IMAGE_ROWS, 96)
        assert torch.equal(prediction, -velocity.float())
    assert not torch.equal(predictions[0], predictions[2])


def test_forward_without_the_rollout_step_is_rejected():
    """A loss that samples its own noise levels has no step to pick a head by."""
    visual, cond = _rows_and_cond(_t2i_layout())

    with pytest.raises(ValueError, match="denoising step"):
        _forward(_tiny_h3_image(), visual, cond, None)


def test_forward_rejects_a_layout_with_reference_rows():
    """An edit layout's image rows start with references, which are not target rows."""
    visual, cond = _rows_and_cond(_t2i_layout(ref_blocks=({"kind": "image", "latent_h": 4, "latent_w": 4},)))

    with pytest.raises(NotImplementedError, match="reference"):
        _forward(_tiny_h3_image(), visual, cond, torch.tensor([0]))


def test_a_forward_through_one_head_gives_every_head_a_gradient():
    """FSDP reduces a wrap's gradients as one buffer of the parameters that have one on a rank, in order: a head
    without a gradient on the ranks that ran another step would have its gradient averaged with that step's."""
    hooked, plain = _tiny_h3_image(), _tiny_h3_image()
    H3ImageTrainPipelineConfig().postprocess_model_after_materialize(hooked)
    visual, cond = _rows_and_cond(_t2i_layout())

    predictions = [_forward(model, visual, cond, torch.tensor([1])) for model in (hooked, plain)]
    for prediction in predictions:
        prediction.square().sum().backward()

    assert torch.equal(*predictions)
    plain_grads = {name: parameter.grad for name, parameter in plain.named_parameters()}
    for name, parameter in hooked.named_parameters():
        if name.startswith("pdd_proj_out.") and not name.startswith("pdd_proj_out.1."):
            assert plain_grads[name] is None
            assert parameter.grad is not None and not parameter.grad.any(), name
        else:
            assert torch.equal(parameter.grad, plain_grads[name]), name


def _grpo_args(**overrides):
    args = Namespace(
        loss_type="policy_loss",
        use_lora=True,
        lora_ipc_weight_sync=True,
        colocate=True,
        train_only=False,
        debug_rollout_only=False,
        diffusion_flow_shift=None,
        sequence_parallel_size=1,
        diffusion_num_steps=6,
        diffusion_eval_num_steps=None,
        rollout_microgroup_size=1,
        diffusion_output_num_frames=1,
        micro_batch_size=1,
        micro_batch_size_sample=None,
        micro_batch_size_tstep=None,
        diffusion_quality="exact",
        diffusion_true_cfg_scale=None,
    )
    vars(args).update(overrides)
    return args


def test_lora_flow_grpo_is_accepted():
    H3ImageTrainPipelineConfig.validate_args(_grpo_args())


@pytest.mark.parametrize(
    "overrides",
    [
        # no trainer: nothing is trained and nothing is synced
        {"debug_rollout_only": True, "use_lora": False, "lora_ipc_weight_sync": False},
        # a trainer without an engine: LoRA still, but no sync mode to hold it to
        {"train_only": True, "lora_ipc_weight_sync": False, "colocate": True},
        # engines on their own GPUs receive the adapters over NCCL
        {"colocate": False, "lora_ipc_weight_sync": False},
        # the recipe's explicit one-pair tiles
        {"micro_batch_size_sample": 1, "micro_batch_size_tstep": 1},
        {"diffusion_eval_num_steps": 6},
    ],
    ids=["rollout-only", "train-only", "adapters-over-nccl", "one-pair-tiles", "eval-steps-of-the-release"],
)
def test_supported_variations_are_accepted(overrides):
    H3ImageTrainPipelineConfig.validate_args(_grpo_args(**overrides))


def test_a_launch_with_the_required_flags_passes_the_real_parser(monkeypatch):
    """validate_args reads the parser's defaults, which a namespace built by hand does not have."""
    argv = (
        "test --hf-checkpoint MiniMaxAI/MiniMax-H3-Image --diffusion-task t2i --prompt-data prompts.jsonl "
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout --rm-type ocr "
        "--rollout-batch-size 8 --n-samples-per-prompt 16 --num-steps-per-rollout 2 --num-rollout 1 "
        "--actor-num-gpus-per-node 8 --rollout-num-gpus 8 --rollout-num-gpus-per-engine 2 --colocate "
        "--diffusion-num-steps 6 --diffusion-guidance-scale 1.0 --diffusion-sde-type cps "
        "--use-lora --lora-ipc-weight-sync"
    )
    monkeypatch.setattr(sys, "argv", shlex.split(argv))

    args = load_fsdp_args(extra_args_provider=get_miles_extra_args_provider())
    set_default_diffusion_args(args)
    miles_validate_args(args)

    assert args.diffusion_model_family == "h3_image"
    assert (args.micro_batch_size_sample, args.micro_batch_size_tstep) == (None, None)


def test_full_weight_training_is_rejected_even_without_an_engine():
    """--train-only syncs nothing, but every rank would still give a different per-step head its gradient."""
    with pytest.raises(ValueError, match="LoRA adapters only"):
        H3ImageTrainPipelineConfig.validate_args(_grpo_args(train_only=True, use_lora=False))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"use_lora": False}, "LoRA adapters only"),
        # merged base weights carry the per-step heads too
        ({"lora_ipc_weight_sync": False}, "as adapters"),
        ({"loss_type": "nft"}, "policy_loss"),
        ({"diffusion_flow_shift": 3.0}, "flow-shift"),
        ({"sequence_parallel_size": 2}, "sequence"),
        # the engine answers each of these with a 400 after both sides have loaded the DiT
        ({"diffusion_num_steps": 10}, "6-step grid"),
        ({"diffusion_eval_num_steps": 40}, "6-step grid"),
        ({"rollout_microgroup_size": 4}, "one image per request"),
        ({"diffusion_output_num_frames": 5}, "one image per request"),
        ({"diffusion_quality": "high"}, "exact or lossless"),
        ({"diffusion_true_cfg_scale": 4.0}, "unguided"),
        # the forward raises on these at the first train step
        ({"micro_batch_size": 2}, "one pair"),
        # the parser folds the tiles into micro_batch_size
        ({"micro_batch_size": 2, "micro_batch_size_sample": 2, "micro_batch_size_tstep": 1}, "one pair"),
    ],
    ids=[
        "full-weight",
        "lora-synced-as-merged-weights",
        "nft",
        "flow-shift",
        "sequence-parallel",
        "rollout-steps",
        "eval-steps",
        "microgroup",
        "frames",
        "quality-high",
        "true-cfg",
        "micro-batch",
        "micro-batch-tiles",
    ],
)
def test_unsupported_training_settings_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        H3ImageTrainPipelineConfig.validate_args(_grpo_args(**overrides))


def test_parallel_plan_keeps_the_fp32_modules_in_fp32():
    """The plan is H3's: a module H3 Image pins to fp32 under another name would be sharded as bf16."""
    patterns = load_fsdp_parallel_plan("h3_image").param_dtype_patterns
    model = _tiny_h3_image()
    fp32_names = {name for name, _ in model.named_parameters() if name.split(".")[0] in model._keep_in_fp32_modules}

    pinned = compile_param_dtype_maps(model, [], patterns, torch.bfloat16).root_map

    assert fp32_names
    assert set(pinned) == fp32_names and set(pinned.values()) == {torch.float32}
