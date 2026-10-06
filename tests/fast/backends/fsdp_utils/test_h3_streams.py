"""H3 packed forward on the SFT training path, without pretrained weights or encoders.

One H3TrainPipelineConfig trains both tasks through one compute_noise_pred; a ref2va pair adds an audio target and
references.

    visual-only t2va input (an RL trajectory) {"visual": [1, rows, 96]}, timesteps {"visual": [sigma * 1000]}
         │ scatter into the packed video buffer at update_mask; audio rows stay zero at the video time
         ▼
    H3 DiT(hidden_states, audio_hidden_states, timestep = 1 - sigma, ...) over packed rows [0, cu_seqlens[1]);
         the padding tail the engine isolates as its own document never reaches it
         ▼
    {"visual": -pred[:, update_mask]}   same [1, rows, 96] as the SFT target ──► sft_loss_formula

    t2va SFT pair: every row of both streams is a target row
    latents_input {"visual": [1, rows_v, 96], "audio": [1, rows_a, 32]}, each stream at its own sigma
         ▼
    H3 DiT timestep = (visual t, audio t); text rows follow visual t
         ▼
    {"visual": -pred[:, update_mask], "audio": -audio_pred}; a silent target returns only "visual"

    ref2va pair: target rows of both streams; fixed reference rows fill ~update_mask
    latents_input {"visual": [1, rows_v, 96], "audio": [1, rows_a, 32]}
    timesteps_input {"visual": [sigma_v * 1000], "audio": [sigma_a * 1000]}
         │ scatter into each modality's packed buffer
         ▼
    H3 DiT over packed rows [0, cu_seqlens[1]); the padding tail never reaches it
         timestep = (visual t, audio t, visual reference t, audio reference t), t = 1 - sigma;
         text rows follow visual t
         ▼
    {"visual": -pred[:, update_mask], "audio": -pred[:, audio_update_mask]}   (diffusers flow direction)
    a silent target (h3_target_has_soundtrack=False) still feeds its noised-silence audio rows, but returns
    only "visual", so the SFT loss supervises the video alone

What each test pins:
  t2va prediction     keeps the batch dim, so the SFT loss accepts it; the DiT sees no padding rows
  t2va joint pair     audio rows reach the DiT unchanged at the audio time; silence returns only "visual"
  ref2va scatter      target and reference rows land in each stream's buffer, per-row timestep routing,
                      velocity sign, padding trim; for diffusers tuple and Transformer2DModelOutput returns
  audio time          moving the audio timestep leaves every other row's time alone
  silent target       audio rows still reach the DiT at the audio time; only the visual prediction returns
  reference times     taken from conditioning, and clamped to max(target t, reference t) as native H3 does
  ref2va validation   SFT only (rollout serves t2va); transformer_ref, micro-batch-size 1 and an audio
                      --fsdp-flow-shift (every H3 SFT) are required
  tiny diffusers H3   real forward/backward through prepare + loss, fp32 and bf16
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest
import torch
from torch import nn

from miles.backends.fsdp_utils.configs.h3 import H3TrainPipelineConfig
from miles.backends.fsdp_utils.loss_hub.sft import prepare_sft_batch, sft_loss_formula
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext
from miles.backends.fsdp_utils.metrics import new_metric_buffer


class _VisualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, **kwargs):
        self.inputs = kwargs
        return Namespace(sample=kwargs["hidden_states"] * self.scale)


def test_t2va_prediction_matches_the_sft_target_and_skips_padding_rows():
    # Packed rows: [text 0-1 | video 2-4 | audio 5-6 | pad 7]; every t2va video row is a target row.
    layout = {
        "seq_len": 8,
        "img_pos": torch.tensor([2, 3, 4]),
        "audio_pos": torch.tensor([5, 6]),
        "text_pos": torch.tensor([0, 1]),
        "update_mask": torch.tensor([True, True, True]),
        "img_position_ids": torch.arange(24.0).view(8, 3),
        "cu_seqlens": torch.tensor([0, 7, 8]),
    }
    cond = {
        "encoder_hidden_states": torch.ones(1, 2, 8),
        "h3_packed_layout": layout,
        "h3_token_tags": torch.tensor([1, 1, 0, 0, 0, 2, 2, -1]),
    }
    config = H3TrainPipelineConfig()
    context = DiffusionLossContext(
        models={"transformer": _VisualHead()},
        train_pipeline_config=config,
        sde_backend=None,
        args=Namespace(
            seed=42,
            fsdp_flow_shift={"visual": 3.0},
            log_loss_sigma_bucket=5,
            diffusion_supervised_streams=["visual", "audio"],
        ),
        forward_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    batch = [{"latent": {"visual": torch.ones(3, 96)}, "cond_kwargs": cond}]
    prepared = prepare_sft_batch(context, batch)
    prediction = config.compute_noise_pred(
        model=context.models["transformer"],
        latents_input=prepared.latents,
        timesteps_input=prepared.timesteps_for_model,
        pos_cond=prepared.pos_cond,
        neg_cond=None,
        joint_cond=None,
        use_cfg=False,
        cfg_batching=False,
        guidance_scale=0.0,
        true_cfg_scale=None,
    )
    assert prediction["visual"].shape == prepared.extras["target"]["visual"].shape == (1, 3, 96)
    # cu_seqlens = [0, 7, 8]: the DiT sees rows 0-6 only; pad row 7 (tag -1) is dropped, not clamped.
    inputs = context.models["transformer"].inputs
    assert inputs["token_tags"].tolist() == [1, 1, 0, 0, 0, 2, 2]
    assert torch.equal(inputs["position_ids"], layout["img_position_ids"][:7])
    assert inputs["timestep_indices"].shape == (7,)
    metrics = new_metric_buffer(None, torch.device("cpu"), ["transformer"], sigma_buckets=5)
    loss = sft_loss_formula(
        context, batch, prepared, new_pred=prediction, old_pred=None, ref_pred=None, metrics=metrics
    )
    assert torch.isfinite(loss) and loss.item() > 0


def _timestep(sigma):
    return torch.tensor([sigma * 1000.0])


def _case():
    # Modality rows are deliberately interleaved in the packed buffer. The
    # modality-local reference rows precede each stream's generated rows.
    layout = {
        "img_pos": torch.tensor([3, 6, 9, 10]),
        "audio_pos": torch.tensor([2, 4, 5, 7, 8]),
        "text_pos": torch.tensor([0, 1]),
        "update_mask": torch.tensor([False, False, True, True]),
        "audio_update_mask": torch.tensor([False, False, True, True, True]),
        "img_position_ids": torch.arange(36).view(12, 3).float(),
        # Row 11 is the padding tail the engine attends to as a separate document.
        "cu_seqlens": torch.tensor([0, 11, 12]),
    }
    cond = {
        "encoder_hidden_states": torch.ones(1, 2, 8),
        "h3_packed_layout": layout,
        "h3_token_tags": torch.tensor([1, 1, 2, 0, 2, 2, 0, 2, 2, 0, 0, -1]),
        "h3_reference_visual": torch.full((2, 96), 11.0),
        "h3_reference_audio": torch.full((2, 32), 13.0),
        "h3_reference_visual_timestep": 0.999,
        "h3_reference_audio_timestep": 1.0,
        "h3_target_has_soundtrack": True,
    }
    latents = {"visual": torch.full((1, 2, 96), 2.0), "audio": torch.full((1, 3, 32), 3.0)}
    timesteps = {"visual": _timestep(0.25), "audio": _timestep(0.75)}
    return latents, timesteps, cond


def _forward(model, latents, timesteps, cond):
    return H3TrainPipelineConfig().compute_noise_pred(
        model=model,
        latents_input=latents,
        timesteps_input=timesteps,
        pos_cond=cond,
        neg_cond=None,
        joint_cond=None,
        use_cfg=False,
        cfg_batching=False,
        guidance_scale=0.0,
        true_cfg_scale=None,
    )


def _row_timesteps(model):
    return model.inputs["timestep"][model.inputs["timestep_indices"]]


class _TwoHeads(nn.Module):
    def __init__(self, as_tuple=False):
        super().__init__()
        self.visual_scale = nn.Parameter(torch.tensor(2.0))
        self.audio_scale = nn.Parameter(torch.tensor(3.0))
        self.as_tuple = as_tuple
        self.inputs = None

    def forward(self, **kwargs):
        self.inputs = kwargs
        video = kwargs["hidden_states"] * self.visual_scale
        audio = kwargs["audio_hidden_states"] * self.audio_scale
        if self.as_tuple:
            return video, audio
        return Namespace(sample=video, audio_sample=audio)


@pytest.mark.parametrize("has_soundtrack", [True, False])
def test_t2va_joint_pair_feeds_audio_at_its_own_time(has_soundtrack):
    # Packed rows: [text 0-1 | video 2-4 | audio 5-6 | pad 7]; t2va has no reference rows.
    layout = {
        "img_pos": torch.tensor([2, 3, 4]),
        "audio_pos": torch.tensor([5, 6]),
        "text_pos": torch.tensor([0, 1]),
        "update_mask": torch.tensor([True, True, True]),
        "img_position_ids": torch.arange(24.0).view(8, 3),
        "cu_seqlens": torch.tensor([0, 7, 8]),
    }
    cond = {
        "encoder_hidden_states": torch.ones(1, 2, 8),
        "h3_packed_layout": layout,
        "h3_token_tags": torch.tensor([1, 1, 0, 0, 0, 2, 2, -1]),
        "h3_target_has_soundtrack": has_soundtrack,
    }
    latents = {"visual": torch.full((1, 3, 96), 2.0), "audio": torch.full((1, 2, 32), 3.0)}
    model = _TwoHeads()
    output = H3TrainPipelineConfig().compute_noise_pred(
        model=model,
        latents_input=latents,
        timesteps_input={"visual": _timestep(0.25), "audio": _timestep(0.75)},
        pos_cond=cond,
        neg_cond=None,
        joint_cond=None,
        use_cfg=False,
        cfg_batching=False,
        guidance_scale=0.0,
        true_cfg_scale=None,
    )
    assert torch.equal(model.inputs["audio_hidden_states"], latents["audio"])
    row_times = _row_timesteps(model)
    assert torch.allclose(row_times[layout["img_pos"]], torch.full((3,), 0.75))
    assert torch.allclose(row_times[layout["audio_pos"]], torch.full((2,), 0.25))
    assert torch.allclose(row_times[layout["text_pos"]], torch.full((2,), 0.75))
    assert torch.equal(output["visual"], -latents["visual"] * model.visual_scale)
    if has_soundtrack:
        assert torch.equal(output["audio"], -latents["audio"] * model.audio_scale)
    else:
        assert output.keys() == {"visual"}


@pytest.mark.parametrize("as_tuple", [False, True])
def test_ref2va_scatter_independent_timesteps_and_velocity_direction(as_tuple):
    latents, timesteps, cond = _case()
    model = _TwoHeads(as_tuple=as_tuple)
    output = _forward(model, latents, timesteps, cond)
    inputs = model.inputs
    layout = cond["h3_packed_layout"]

    assert inputs["hidden_states"].shape == (1, 4, 96)
    assert inputs["audio_hidden_states"].shape == (1, 5, 32)
    for name, input_key, mask_key, scale in (
        ("visual", "hidden_states", "update_mask", model.visual_scale),
        ("audio", "audio_hidden_states", "audio_update_mask", model.audio_scale),
    ):
        mask = layout[mask_key]
        assert torch.equal(inputs[input_key][:, mask], latents[name])
        assert torch.equal(inputs[input_key][:, ~mask], cond[f"h3_reference_{name}"][None])
        # Both transformer outputs include reference rows. Training consumes
        # only target rows, preserving B even when micro-batch size is one.
        assert output[name].shape == latents[name].shape
        assert torch.equal(output[name], -latents[name] * scale)

    row_times = _row_timesteps(model)
    assert row_times.shape == (11,)
    assert torch.allclose(row_times[layout["img_pos"]], torch.tensor([0.999, 0.999, 0.75, 0.75]))
    assert torch.allclose(row_times[layout["audio_pos"]], torch.tensor([1.0, 1.0, 0.25, 0.25, 0.25]))
    assert torch.allclose(row_times[layout["text_pos"]], torch.tensor([0.75, 0.75]))
    assert inputs["token_tags"].tolist() == [1, 1, 2, 0, 2, 2, 0, 2, 2, 0, 0]
    assert torch.equal(inputs["position_ids"], layout["img_position_ids"][:11])


def test_audio_timestep_changes_do_not_change_video_or_reference_rows():
    latents, timesteps, cond = _case()
    model = _TwoHeads()
    _forward(model, latents, timesteps, cond)
    before = _row_timesteps(model)
    _forward(model, latents, {**timesteps, "audio": _timestep(0.5)}, cond)
    after = _row_timesteps(model)
    target_audio_rows = cond["h3_packed_layout"]["audio_pos"][cond["h3_packed_layout"]["audio_update_mask"]]
    unchanged = torch.ones(11, dtype=torch.bool)
    unchanged[target_audio_rows] = False
    assert torch.equal(before[unchanged], after[unchanged])
    assert torch.equal(after[target_audio_rows], torch.full((3,), 0.5))


def test_silent_target_feeds_its_audio_rows_but_predicts_only_video():
    latents, timesteps, cond = _case()
    cond["h3_target_has_soundtrack"] = False
    model = _TwoHeads()
    output = _forward(model, latents, timesteps, cond)
    layout = cond["h3_packed_layout"]
    assert torch.equal(model.inputs["audio_hidden_states"][:, layout["audio_update_mask"]], latents["audio"])
    target_audio_rows = layout["audio_pos"][layout["audio_update_mask"]]
    assert torch.equal(_row_timesteps(model)[target_audio_rows], torch.full((3,), 0.25))
    assert output.keys() == {"visual"}


def test_reference_augmentation_timesteps_are_taken_from_conditioning():
    latents, timesteps, cond = _case()
    cond["h3_reference_visual_timestep"] = 0.9
    cond["h3_reference_audio_timestep"] = 0.95
    model = _TwoHeads()
    _forward(model, latents, timesteps, cond)
    rows = _row_timesteps(model)
    layout = cond["h3_packed_layout"]
    assert torch.allclose(rows[layout["img_pos"][:2]], torch.tensor([0.9, 0.9]))
    assert torch.allclose(rows[layout["audio_pos"][:2]], torch.tensor([0.95, 0.95]))


def test_reference_timesteps_follow_native_clamp_when_targets_are_cleaner():
    latents, _, cond = _case()
    cond["h3_reference_audio_timestep"] = 0.95
    model = _TwoHeads()
    _forward(model, latents, {"visual": _timestep(0.00001), "audio": _timestep(0.00001)}, cond)
    rows = _row_timesteps(model)
    layout = cond["h3_packed_layout"]
    # Native H3 pins the reference latent but conditions its rows at
    # max(target timestep, reference augmentation timestep).
    assert torch.allclose(rows[layout["img_pos"][:2]], torch.tensor([0.99999, 0.99999]))
    assert torch.allclose(rows[layout["audio_pos"][:2]], torch.tensor([0.99999, 0.99999]))


def _sft_args(**overrides):
    return Namespace(
        **(
            {
                "diffusion_task": "ref2va",
                "loss_type": "sft_loss",
                "train_only": True,
                "update_weight_target_modules": ["transformer_ref"],
                "micro_batch_size": 1,
                "fsdp_flow_shift": {"visual": 8.0, "audio": 3.0},
                "seed": 42,
                "log_loss_sigma_bucket": 5,
                "diffusion_supervised_streams": ["visual", "audio"],
            }
            | overrides
        )
    )


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"loss_type": "policy_loss", "train_only": False, "use_lora": True, "lora_ipc_weight_sync": True}, "t2va"),
        ({"update_weight_target_modules": ["transformer"]}, "transformer_ref"),
        ({"micro_batch_size": 2}, "micro-batch-size 1"),
        ({"fsdp_flow_shift": {"visual": 8.0}}, "audio"),
        ({"fsdp_flow_shift": None}, "audio"),
    ],
)
def test_ref2va_validation_rejects_incompatible_training_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        H3TrainPipelineConfig.validate_args(_sft_args(**overrides))


@pytest.mark.parametrize("forward_dtype", [torch.float32, torch.bfloat16])
def test_real_tiny_diffusers_h3_joint_sft_forward_and_backward(forward_dtype):
    # Exercise the real attention, timestep/modality AdaLN tables, modality
    # projections and both heads. No pretrained weights, CUDA, or encoders.
    module = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = module.MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            hidden_size=24,
            num_layers=1,
            num_refiner_layers=1,
            ffn_dim=32,
            in_channels=24,
            audio_in_channels=32,
            patch_size=(1, 2, 2),
            text_dim=8,
            freq_dim=8,
            time_embed_hidden_dim=24,
            time_embed_dim=16,
            rope_freq_dim=2,
        )
    if forward_dtype == torch.bfloat16:
        model.to(dtype=forward_dtype)
        # Match the released mixed checkpoint: input/output projections,
        # timestep MLP and RoPE stay fp32; attention/refiner/block weights bf16.
        for name, layer in model.named_modules():
            if any(fragment in name for fragment in model._keep_in_fp32_modules):
                layer.float()
    config = H3TrainPipelineConfig()
    args = _sft_args()
    H3TrainPipelineConfig.validate_args(args)
    latents, _, cond = _case()
    batch = [{"latent": {name: latent[0] for name, latent in latents.items()}, "cond_kwargs": cond}]
    context = DiffusionLossContext(
        models={"transformer_ref": model},
        train_pipeline_config=config,
        sde_backend=None,
        args=args,
        forward_dtype=forward_dtype,
        device=torch.device("cpu"),
    )
    prepared = prepare_sft_batch(context, batch)
    with torch.autocast("cpu", dtype=forward_dtype, enabled=forward_dtype != torch.float32):
        prediction = config.compute_noise_pred(
            model=model,
            latents_input=prepared.latents,
            timesteps_input=prepared.timesteps_for_model,
            pos_cond=prepared.pos_cond,
            neg_cond=None,
            joint_cond=None,
            use_cfg=False,
            cfg_batching=False,
            guidance_scale=0.0,
            true_cfg_scale=None,
        )
    assert {name: value.shape for name, value in prediction.items()} == {
        "visual": torch.Size((1, 2, 96)),
        "audio": torch.Size((1, 3, 32)),
    }
    metrics = new_metric_buffer(None, torch.device("cpu"), ["transformer_ref"], sigma_buckets=5)
    loss = sft_loss_formula(
        context, batch, prepared, new_pred=prediction, old_pred=None, ref_pred=None, metrics=metrics
    )
    assert torch.isfinite(loss) and loss.item() > 0
    assert prepared.extras["target"]["visual"].dtype == torch.float32
    assert prepared.extras["sigmas"]["audio"].dtype == torch.float32
    loss.backward()
    for layer in (
        model.proj_in,
        model.audio_proj_in,
        model.proj_out,
        model.audio_proj_out,
        model.time_embedder.linear_1,
    ):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
