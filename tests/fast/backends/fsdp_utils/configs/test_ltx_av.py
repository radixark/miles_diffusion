"""LTX audio-video training forward:

    latents {"visual"}           --> model(video=V, audio=None) --> {"visual"}           the video-only rounding path
    latents {"visual", "audio"}  --> model(video=V, audio=A)    --> {"visual", "audio"}  each from its own Modality
    A.positions                  == the engine's audio RoPE coordinates, kept in fp32
    validate_args                --> patch group and extra streams agree; no engine SP for audio
    an audio command line         --> passes the real parser
    audio with patch group ltx    --> rejected on the real parser too (the parser calls the checks)

ltx_core is replaced by a stand-in, so the file runs without it.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import dataclasses
import shlex
import sys
import types
from argparse import Namespace
from dataclasses import dataclass

import pytest
import torch

from miles.backends.fsdp_utils.configs.ltx import LTXTrainPipelineConfig
from miles.backends.fsdp_utils.models.ltx.positions import prepare_audio_positions, prepare_video_positions

HEIGHT, WIDTH, NUM_FRAMES, FPS = 64, 96, 9, 24  # 2 x 2 x 3 latent grid -> 12 video tokens
NUM_AUDIO_FRAMES = 7


@dataclass(frozen=True)
class _Modality:
    latent: torch.Tensor
    sigma: torch.Tensor
    timesteps: torch.Tensor
    positions: torch.Tensor
    context: torch.Tensor
    enabled: bool = True
    context_mask: torch.Tensor | None = None


# copied from ltx_core.utils.to_denoised
def _to_denoised(sample, velocity, sigma, calc_dtype=torch.float32):
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)


@pytest.fixture(autouse=True)
def fake_ltx_core(monkeypatch):
    modality = types.ModuleType("ltx_core.model.transformer.modality")
    modality.Modality = _Modality
    utils = types.ModuleType("ltx_core.utils")
    utils.to_denoised = _to_denoised
    for name in ("ltx_core", "ltx_core.model", "ltx_core.model.transformer"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "ltx_core.model.transformer.modality", modality)
    monkeypatch.setitem(sys.modules, "ltx_core.utils", utils)


class _RecordingModel:
    """Records each call; the velocity depends on latent, positions, context and timesteps."""

    def __init__(self):
        self.calls = []

    @staticmethod
    def _velocity(modality):
        if modality is None:
            return None
        shift = modality.positions.float().mean() + modality.context.float().mean() + modality.timesteps.float().mean()
        return (modality.latent.flip(-1) * 0.37 + shift).to(modality.latent.dtype)

    def __call__(self, *, video, audio, perturbations):
        self.calls.append({"video": video, "audio": audio})
        return self._velocity(video), self._velocity(audio)


def _config() -> LTXTrainPipelineConfig:
    config = LTXTrainPipelineConfig()
    config.configure(
        Namespace(
            diffusion_height=HEIGHT, diffusion_width=WIDTH, diffusion_output_num_frames=NUM_FRAMES, diffusion_fps=FPS
        )
    )
    return config


def _inputs(with_audio: bool):
    # Three samples with distinct sigmas, one at 0 (the clamp); fp32 contexts, so the forward's own cast is pinned.
    gen = torch.Generator().manual_seed(0)
    latents = {"visual": torch.randn(3, 12, 128, generator=gen).bfloat16()}
    timesteps = {"visual": torch.tensor([913.7, 0.0, 412.5])}
    cond = {"context": torch.randn(3, 16, 32, generator=gen)}
    if with_audio:
        latents["audio"] = torch.randn(3, NUM_AUDIO_FRAMES, 128, generator=gen).bfloat16()
        # The engine gives both streams one sigma grid; different values here pin which stream feeds which Modality.
        timesteps["audio"] = torch.tensor([497.3, 0.0, 268.1])
        cond["audio_context"] = torch.randn(3, 16, 24, generator=gen)
    return latents, timesteps, cond


def _compute_noise_pred(config, model, latents, timesteps, cond):
    return config.compute_noise_pred(
        model=model,
        latents_input=latents,
        timesteps_input=timesteps,
        pos_cond=cond,
        neg_cond=None,
        joint_cond=None,
        use_cfg=False,
        cfg_batching=False,
        guidance_scale=1.0,
        true_cfg_scale=None,
    )


def _video_only_reference(config, model, latents, timesteps, cond):
    """The video-only forward the strict e2e standards were recorded against, written out op by op."""
    latents_input = latents["visual"]
    cond = dict(cond)
    batch_size, num_tokens, _ = latents_input.shape
    cond["positions"] = prepare_video_positions(
        batch_size=batch_size,
        num_tokens=num_tokens,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        fps=FPS,
        device=latents_input.device,
        dtype=latents_input.dtype,
    )
    dtype = latents_input.dtype
    B = latents_input.shape[0]
    sigma_scaled = timesteps["visual"].to(latents_input.dtype)
    sigma_unit = sigma_scaled / float(config.num_train_timesteps)
    per_token_t = sigma_unit.view(B, 1).to(dtype)
    video_modality = _Modality(
        enabled=True,
        latent=latents_input,
        sigma=sigma_unit.reshape(B),
        timesteps=per_token_t,
        positions=cond["positions"].to(dtype),
        context=cond["context"].to(dtype),
        context_mask=None,
    )
    velocity, _ = model(video=video_modality, audio=None, perturbations=None)
    x0_pred = _to_denoised(latents_input, velocity, per_token_t.unsqueeze(-1)).float()
    sigma_safe = torch.clamp(sigma_unit, min=1e-8).view(B, 1, 1)
    return ((latents_input.float() - x0_pred) / sigma_safe).to(dtype)


class TestVideoOnlyForward:
    # Catches any change to the video-only rounding path (a cast, an op order) that strict e2e standards would see.
    def test_video_only_forward_keeps_its_rounding_path(self):
        config, latents, timesteps, cond = _config(), *_inputs(with_audio=False)
        model, reference_model = _RecordingModel(), _RecordingModel()

        noise_pred = _compute_noise_pred(config, model, latents, timesteps, cond)
        reference = _video_only_reference(config, reference_model, latents, timesteps, cond)

        assert noise_pred.keys() == {"visual"}
        assert torch.equal(noise_pred["visual"], reference)
        (call,), (reference_call,) = model.calls, reference_model.calls
        assert call["audio"] is None
        for field in dataclasses.fields(_Modality):
            new, old = getattr(call["video"], field.name), getattr(reference_call["video"], field.name)
            if isinstance(old, torch.Tensor):
                assert new.dtype == old.dtype and torch.equal(new, old), field.name
            else:
                assert new == old, field.name


class TestJointForward:
    # Catches the audio Modality built from the wrong stream's inputs, with bf16 positions, or left uncast.
    def test_audio_modality_inputs(self):
        config, latents, timesteps, cond = _config(), *_inputs(with_audio=True)
        model = _RecordingModel()

        noise_pred = _compute_noise_pred(config, model, latents, timesteps, cond)

        assert noise_pred.keys() == {"visual", "audio"}
        assert noise_pred["audio"].shape == latents["audio"].shape
        (call,) = model.calls
        audio = call["audio"]
        assert audio.enabled is True
        assert audio.context_mask is None
        assert audio.latent is latents["audio"]
        assert torch.equal(audio.context, cond["audio_context"].bfloat16())
        assert torch.equal(audio.sigma, timesteps["audio"].bfloat16() / 1000.0)
        assert audio.timesteps.dtype == torch.bfloat16
        assert torch.equal(audio.timesteps, audio.sigma.view(3, 1))
        assert audio.positions.dtype == torch.float32
        expected_positions = prepare_audio_positions(batch_size=3, num_frames=NUM_AUDIO_FRAMES, device="cpu")
        assert torch.equal(audio.positions, expected_positions)

    # Catches a stream converted with another stream's sigma, one sigma for the whole batch, or a missing clamp.
    def test_each_stream_converts_its_own_velocity(self):
        # noise_pred = (x - x0) / sigma with x0 = x - sigma * v in fp32, per sample; sigma 0 is clamped to 1e-8.
        config, latents, timesteps, cond = _config(), *_inputs(with_audio=True)
        model = _RecordingModel()

        noise_pred = _compute_noise_pred(config, model, latents, timesteps, cond)

        (call,) = model.calls
        for stream_name, modality in (("visual", call["video"]), ("audio", call["audio"])):
            sigma = modality.sigma.float().view(3, 1, 1)
            velocity = model._velocity(modality)
            x0 = (latents[stream_name].float() - velocity.float() * sigma).to(torch.bfloat16)
            expected = ((latents[stream_name].float() - x0.float()) / sigma.clamp(min=1e-8)).to(torch.bfloat16)
            assert torch.equal(noise_pred[stream_name], expected), stream_name

    # Catches a prediction cut off from the graph (a detached velocity): the loss would not train that stream.
    def test_each_prediction_carries_the_gradient(self):
        config, latents, timesteps, cond = _config(), *_inputs(with_audio=True)
        latents = {name: latent.clone().requires_grad_() for name, latent in latents.items()}

        noise_pred = _compute_noise_pred(config, _RecordingModel(), latents, timesteps, cond)

        for stream_name in ("visual", "audio"):
            (grad,) = torch.autograd.grad(noise_pred[stream_name].float().sum(), latents[stream_name])
            assert grad.abs().sum() > 0, stream_name


class TestAudioPositions:
    def test_engine_coordinates(self):
        # The engine's LTX2 audio RoPE: frame f spans mel frames [4f - 3, 4f + 1) clipped at 0, 160 samples per mel
        # frame at 16 kHz, so 0.04 s per latent frame after the first.
        positions = prepare_audio_positions(batch_size=2, num_frames=4, device="cpu")
        expected = torch.tensor([[0.0, 0.01], [0.01, 0.05], [0.05, 0.09], [0.09, 0.13]])
        assert positions.shape == (2, 1, 4, 2)
        assert positions.dtype == torch.float32
        assert torch.equal(positions[0, 0], expected)
        assert torch.equal(positions[1], positions[0])


class TestValidateArgs:
    @staticmethod
    def _args(extra_streams, patch_groups, *, gpus_per_engine=1, tp_size=None, sp_degree=None, cfg_parallel=False):
        return Namespace(
            rollout_extra_streams=extra_streams,
            rollout_patch_groups=patch_groups,
            rollout_num_gpus_per_engine=gpus_per_engine,
            sglang_tp_size=tp_size,
            sglang_sp_degree=sp_degree,
            sglang_enable_cfg_parallel=cfg_parallel,
        )

    @pytest.mark.parametrize(
        ("extra_streams", "patch_groups"),
        [([], ["ltx"]), (["audio"], []), (["audio"], ["wan"])],
    )
    def test_accepted(self, extra_streams, patch_groups):
        LTXTrainPipelineConfig.validate_args(self._args(extra_streams, patch_groups))

    # Catches a run whose engine cross-attention (patch group) does not match the forward it trains.
    @pytest.mark.parametrize(
        ("extra_streams", "patch_groups", "message"),
        [
            (["audio"], ["ltx"], "must not use --rollout-patch-group ltx"),
            (["audio"], ["wan", "ltx"], "must not use --rollout-patch-group ltx"),
            ([], [], "needs --rollout-patch-group ltx"),
            ([], ["wan"], "needs --rollout-patch-group ltx"),
        ],
    )
    def test_patch_group_must_match_the_streams(self, extra_streams, patch_groups, message):
        with pytest.raises(ValueError, match=message):
            LTXTrainPipelineConfig.validate_args(self._args(extra_streams, patch_groups))

    # The engine's SP degree, when unset, is the GPUs TP and CFG parallelism leave over; audio rollout needs it at 1.
    @pytest.mark.parametrize(
        ("gpus_per_engine", "tp_size", "sp_degree", "cfg_parallel", "accepted"),
        [
            (1, None, None, False, True),
            (2, 2, None, False, True),  # md's default for a 2-GPU engine: all GPUs to TP
            (2, 2, 1, False, True),
            (2, None, 1, False, True),  # an explicit SP degree wins over the GPUs TP leaves
            (2, 1, None, False, False),  # TP 1 leaves the second GPU to SP
            (2, None, 2, False, False),
            (2, 1, None, True, True),  # CFG parallelism takes the second GPU
        ],
    )
    def test_audio_rejects_engine_sequence_parallelism(
        self, gpus_per_engine, tp_size, sp_degree, cfg_parallel, accepted
    ):
        args = self._args(
            ["audio"],
            [],
            gpus_per_engine=gpus_per_engine,
            tp_size=tp_size,
            sp_degree=sp_degree,
            cfg_parallel=cfg_parallel,
        )
        if accepted:
            LTXTrainPipelineConfig.validate_args(args)
        else:
            with pytest.raises(ValueError, match="engine sequence parallelism"):
                LTXTrainPipelineConfig.validate_args(args)

    def test_video_only_keeps_engine_sequence_parallelism(self):
        LTXTrainPipelineConfig.validate_args(self._args([], ["ltx"], gpus_per_engine=2, tp_size=1))

    def _parse(self, monkeypatch, patch_group=None, gpus_per_engine=1):
        # Through parse: LTX declares audio, and its checks run after the defaults set TP and split the patch groups.
        from miles.backends.fsdp_utils.arguments import load_fsdp_args
        from miles.utils.arguments import (
            get_miles_extra_args_provider,
            miles_validate_args,
            set_default_diffusion_args,
        )

        argv = (
            "test --hf-checkpoint Lightricks/LTX-2.3 --diffusion-task t2v --prompt-data prompts.jsonl "
            "--rollout-batch-size 2 --n-samples-per-prompt 2 --global-batch-size 4 --num-rollout 1 "
            f"--actor-num-gpus-per-node 2 --rollout-num-gpus 2 --rollout-num-gpus-per-engine {gpus_per_engine} "
            "--micro-batch-size 1 --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
            "--rm-type pickscore --diffusion-sde-type cps --diffusion-guidance-scale 1.0 "
            "--rollout-extra-streams audio" + (f" --rollout-patch-group {patch_group}" if patch_group else "")
        )
        monkeypatch.setattr(sys, "argv", shlex.split(argv))
        args = load_fsdp_args(extra_args_provider=get_miles_extra_args_provider())
        set_default_diffusion_args(args)
        miles_validate_args(args)
        return args

    @pytest.mark.parametrize("gpus_per_engine", [1, 2])
    def test_audio_command_line_parses(self, monkeypatch, gpus_per_engine):
        args = self._parse(monkeypatch, gpus_per_engine=gpus_per_engine)
        assert args.diffusion_model_family == "ltx"
        assert args.rollout_extra_streams == ["audio"]
        assert "ltx" not in args.rollout_patch_groups
        assert args.sglang_tp_size == gpus_per_engine

    def test_a_parsed_command_line_reaches_the_ltx_checks(self, monkeypatch):
        """The rules are tested on hand-built namespaces; a parser that no longer called them would roll audio out
        with the engine's audio-video cross-attention off while the trainer runs the joint forward."""
        with pytest.raises(ValueError, match="must not use --rollout-patch-group ltx"):
            self._parse(monkeypatch, "ltx")
