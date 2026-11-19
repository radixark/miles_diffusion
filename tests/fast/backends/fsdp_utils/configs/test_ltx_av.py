"""LTX audio-video training forward:

    latents {"visual"}           --> model(video=V, audio=None) --> {"visual"}           bit for bit the video-only code
    latents {"visual", "audio"}  --> model(video=V, audio=A)    --> {"visual", "audio"}
    A.positions                  == the engine's audio RoPE bounds, kept in fp32
    validate_args                --> patch group, extra streams and engine SP must agree

ltx_core is replaced by a stand-in (its ``Modality`` fields and its ``to_denoised``, copied from ltx-core 1.1.6), so the
test runs where ltx_core is not installed; GPU parity against the engine is the G0 gate.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

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
    """Deterministic velocities that depend on every Modality field, so a wrong input changes the output."""

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
        Namespace(diffusion_height=HEIGHT, diffusion_width=WIDTH, diffusion_output_num_frames=NUM_FRAMES, diffusion_fps=FPS)
    )
    return config


def _inputs(with_audio: bool):
    gen = torch.Generator().manual_seed(0)
    latents = {"visual": torch.randn(1, 12, 128, generator=gen).bfloat16()}
    timesteps = {"visual": torch.tensor([913.7])}
    cond = {"context": torch.randn(1, 16, 32, generator=gen).bfloat16()}
    if with_audio:
        latents["audio"] = torch.randn(1, NUM_AUDIO_FRAMES, 128, generator=gen).bfloat16()
        # The engine gives both streams one sigma grid; a different value here pins which stream feeds which Modality.
        timesteps["audio"] = torch.tensor([497.3])
        cond["audio_context"] = torch.randn(1, 16, 24, generator=gen).bfloat16()
    return latents, timesteps, cond


def _noise_pred(config, model, latents, timesteps, cond):
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


def _video_only_noise_pred_before_av(config, model, latents, timesteps, cond):
    """Frozen copy of the video-only forward before the audio stream was added (ltx.py at 067fe267)."""
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
    def test_bit_for_bit_the_forward_before_audio(self):
        config, latents, timesteps, cond = _config(), *_inputs(with_audio=False)
        model, reference_model = _RecordingModel(), _RecordingModel()

        noise_pred = _noise_pred(config, model, latents, timesteps, cond)
        reference = _video_only_noise_pred_before_av(config, reference_model, latents, timesteps, cond)

        assert noise_pred.keys() == {"visual"}
        assert torch.equal(noise_pred["visual"], reference)
        ((call,), (reference_call,)) = model.calls, reference_model.calls
        assert call["audio"] is None
        for field in ("latent", "sigma", "timesteps", "positions", "context"):
            new, old = getattr(call["video"], field), getattr(reference_call["video"], field)
            assert new.dtype == old.dtype and torch.equal(new, old), field


class TestJointForward:
    def test_audio_modality_inputs(self):
        config, latents, timesteps, cond = _config(), *_inputs(with_audio=True)
        model = _RecordingModel()

        noise_pred = _noise_pred(config, model, latents, timesteps, cond)

        assert noise_pred.keys() == {"visual", "audio"}
        assert noise_pred["audio"].shape == latents["audio"].shape
        (call,) = model.calls
        audio = call["audio"]
        assert audio.latent is latents["audio"]
        assert torch.equal(audio.context, cond["audio_context"])
        assert torch.equal(audio.sigma, (timesteps["audio"].bfloat16() / 1000.0).reshape(1))
        assert audio.positions.dtype == torch.float32
        assert torch.equal(audio.positions, prepare_audio_positions(batch_size=1, num_frames=NUM_AUDIO_FRAMES, device="cpu"))

    def test_each_stream_converts_its_own_velocity(self):
        # noise_pred = (x - x0) / sigma with x0 = x - sigma * v, done in fp32 per stream.
        config, latents, timesteps, cond = _config(), *_inputs(with_audio=True)
        model = _RecordingModel()

        noise_pred = _noise_pred(config, model, latents, timesteps, cond)

        (call,) = model.calls
        for stream_name, modality in (("visual", call["video"]), ("audio", call["audio"])):
            sigma = modality.sigma.float().view(1, 1, 1)
            velocity = model._velocity(modality)
            x0 = (latents[stream_name].float() - velocity.float() * sigma).to(torch.bfloat16)
            expected = ((latents[stream_name].float() - x0.float()) / sigma).to(torch.bfloat16)
            assert torch.equal(noise_pred[stream_name], expected), stream_name


class TestAudioPositions:
    def test_engine_bounds(self):
        # The engine's LTX2 audio RoPE: frame f spans mel frames [4f - 3, 4f + 1) clipped at 0, 160 samples per mel
        # frame at 16 kHz, so 0.04 s per latent frame after the first.
        positions = prepare_audio_positions(batch_size=2, num_frames=4, device="cpu")
        expected = torch.tensor([[0.0, 0.01], [0.01, 0.05], [0.05, 0.09], [0.09, 0.13]])
        assert positions.shape == (2, 1, 4, 2)
        assert positions.dtype == torch.float32
        torch.testing.assert_close(positions[0, 0], expected, rtol=0, atol=1e-7)
        assert torch.equal(positions[1], positions[0])


class TestValidateArgs:
    @staticmethod
    def _args(extra_streams, patch_groups, sp_degree=None, gpus_per_engine=1):
        return Namespace(
            rollout_extra_streams=extra_streams,
            rollout_patch_groups=patch_groups,
            sglang_sp_degree=sp_degree,
            rollout_num_gpus_per_engine=gpus_per_engine,
        )

    @pytest.mark.parametrize(
        ("extra_streams", "patch_groups"),
        [([], ["ltx"]), (["audio"], ["ltx_av"]), (["audio"], ["ltx_av", "wan"])],
    )
    def test_accepted(self, extra_streams, patch_groups):
        LTXTrainPipelineConfig.validate_args(self._args(extra_streams, patch_groups))

    @pytest.mark.parametrize(
        ("extra_streams", "patch_groups", "message"),
        [
            (["audio"], ["ltx"], "set --rollout-patch-group ltx_av"),
            (["audio"], [], "set --rollout-patch-group ltx_av"),
            (["audio"], ["ltx_av", "ltx"], "set --rollout-patch-group ltx_av"),
            ([], ["ltx_av"], "trains video alone"),
        ],
    )
    def test_patch_group_must_match_the_streams(self, extra_streams, patch_groups, message):
        with pytest.raises(ValueError, match=message):
            LTXTrainPipelineConfig.validate_args(self._args(extra_streams, patch_groups))

    # The engine shards audio under SP; an unset SP degree takes the GPUs TP leaves, so only 1 is safe on 2+ GPUs.
    @pytest.mark.parametrize(
        ("sp_degree", "gpus_per_engine", "accepted"),
        [(None, 1, True), (1, 2, True), (None, 2, False), (2, 2, False)],
    )
    def test_audio_needs_engine_sp_degree_one(self, sp_degree, gpus_per_engine, accepted):
        args = self._args(["audio"], ["ltx_av"], sp_degree=sp_degree, gpus_per_engine=gpus_per_engine)
        if accepted:
            LTXTrainPipelineConfig.validate_args(args)
        else:
            with pytest.raises(ValueError, match="--sglang-sp-degree 1"):
                LTXTrainPipelineConfig.validate_args(args)

    def test_video_only_keeps_engine_sequence_parallelism(self):
        LTXTrainPipelineConfig.validate_args(self._args([], ["ltx"], sp_degree=2, gpus_per_engine=2))

    def test_the_recipe_parses_with_audio(self, monkeypatch):
        # Through parse: the LTX config declares audio, and its own checks run after the patch groups are split.
        from miles.backends.fsdp_utils.arguments import load_fsdp_args
        from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args, set_default_diffusion_args

        argv = (
            "test --hf-checkpoint Lightricks/LTX-2.3 --diffusion-task t2v --prompt-data prompts.jsonl "
            "--rollout-batch-size 2 --n-samples-per-prompt 2 --global-batch-size 4 --num-rollout 1 "
            "--actor-num-gpus-per-node 1 --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 --micro-batch-size 1 "
            "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout --rm-type pickscore "
            "--diffusion-sde-type cps --diffusion-guidance-scale 1.0 "
            "--rollout-extra-streams audio --rollout-patch-group ltx_av"
        )
        monkeypatch.setattr(sys, "argv", shlex.split(argv))
        args = load_fsdp_args(extra_args_provider=get_miles_extra_args_provider())
        set_default_diffusion_args(args)
        miles_validate_args(args)
        assert args.diffusion_model_family == "ltx"
        assert args.rollout_extra_streams == ["audio"]
        assert args.rollout_patch_groups == ["ltx_av"]
