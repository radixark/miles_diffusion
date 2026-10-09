"""H3 Ref2VA encode on CPU: one SFT sample -> one cached joint video/audio train pair.

    Sample(prompt, conditions=[image | video | video_audio | audio ...], target={"visual"})
      │
      ├─ checkpoint     FL2VA/video_vae + FL2VA/audio_vae + root text_encoder / processor (shared by all tasks)
      ├─ canvas         rounded pixel size -> native H3 aspect-ratio preset
      ├─ request        conditions go to the native validator verbatim (unknown fields rejected there)
      ├─ target audio   the target video's soundtrack over its frames' interval (first frame + frames / 24 s);
      │                 short or a separate audio file = error; a silent video = zero latent rows
      │                 marked h3_target_has_soundtrack=False, so the trainer supervises its video only
      ├─ references     ordered rows per condition (keyframes lead), native noise augmentation per block
      │                 at SGLang's condition timesteps, which the pair also hands the trainer
      └─ presentation   H3 labels + vision tokens -> HF Qwen3-VL with modality ids and multimodal RoPE
      ▼
    {"latent": {"visual": rows, "audio": rows}, "cond_kwargs": packed layout + reference rows}

What each test pins:
  canvas / request  preset mapping; unknown condition fields reach the native validator
  HF adapter        modality ids, grids and RoPE positions reach Qwen3-VL, every tensor on the model's device
  target audio      per-channel crop over the video interval, which lasts target frames / 24 s whatever the
                    file's average fps; short audio fails; a separate audio file is rejected; a silent target
                    yields zero latent rows of the interval's length without decoding audio
  target cadence    a target averaging 24 fps with unevenly spaced frames is rejected: its audio would drift
  references        video/audio pairs keep their order and boundaries through augmentation, which runs at
                    SGLang's condition timesteps
  pair              two targets marked as having a soundtrack, SGLang's condition timesteps, weights-only cache load
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import io
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from miles.rollout.encoder_hub.h3 import common, ref2va
from miles.utils.types import Sample

NATIVE = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3"


def native_stub(monkeypatch, suffix, **symbols):
    module = types.ModuleType(f"{NATIVE}.{suffix}")
    module.__dict__.update(symbols)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_canvas_maps_rounded_pixels_to_semantic_preset(monkeypatch):
    # These two grids differ from the exact ratios due to native area cap and
    # nearest-32 rounding. Feeding 1344:768 or reducing to 7:4 is invalid.
    shapes = {(16, 9, 768): (768, 1344), (21, 9, 768): (672, 1536)}

    def resolve(*, width, height, base_short_edge):
        h, w = shapes.get((width, height, base_short_edge), (base_short_edge, base_short_edge))
        return {"height": h, "width": w}

    native_stub(monkeypatch, "resolved_plan", MINIMAX_H3_BASE_SHORT_EDGE=768, minimax_h3_resolve_spatial_shape=resolve)
    native_stub(monkeypatch, "task_profiles", MINIMAX_H3_FINITE_ASPECT_RATIOS=("21:9", "16:9"))
    assert ref2va._target_canvas_spec(768, 1344) == {"short_edge": 768, "aspect_ratio": "16:9"}
    assert ref2va._target_canvas_spec(672, 1536) == {"short_edge": 768, "aspect_ratio": "21:9"}
    with pytest.raises(ValueError, match="aspect-ratio preset"):
        ref2va._target_canvas_spec(768, 1408)


def test_request_passes_unknown_condition_fields_to_native_validator(monkeypatch):
    def validate(**kwargs):
        assert kwargs["conditions"][0]["misspelled_seek"] == 2.0
        assert kwargs["target"]["aspect_ratio"] == "16:9"
        assert kwargs["target"]["duration_seconds"] == 107 / 24
        raise ValueError("conditions[0] has unknown fields: misspelled_seek")

    native_stub(monkeypatch, "request_validation", minimax_h3_validate_canonical_request=validate)
    native_stub(monkeypatch, "prequeue", minimax_h3_prepare_for_queue=Mock())
    monkeypatch.setattr(ref2va, "_target_canvas_spec", lambda h, w: {"short_edge": 768, "aspect_ratio": "16:9"})
    with pytest.raises(ValueError, match="unknown fields"):
        ref2va._reference_request(
            Sample(
                prompt="sing",
                conditions=[{"type": "video", "uri": "a.mp4", "role": "reference", "misspelled_seek": 2.0}],
            ),
            {"video": torch.empty(3, 107, 768, 1344, device="meta")},
            123,
        )


def test_hf_adapter_passes_modality_ids_and_multimodal_rope():
    class Qwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(image_token_id=11, video_token_id=12)

        def get_rope_index(self, *, input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask):
            assert mm_token_type_ids.tolist() == [[0, 1, 2, 0]]
            assert image_grid_thw.tolist() == [[1, 2, 2]]
            assert video_grid_thw.tolist() == [[1, 2, 2]]
            return torch.arange(4).expand(3, 1, 4), None

        def forward(self, **kwargs):
            self.received = kwargs
            return SimpleNamespace(last_hidden_state=torch.zeros(1, 4, 5120))

    # A meta-device encoder exposes any tensor left on the host.
    model = Qwen().to("meta")
    hidden = ref2va._encode_ids(
        {"text_encoder": model, "device": torch.device("meta")},
        torch.tensor([3, 11, 12, 4]),
        pixel_values=torch.zeros(1, 4),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        pixel_values_videos=torch.zeros(1, 4),
        video_grid_thw=torch.tensor([[1, 2, 2]]),
    )
    assert hidden.shape == (4, 5120)
    assert all(value.device.type == "meta" for value in model.received.values() if isinstance(value, torch.Tensor))
    assert model.received["mm_token_type_ids"].shape == (1, 4)
    assert model.received["pixel_values"].dtype == torch.bfloat16
    assert model.received["position_ids"].shape == (3, 1, 4)
    assert model.received["use_cache"] is False


def test_audio_uses_video_interval_and_crops_padding_per_channel(monkeypatch):
    # Distinct channels catch accidental flat-prefix truncation: row 3 is
    # channel zero's padding, not the start of channel one's supervision.
    raw = torch.arange(8 * 32).reshape(8, 32).float()
    encode = Mock(return_value={"rows": raw, "ref_audio_t": 4, "duration_seconds": 2 / 24})
    native_stub(monkeypatch, "reference_encoding", minimax_h3_encode_reference_audio_rows=encode)
    native_stub(monkeypatch, "time_request", minimax_h3_audio_latent_t=lambda duration: round(duration * 40))
    sample = Sample(target={"visual": "target.mp4"})
    # A two-frame 24 fps window in a file averaging 27 fps: the audio spans the window's 2/24 s.
    media_clip = {
        "video": torch.zeros(3, 2, 1, 1, dtype=torch.uint8),
        "fps": 27.0,
        "frame_times_seconds": (2.0, 2.0 + 1 / 24),
        "audio_sample_rate": 48000,
    }
    result = common.encode_target_audio(
        {"audio_vae": object(), "audio_vae_arch": object()},
        sample,
        media_clip,
    )
    assert torch.equal(result, torch.cat((raw[:3], raw[4:7])))
    assert encode.call_args.args[1] == "target.mp4"
    assert encode.call_args.kwargs["start_time_seconds"] == 2.0
    assert encode.call_args.kwargs["max_duration_seconds"] == 2 / 24
    # The soundtrack decodes at its native rate, so it is resampled once.
    assert encode.call_args.kwargs["material_chain"] == "audio"
    assert encode.call_args.kwargs["source_sample_rate"] == 48000


def test_short_target_audio_fails_instead_of_padding_or_partial_supervision(monkeypatch):
    native_stub(
        monkeypatch,
        "reference_encoding",
        minimax_h3_encode_reference_audio_rows=lambda *a, **kw: {
            "rows": torch.zeros(6, 32),
            "ref_audio_t": 3,
            "duration_seconds": 0.05,
        },
    )
    native_stub(monkeypatch, "time_request", minimax_h3_audio_latent_t=lambda duration: round(duration * 40))
    with pytest.raises(ValueError, match="shorter than the selected video interval"):
        common.encode_target_audio(
            {"audio_vae": object(), "audio_vae_arch": object()},
            Sample(target={"visual": "clip.mp4"}),
            {
                "video": torch.zeros(3, 2, 1, 1, dtype=torch.uint8),
                "frame_times_seconds": (1.0, 1.0 + 1 / 24),
                "audio_sample_rate": 32000,
            },
        )


def test_separate_target_audio_file_is_rejected():
    with pytest.raises(ValueError, match="own soundtrack"):
        common.encode_target_audio(
            {"audio_vae": object(), "audio_vae_arch": object()},
            Sample(target={"visual": "clip.mp4", "audio": "clip.wav"}),
            {"frame_times_seconds": (0.0,), "audio_sample_rate": 48000},
        )


def test_silent_target_yields_zero_latent_rows_without_decoding_audio(monkeypatch):
    # Two frames last 2/24 s -> 3 audio latent frames per channel -> 2 x 3 rows of zeros.
    encode = Mock()
    native_stub(monkeypatch, "reference_encoding", minimax_h3_encode_reference_audio_rows=encode)
    native_stub(monkeypatch, "time_request", minimax_h3_audio_latent_t=lambda duration: round(duration * 40))
    rows = common.encode_target_audio(
        {"audio_vae": object(), "audio_vae_arch": object()},
        Sample(target={"visual": "silent.mp4"}),
        {
            "video": torch.zeros(3, 2, 1, 1, dtype=torch.uint8),
            "frame_times_seconds": (0.0, 1 / 24),
            "audio_sample_rate": None,
        },
    )
    assert torch.equal(rows, torch.zeros(6, 32))
    encode.assert_not_called()


def reference_fixture():
    # Request order interleaves media; keyframes become a separate leading
    # visual block while the independent references retain their order.
    materials = [
        SimpleNamespace(condition_type=kind, condition_index=index, material_chain=chain)
        for index, (kind, chain) in enumerate(
            [
                ("audio", "audio"),
                ("video", "video.reference_preserve"),
                ("image", "image.reference_preserve"),
                ("video_audio", "video_audio.reference_preserve"),
                ("image", "image.target_canvas"),
            ]
        )
    ]
    keyframe = {"rows": torch.full((1, 96), 50.0), "latent_h": 2, "latent_w": 2}
    extra = {
        "minimax_h3_reference_image_rows": {
            "images": [
                {"condition_index": 2, "rows": torch.full((1, 96), 30.0), "latent_h": 2, "latent_w": 2},
            ]
        },
        "minimax_h3_reference_video_rows": {
            "videos": [
                {
                    "condition_index": index,
                    "rows": torch.full((2, 96), value),
                    "latent_t": 2,
                    "latent_h": 2,
                    "latent_w": 2,
                }
                for index, value in [(1, 20.0), (3, 40.0)]
            ]
        },
        "minimax_h3_reference_audio_rows": {
            "audios": [
                {"condition_index": index, "rows": torch.full((2 * length, 32), value), "ref_audio_t": length}
                for index, length, value in [(0, 1, 10.0), (1, 0, 20.0), (3, 2, 40.0)]
            ]
        },
        "minimax_h3_keyframe_cond_rows": {
            "rows": keyframe["rows"],
            "keyframes": [keyframe],
            "semantic_frame_indices": [0],
            "frame_count": 107,
        },
    }
    return SimpleNamespace(materials=materials), extra


def test_ordered_reference_rows_preserve_video_audio_pairs_and_hybrid_keyframe():
    plan, extra = reference_fixture()
    result = ref2va._ordered_reference_rows(plan, extra)
    assert [block["kind"] for block in result["blocks"]] == ["audio", "video", "image", "video_audio"]
    assert result["visual"][:, 0].tolist() == [50, 20, 20, 30, 40, 40]
    assert result["audio"][:, 0].tolist() == [10, 10, 40, 40, 40, 40]
    assert result["visual_shapes"] == [(1, 2, 2), (2, 2, 2), (1, 2, 2), (2, 2, 2)]
    assert result["audio_latent_ts"] == [1, 2]
    assert result["blocks"][1]["ref_audio_t"] == 0


def native_condition_timesteps(monkeypatch):
    # Values unlike the real 0.999 / 1.0, so a hardcoded copy of SGLang's constants fails.
    native_stub(
        monkeypatch, "denoise_loop", MINIMAX_H3_IMGVID_COND_TIMESTEP=0.25, MINIMAX_H3_AUDIO_REF_COND_TIMESTEP=0.75
    )


def test_augmentation_keeps_each_reference_boundary_at_native_condition_timesteps(monkeypatch):
    native_condition_timesteps(monkeypatch)
    plan, extra = reference_fixture()
    references = ref2va._ordered_reference_rows(plan, extra)
    video_aug = Mock(side_effect=lambda rows, **kwargs: rows + 0.001)
    audio_aug = Mock(side_effect=lambda rows, **kwargs: rows)
    native_stub(
        monkeypatch,
        "condition_noise",
        minimax_h3_imgvid_cond_noise_aug_rows=video_aug,
        minimax_h3_audio_cond_noise_aug_rows=audio_aug,
    )
    visual, audio = ref2va._augment_references(references, latent_t=32, seed=123)
    assert video_aug.call_args.kwargs == {
        "condition_shapes": [(1, 2, 2), (2, 2, 2), (1, 2, 2), (2, 2, 2)],
        "target_latent_t": 32,
        "imgvid_cond_num_frames": 4,
        "seed": 123,
        "noise_aug": 0.25,
    }
    assert audio_aug.call_args.kwargs == {"condition_audio_t": [1, 2], "seed": 123, "noise_aug": 0.75}
    assert torch.equal(audio, references["audio"])
    assert torch.allclose(visual, references["visual"] + 0.001)


def test_ref2va_pair_has_two_targets_and_cache_can_load_weights_only(monkeypatch):
    native_condition_timesteps(monkeypatch)
    plan, extra = reference_fixture()
    plan.shape = {"frame_count": 5, "height": 32, "width": 32}
    native_stub(
        monkeypatch,
        "reference_encoding",
        minimax_h3_encode_reference_video_rows=lambda *args: (
            torch.full((2, 96), 7.0),
            2,
            2,
            2,
        ),
    )
    packed = {
        "token_tags": torch.tensor([1, 1, 0, 0]),
        "text_pos": torch.tensor([0, 1]),
        "update_mask": torch.tensor([False] * 6 + [True] * 2),
        "audio_update_mask": torch.tensor([False] * 6 + [True] * 4),
    }
    pack = Mock(return_value=packed)
    native_stub(monkeypatch, "packed_sequence", minimax_h3_packed_sequence_ref2va_blocks=pack)
    batch = SimpleNamespace(extra=extra)
    monkeypatch.setattr(ref2va, "_reference_request", lambda *args: (batch, plan))
    monkeypatch.setattr(common, "encode_target_audio", lambda *args: torch.full((4, 32), 9.0))
    monkeypatch.setattr(
        ref2va,
        "_encode_text_and_fill_reference_rows",
        lambda *args: {
            "text_len": 2,
            "hidden_states": torch.zeros(2, 5120),
            "text_token_tags": torch.tensor([1, 0]),
        },
    )
    monkeypatch.setattr(ref2va, "_augment_references", lambda refs, **kwargs: (refs["visual"], refs["audio"]))
    prompt = "A person sings"
    pair = ref2va.encode_sample(
        {"vae": None, "vae_arch": None},
        Sample(prompt=prompt, target={"visual": "target.mp4"}),
        {
            "fps": 24.0,
            "frame_times_seconds": tuple(index / 24 for index in range(5)),
            "video": torch.zeros(3, 5, 32, 32, dtype=torch.uint8),
            "audio_sample_rate": 48000,
        },
        torch.Generator().manual_seed(123),
        SimpleNamespace(),
    )
    assert pair["latent"]["visual"].shape == (2, 96)
    assert pair["latent"]["audio"].shape == (4, 32)
    assert pair["cond_kwargs"]["h3_reference_audio"].shape == (6, 32)
    assert pair["cond_kwargs"]["h3_token_tags"].tolist() == [1, 0, 0, 0]
    assert pair["cond_kwargs"]["h3_target_has_soundtrack"] is True
    assert pair["cond_kwargs"]["h3_reference_visual_timestep"] == 0.25
    assert pair["cond_kwargs"]["h3_reference_audio_timestep"] == 0.75
    assert pack.call_args.kwargs["keyframe_frame_indices"] == [0]
    assert pack.call_args.kwargs["frame_count"] == 107
    assert pair["prompt"] == prompt
    buffer = io.BytesIO()
    torch.save(pair, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    assert torch.equal(restored["latent"]["audio"], torch.full((4, 32), 9.0))


def test_variable_rate_target_averaging_24_fps_is_rejected(monkeypatch):
    # Five frames over 4/24 s average 24 fps, but frames 1 and 3 sit half a frame early:
    #   constant 24 fps  |     |     |     |     |
    #   this target      |  |        |  |        |
    # The target audio is cropped to 5/24 s from frame 0, so it would drift off these frames.
    for suffix in ("denoise_loop", "packed_sequence", "reference_encoding"):
        monkeypatch.setitem(sys.modules, f"{NATIVE}.{suffix}", Mock())
    media_clip = {
        "fps": 24.0,
        "frame_times_seconds": (0.0, 0.5 / 24, 2 / 24, 2.5 / 24, 4 / 24),
        "video": torch.zeros(3, 5, 32, 32, dtype=torch.uint8),
    }
    with pytest.raises(ValueError, match="constant 24 fps"):
        ref2va.encode_sample(
            {},
            Sample(target={"visual": "target.mp4"}),
            media_clip,
            torch.Generator().manual_seed(0),
            SimpleNamespace(),
        )
