"""H3 encode contract: what validate_args rejects, what the engine geometry yields.

    frames 17n+5 ----.                        .-- latent_t = (f-1)/16*4+... = 32 @107f
    ref2va 4-15 s ---+                        |
    canvas %32 ------+-- validate_args        +-- video rows = t*(H/32)*(W/32) = 32256
    stride == 1 ----'                         '-- packed seq: [text|video|audio|pad] aligned 64

    t2va pair: {"latent": {"visual", "audio" = the soundtrack, one row per packed audio slot},
                "cond_kwargs": packed layout + text + h3_target_has_soundtrack}

Each test pins one side: rejection of off-grid inputs, an engine-derived
geometry invariant the cached latents depend on, or the t2va pair's streams.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

pytest.importorskip("sglang.multimodal_gen")

from miles.rollout.encoder_hub.h3 import ref2va, t2va


def _args(**overrides):
    base = dict(
        sft_encoder_checkpoint="MiniMaxAI/MiniMax-H3",
        diffusion_height=768,
        diffusion_width=1344,
        diffusion_output_num_frames=107,
        sft_frame_stride=1,
    )
    base.update(overrides)
    return Namespace(**base)


class TestValidateArgs:
    def test_serving_grid_spec_passes(self):
        t2va.validate_args(_args())

    def test_rejects_off_grid_frame_count(self):
        with pytest.raises(ValueError, match="17n\\+5"):
            t2va.validate_args(_args(diffusion_output_num_frames=96))

    def test_rejects_wrong_short_edge(self):
        with pytest.raises(ValueError, match="short_edge"):
            t2va.validate_args(_args(diffusion_height=480, diffusion_width=832))

    @pytest.mark.parametrize("num_frames", [90, 379])
    def test_ref2va_rejects_targets_outside_4_to_15_seconds(self, num_frames):
        # Both sit on the 17n+5 grid: 90 frames = 3.75 s, 379 frames = 15.8 s at 24 fps.
        with pytest.raises(ValueError, match="must last 4-15 s"):
            ref2va.validate_args(_args(diffusion_output_num_frames=num_frames))


class TestGeometry:
    def test_t2va_packed_layout_invariants(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
            minimax_h3_packed_sequence,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
            minimax_h3_audio_latent_t,
            minimax_h3_video_latent_t,
        )

        latent_t = minimax_h3_video_latent_t(107)
        assert latent_t == 32
        audio_t = minimax_h3_audio_latent_t(107 / 24)
        assert audio_t == 178

        packed = minimax_h3_packed_sequence(
            text_len=7,
            latent_t=latent_t,
            latent_h=768 // 16,
            latent_w=1344 // 16,
            audio_t=audio_t,
            include_keyframe_cond=False,
        )
        video_rows = latent_t * (768 // 32) * (1344 // 32)
        assert packed["img_pos"].shape[0] == video_rows
        # t2va has no conditioning rows: every video row is a train target.
        assert bool(packed["update_mask"].all())
        assert packed["audio_pos"].shape[0] == audio_t * 2
        assert packed["seq_len"] % 64 == 0
        assert packed["seq_len"] >= 7 + video_rows + audio_t * 2
        assert packed["img_position_ids"].shape == (packed["seq_len"], 3)


def test_t2va_pair_carries_the_soundtrack_in_every_packed_audio_slot(monkeypatch):
    import torch

    from miles.rollout.encoder_hub.h3 import common
    from miles.utils.types import Sample

    native = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3"
    # 5 frames -> 2 latent frames on a 32x32 canvas; the real packer lays out the rows around them.
    monkeypatch.setattr(
        f"{native}.reference_encoding.minimax_h3_encode_reference_video_rows",
        lambda vae, frames, arch: (torch.zeros(2, 96), 2, 2, 2),
    )
    monkeypatch.setattr(f"{native}.presentation.minimax_h3_text_only_ids", lambda tokenizer, prompt: torch.arange(3))
    monkeypatch.setattr(common, "encode_target_audio", lambda encoder, sample, media_clip: torch.ones(18, 32))

    def text_encoder(input_ids, attention_mask, use_cache):
        return Namespace(last_hidden_state=torch.zeros(1, 3, 5120))

    pair = t2va.encode_sample(
        {
            "device": torch.device("cpu"),
            "tokenizer": None,
            "text_encoder": text_encoder,
            "vae": None,
            "vae_arch": None,
        },
        Sample(prompt="a dog barks", target={"visual": "clip.mp4"}),
        {"video": torch.zeros(3, 5, 32, 32, dtype=torch.uint8), "audio_sample_rate": 48000},
        None,
        Namespace(),
    )
    assert pair["latent"].keys() == {"visual", "audio"}
    assert pair["latent"]["audio"].shape[0] == pair["cond_kwargs"]["h3_packed_layout"]["audio_pos"].shape[0] == 18
    assert pair["cond_kwargs"]["h3_target_has_soundtrack"] is True
