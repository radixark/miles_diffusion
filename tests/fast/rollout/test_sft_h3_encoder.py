"""H3 encode contract: what validate_args rejects, what the engine geometry yields.

    frames 17n+5 --.                        .-- latent_t = (f-1)/16*4+... = 32 @107f
    canvas %32 ----+-- validate_args        +-- video rows = t*(H/32)*(W/32) = 32256
    stride == 1 --'                         '-- packed seq: [text|video|pad] aligned 64

Each test pins one side: rejection of off-grid inputs, or an engine-derived
geometry invariant the cached latents depend on.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

pytest.importorskip("sglang.multimodal_gen")

from miles.rollout.encoder_hub import h3


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
        h3.validate_args(_args())

    def test_rejects_off_grid_frame_count(self):
        with pytest.raises(ValueError, match="17n\\+5"):
            h3.validate_args(_args(diffusion_output_num_frames=96))

    def test_rejects_wrong_short_edge(self):
        with pytest.raises(ValueError, match="short_edge"):
            h3.validate_args(_args(diffusion_height=480, diffusion_width=832))


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
