from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.backends.fsdp_utils.model_backend import MilesModelBackend
from miles.backends.fsdp_utils.models.ltx.positions import prepare_video_positions


class _LTXConfig:
    model_package = "miles.backends.fsdp_utils.models.ltx"


class TestLTXPackage:
    def test_component_loading_delegates_to_loading_module(self, monkeypatch):
        backend = MilesModelBackend(_LTXConfig())
        sentinel = object()
        seen = {}

        def fake_load_component(component, args, *, master_dtype, materialize_weights):
            seen.update(
                component=component,
                args=args,
                master_dtype=master_dtype,
                materialize_weights=materialize_weights,
            )
            return sentinel

        monkeypatch.setattr(backend._pkg.loading, "load_component", fake_load_component)
        args = object()
        result = backend.load_component(
            "transformer",
            args,
            master_dtype=torch.float32,
            materialize_weights=True,
        )

        assert result is sentinel
        assert seen == {
            "component": "transformer",
            "args": args,
            "master_dtype": torch.float32,
            "materialize_weights": True,
        }
        assert not hasattr(backend._pkg.modeling, "load_component")


class TestLTXAttentionBackend:
    # LTX maps --fsdp-attention-backend to ltx_core's AttentionFunction instead of the
    # diffusers set_attention_backend(str) method its native transformers don't have.
    def test_unknown_backend_raises(self):
        pytest.importorskip("ltx_core")
        with pytest.raises(ValueError, match="not an ltx_core backend"):
            MilesModelBackend(_LTXConfig()).set_attention_backend(torch.nn.Linear(2, 2), "bogus")

    def test_alias_noops_without_attention_submodule(self):
        pytest.importorskip("ltx_core")
        # "sdpa"/"native" alias -> PYTORCH; a model with no Attention submodule is a no-op
        MilesModelBackend(_LTXConfig()).set_attention_backend(torch.nn.Linear(2, 2), "sdpa")  # no raise

    def test_swaps_callable_on_attention_submodule(self):
        pytest.importorskip("ltx_core")
        from ltx_core.model.transformer.attention import Attention

        model = torch.nn.Sequential(Attention(query_dim=16, heads=1, dim_head=8))
        MilesModelBackend(_LTXConfig()).set_attention_backend(model, "sdpa_math")
        # SDPA_MATH resolves to a PytorchAttention callable on both paths
        assert type(model[0].attention_function).__name__ == "PytorchAttention"
        assert type(model[0].masked_attention_function).__name__ == "PytorchAttention"


class TestLtxVideoPositions:
    # Positions are spatiotemporal patch coordinates consumed by LTX RoPE, not
    # positional embeddings. Their token count must match the rollout latents.
    def test_position_shape_and_token_guard(self):
        # 512x512, 25 frames -> latent 16x16 spatial, (25-1)//8+1 = 4 frames => 4*16*16 = 1024 tokens
        num_tokens = 4 * 16 * 16
        positions = prepare_video_positions(
            batch_size=2,
            num_tokens=num_tokens,
            height=512,
            width=512,
            num_frames=25,
            fps=24.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert positions.shape == (2, 3, num_tokens, 2)

    def test_token_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="token count mismatch"):
            prepare_video_positions(
                batch_size=1,
                num_tokens=999,
                height=512,
                width=512,
                num_frames=25,
                fps=24.0,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
