"""Load the native π₀.₅ model using OpenPI-compatible weight names."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan

from .bundle import read_bundle


def tie_openpi_embeddings(model):
    """Preserve PaliGemma aliases even inside Accelerate's meta initialization."""
    paligemma = model.paligemma_with_expert.paligemma
    embedding = paligemma.model.language_model.embed_tokens.weight
    # init_empty_weights replaces register_parameter and creates a new Parameter
    # on assignment. Assign the existing parameter in the registry directly so
    # rank 0 and meta ranks enumerate the same parameters before FSDP sharding.
    paligemma.lm_head._parameters["weight"] = embedding


class Pi05ModelBackend:
    """Implements the FSDP actor backend protocol with lazy optional dependencies."""

    def __init__(self, train_pipeline_config):
        self.config = train_pipeline_config

    def load_component(self, component, *, checkpoint_path, master_dtype, materialize_weights):
        if component != "policy":
            raise ValueError("π₀.₅ exposes one trainable component: policy")
        if read_bundle(checkpoint_path).get("validation_profile"):
            raise ValueError("Local smoke-test bundles are not production π₀.₅ checkpoints")
        from safetensors.torch import load_model

        from .model import Pi05Policy

        config = SimpleNamespace(**read_bundle(checkpoint_path)["model"])
        model = Pi05Policy(config).to(dtype=master_dtype)
        tie_openpi_embeddings(model)
        # Other ranks are inside init_empty_weights; do not read the multi-GB weights there.
        if materialize_weights:
            load_model(model, str(Path(checkpoint_path) / "model.safetensors"), strict=True)
        return model

    def enable_gradient_checkpointing(self, model):
        model.gradient_checkpointing_enable()

    def fsdp_parallel_plan(self, model):
        # OpenPI jointly computes prefix/suffix attention by calling layer children,
        # bypassing GemmaDecoderLayer.forward. Wrapping those decoder blocks leaves
        # sharded parameters in the forward. Wrap only children invoked via __call__.
        # Embeddings and tied LM heads stay together in the root FSDP group.
        return FSDPParallelPlan(no_split_modules=("SiglipEncoderLayer", "GemmaMLP", "Pi05RMSNorm"))

    def set_attention_backend(self, model, backend):
        if backend != "eager":
            raise ValueError("π₀.₅ joint training attention supports only eager attention here")

    def enable_deterministic_attention(self, backend):
        if backend not in (None, "eager"):
            raise ValueError("π₀.₅ uses official eager attention")

    def sequence_parallel_plan(self, model):
        raise NotImplementedError("π₀.₅ training currently supports FSDP/DP with sequence-parallel-size=1")

    def install_sequence_parallel_attention(self, model, parallel_state):
        raise NotImplementedError("π₀.₅ sequence parallel training is not implemented")
