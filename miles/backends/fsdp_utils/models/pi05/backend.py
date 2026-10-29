"""Load the official PI0Pytorch module without copying its architecture or training objective."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan

from .checkpoint import read_bundle


def openpi_model_config(bundle):
    from openpi.models.pi0_config import Pi0Config

    return Pi0Config(**read_bundle(bundle)["model"])


def check_openpi_runtime():
    try:
        import transformers
        from transformers.models.siglip import check
    except ImportError as exc:
        raise RuntimeError(
            "π₀.₅ training needs OpenPI's patched Transformers 4.53.2; see docs/models/pi05.md"
        ) from exc
    if transformers.__version__ != "4.53.2" or not check.check_whether_transformers_replace_is_installed_correctly():
        raise RuntimeError("π₀.₅ training needs OpenPI's patched Transformers 4.53.2; see docs/models/pi05.md")


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
        check_openpi_runtime()
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from safetensors.torch import load_model

        config = dataclasses.replace(openpi_model_config(checkpoint_path), dtype="float32", pytorch_compile_mode=None)
        model = PI0Pytorch(config).to(dtype=master_dtype)
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
        return FSDPParallelPlan(no_split_modules=("SiglipEncoderLayer", "GemmaMLP", "GemmaRMSNorm"))

    def set_attention_backend(self, model, backend):
        if backend != "eager":
            raise ValueError("OpenPI's joint training attention supports only eager attention here")

    def enable_deterministic_attention(self, backend):
        if backend not in (None, "eager"):
            raise ValueError("π₀.₅ uses official eager attention")

    def sequence_parallel_plan(self, model):
        raise NotImplementedError("π₀.₅ training currently supports FSDP/DP with sequence-parallel-size=1")

    def install_sequence_parallel_attention(self, model, parallel_state):
        raise NotImplementedError("π₀.₅ sequence parallel training is not implemented")
