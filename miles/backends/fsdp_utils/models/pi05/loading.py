"""Load the native π₀.₅ model using OpenPI-compatible weight names."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from .bundle import read_bundle


def tie_openpi_embeddings(model):
    """Preserve PaliGemma aliases even inside Accelerate's meta initialization."""
    paligemma = model.paligemma_with_expert.paligemma
    embedding = paligemma.model.language_model.embed_tokens.weight
    # init_empty_weights replaces register_parameter and creates a new Parameter
    # on assignment. Assign the existing parameter in the registry directly so
    # rank 0 and meta ranks enumerate the same parameters before FSDP sharding.
    paligemma.lm_head._parameters["weight"] = embedding


def load_component(component, *, checkpoint_path, master_dtype, materialize_weights):
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
