"""Convert official pi05_base using OpenPI's tensor mapping and Miles' native model.

The upstream conversion helpers only reshape/rename JAX tensors. Materialization
uses Miles, so neither training nor conversion needs patched Transformers.
"""

import argparse
import dataclasses
import importlib.util
import json
import subprocess
from pathlib import Path

from .provenance import sha256_file

OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
SOURCE = "gs://openpi-assets/checkpoints/pi05_base"


def convert(args):
    import numpy as np
    import torch
    from openpi.models.gemma import get_config
    from openpi.shared import download
    from openpi.training.config import get_config as train_config
    from safetensors.torch import save_model

    from miles.backends.fsdp_utils.models.pi05.loading import tie_openpi_embeddings
    from miles.backends.fsdp_utils.models.pi05.model import Pi05Policy, paligemma_config

    if args.output.exists():
        raise FileExistsError(args.output)
    revision = subprocess.check_output(
        ["git", "-C", str(args.openpi_checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != OPENPI_REVISION:
        raise ValueError(f"Expected OpenPI {OPENPI_REVISION}, got {revision}")
    source = args.openpi_checkout / "examples/convert_jax_model_to_pytorch.py"
    spec = importlib.util.spec_from_file_location("openpi_tensor_converter", source)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    print(f"Loading official base from {args.checkpoint or SOURCE}", flush=True)
    checkpoint = args.checkpoint or download.maybe_download(SOURCE, gs={"token": "anon"})
    config = dataclasses.replace(train_config("pi05_libero").model, dtype="float32", pytorch_compile_mode=None)
    print("Converting JAX tensors to the native Miles model", flush=True)
    initial = converter.slice_initial_orbax_checkpoint(str(checkpoint.resolve()), restore_precision="float32")
    projections = {}
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        params = initial["projection_params"][name]
        for src, dst in (("kernel", "weight"), ("bias", "bias")):
            value = params[src]
            if isinstance(value, dict):
                value = value["value"]
            tensor = torch.from_numpy(np.asarray(value))
            projections[f"{name}.{dst}"] = tensor.T if src == "kernel" else tensor
    prefix, expert = converter.slice_paligemma_state_dict(
        initial["paligemma_params"], paligemma_config(config.paligemma_variant)
    )
    expert = converter.slice_gemma_state_dict(
        expert, get_config(config.action_expert_variant), num_expert=1, checkpoint_dir=str(checkpoint), pi05=True
    )
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = Pi05Policy(config).float()
    tie_openpi_embeddings(model)
    loaded = model.load_state_dict({**prefix, **expert, **projections}, strict=False)
    allowed = {
        "paligemma_with_expert.paligemma.lm_head.weight",  # tied to loaded prefix embedding
        "paligemma_with_expert.gemma_expert.lm_head.weight",  # unused by action prediction
    }
    if loaded.unexpected_keys or set(loaded.missing_keys) - allowed:
        raise ValueError(f"Incomplete base conversion: {loaded}")
    args.output.mkdir(parents=True)
    weights = args.output / "model.safetensors"
    save_model(model, str(weights))
    provenance = {
        "kind": "official_jax_tensor_mapping_native_materialization",
        "source_uri": SOURCE,
        "local_checkpoint": str(checkpoint.resolve()),
        "openpi_revision": revision,
        "converter_sha256": sha256_file(source),
        "converted_weights_sha256": sha256_file(weights),
        "downstream_model_config": dataclasses.asdict(config),
        "load_audit": {"missing_keys": loaded.missing_keys, "unexpected_keys": loaded.unexpected_keys},
        "unused_lm_head_seed": 0,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Converted {SOURCE} to {weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Previously downloaded official pi05_base JAX directory")
    convert(parser.parse_args())
