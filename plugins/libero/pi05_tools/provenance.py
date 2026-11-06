"""Base-source checks for the LIBERO training recipe."""

import hashlib
from pathlib import Path

from miles.backends.fsdp_utils.models.pi05.bundle import read_bundle


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_base_initialization(bundle):
    """Check the declared base source and verify a weight checksum when supplied."""
    metadata = read_bundle(bundle)
    source = metadata.get("initialization") or {}
    if source.get("source_uri") != "gs://openpi-assets/checkpoints/pi05_base":
        raise ValueError("This run requires the official pi05_base initializer, not a LIBERO-finetuned checkpoint")
    if metadata.get("training_checkpoint"):
        raise ValueError("Use the original base bundle with --load for resume, not an exported trained bundle")
    expected = source.get("converted_weights_sha256")
    if expected and sha256_file(Path(bundle) / "model.safetensors") != expected:
        raise ValueError("Base bundle weights do not match the recorded conversion SHA256")
    return source
