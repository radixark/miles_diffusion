"""Build a π₀.₅ LIBERO bundle and a lightweight miles JSONL index (no image/latent cache)."""

import argparse
import dataclasses
import json
import shutil
import tempfile
from pathlib import Path

from miles.backends.fsdp_utils.models.pi05.checkpoint import META_FILE, serving_config, sha256_file


def prepare(
    checkpoint, norm_stats, out, manifest, limit=None, episodes=None, provenance=None, dataset_source=None, source=None
):
    from openpi.shared import normalize
    from openpi.training import config

    from miles.backends.fsdp_utils.models.pi05.data import create_raw_dataset

    if out.exists() or manifest.exists():
        raise FileExistsError("Use new bundle and manifest paths")
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Convert the official JAX checkpoint to PyTorch first: {weights}")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    initialization = json.loads(provenance.read_text()) if provenance is not None else {"source_uri": source}
    if not initialization.get("source_uri"):
        raise ValueError("Declare the checkpoint origin with --source or --provenance")
    if source and source != initialization["source_uri"]:
        raise ValueError("--source differs from the provenance source")
    expected = initialization.get("converted_weights_sha256")
    if expected and sha256_file(weights) != expected:
        raise ValueError("Checkpoint weights do not match the provenance SHA256")
    train_config = config.get_config("pi05_libero")
    model = dataclasses.replace(train_config.model, dtype="float32", pytorch_compile_mode=None)
    data = train_config.data.create(train_config.assets_dirs, model)
    dataset_provenance = json.loads(dataset_source.read_text()) if dataset_source else None
    dataset = create_raw_dataset(model, data, episodes, (dataset_provenance or {}).get("revision"))
    count = len(dataset) if limit is None else min(limit, len(dataset))
    if not count:
        raise ValueError("LIBERO dataset is empty")
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".pi05-prepare-") as tmp:
        stage = Path(tmp) / "bundle"
        stats_dir = stage / "assets" / "physical-intelligence" / "libero"
        stats_dir.mkdir(parents=True)
        shutil.copyfile(norm_stats, stats_dir / "norm_stats.json")
        stats = normalize.load(stats_dir)
        from openpi import transforms

        transforms.Normalize(stats, use_quantiles=data.use_quantile_norm)  # validate quantile stats
        if not {"actions", "state"} <= stats.keys():
            raise ValueError("LIBERO normalization statistics must include actions and state")
        metadata = {
            "version": 1,
            "config_name": "pi05_libero",
            "model": dataclasses.asdict(model),
            "dataset_repo_id": data.repo_id,
            "dataset_episodes": episodes,
            "manifest_rows": count,
            "initialization": initialization,
            "normalization_sha256": sha256_file(norm_stats),
            "dataset_source": dataset_provenance,
        }
        (stage / META_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
        (stage / "config.json").write_text(json.dumps(serving_config(metadata["model"]), indent=2) + "\n")
        shutil.copyfile(weights, stage / "model.safetensors")
        # LeRobot obtains the real task prompt. JSONL's prompt is only a cursor entry.
        with manifest.open("x") as stream:
            for index in range(count):
                stream.write(json.dumps({"prompt": "LIBERO", "metadata": {"openpi_index": index}}) + "\n")
        stage.rename(out)
    print(f"Prepared {count} dataset indices in {manifest}; bundle: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--norm-stats", type=Path, required=True, help="LIBERO norm_stats.json computed from the training dataset"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Optional maximum number of training examples")
    parser.add_argument("--episodes", type=int, nargs="+", help="Only download/use these LeRobot episode indices")
    parser.add_argument(
        "--provenance", type=Path, help="Audited source/conversion JSON, including converted_weights_sha256"
    )
    parser.add_argument("--source", help="Original checkpoint URI; required unless --provenance supplies it")
    parser.add_argument("--dataset-source", type=Path, help="Dataset source/revision JSON")
    args = parser.parse_args()
    prepare(
        args.checkpoint,
        args.norm_stats,
        args.out,
        args.manifest,
        args.limit,
        args.episodes,
        args.provenance,
        args.dataset_source,
        args.source,
    )
