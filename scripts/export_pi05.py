"""Export a miles full-model DCP checkpoint into an OpenPI/SGLang π₀.₅ bundle."""

import argparse
from pathlib import Path

from miles.backends.fsdp_utils.models.pi05.checkpoint import export_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True, help="iter_XXXXXXX checkpoint directory")
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(export_checkpoint(args.ckpt_dir, args.base_bundle, args.out))
