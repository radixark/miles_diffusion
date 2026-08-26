"""Export a miles LoRA checkpoint (torch.distributed.checkpoint) to safetensors.

Thin CLI over :func:`miles.backends.fsdp_utils.lora_export.export_lora_safetensors`.

Usage:
    python3 scripts/export_lora.py --ckpt-dir <run>/ckpt/iter_0000070 \\
        --out practical_dynamics_fx.safetensors --lora-rank 64 --lora-alpha 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

from miles.backends.fsdp_utils.lora_export import export_lora_safetensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True, help="iter_XXXXXXX checkpoint directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    args = parser.parse_args()
    export_lora_safetensors(args.ckpt_dir, args.out, args.lora_rank, args.lora_alpha)


if __name__ == "__main__":
    main()
