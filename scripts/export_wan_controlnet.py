"""Export a Miles Wan ControlNet DCP checkpoint without loading backbone weights.

Example:
    python scripts/export_wan_controlnet.py --ckpt-dir RUN/iter_0000020 \
        --base-checkpoint /models/Wan2.2-TI2V-5B-Diffusers \
        --num-control-blocks 4 --out RUN/controlnet
"""

import argparse

from miles.backends.fsdp_utils.models.wan_controlnet.export import export_controlnet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--num-control-blocks", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = export_controlnet(
        args.ckpt_dir,
        args.out,
        base_checkpoint=args.base_checkpoint,
        num_control_blocks=args.num_control_blocks,
    )
    print(f"Exported ControlNet only: {out}")


if __name__ == "__main__":
    main()
