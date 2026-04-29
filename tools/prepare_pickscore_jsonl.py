#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "flow_grpo" / "dataset" / "pickscore" / "train.txt"
    dst = repo_root / "data" / "pickscore" / "train.jsonl"

    if not src.exists():
        raise FileNotFoundError(f"Missing PickScore prompts: {src}")

    # Convert plain text prompts into Miles JSONL format: {"input": "..."} per line.
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            prompt = line.strip()
            if not prompt:
                continue
            f_out.write(json.dumps({"input": prompt}, ensure_ascii=True) + "\n")

    print(f"Wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
