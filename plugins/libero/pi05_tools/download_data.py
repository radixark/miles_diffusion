"""Download the pinned LeRobot LIBERO demonstrations and record their source."""

import argparse
import json
from pathlib import Path

REPO_ID = "physical-intelligence/libero"
REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"


def download(output):
    from huggingface_hub import snapshot_download

    # The loader resolves this exact directory via HF_LEROBOT_HOME.
    from lerobot.common.constants import HF_LEROBOT_HOME

    root = Path(HF_LEROBOT_HOME) / REPO_ID
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=REVISION, local_dir=root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"repo_id": REPO_ID, "revision": REVISION}, indent=2) + "\n")
    print(f"Downloaded {REPO_ID}@{REVISION} to {root}; source: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    download(parser.parse_args().source_output)
