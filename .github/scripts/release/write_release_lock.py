"""Write release-lock.json for a release branch cut.

Ported from radixark/miles (.github/scripts/release/write_release_lock.py) with
miles_diffusion's inputs: no Megatron-LM, one CUDA 12.9 x86_64 image, and the
base image pinned by digest because `docker build --pull` re-resolves its tag.
"""

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-branch", required=True)
    parser.add_argument("--sglang-commit", required=True)
    parser.add_argument("--ci-image-tag", required=True)
    parser.add_argument("--ci-image-digest", required=True)
    parser.add_argument("--sglang-image-tag", required=True)
    parser.add_argument("--sglang-image-digest", required=True)
    parser.add_argument("--fa3-wheels-repo", required=True)
    parser.add_argument("--fa3-wheels-tag", required=True)
    parser.add_argument("--fa3-wheels-fingerprint", required=True)
    args = parser.parse_args()

    assert len(args.sglang_commit) == 40, "failed to resolve the sglang commit"
    assert args.ci_image_digest.startswith("sha256:"), "failed to resolve ci_image_digest"
    assert args.sglang_image_digest.startswith("sha256:"), "failed to resolve sglang_image_digest"
    assert args.fa3_wheels_fingerprint, "failed to fingerprint the FA3 wheels release"

    # Restored inputs: release CI re-installs sglang at sglang_commit and runs
    # inside ci_image_tag; release-docker.yml bakes sglang_commit and builds
    # FROM sglang_image_tag@sglang_image_digest. ci_image_digest and the
    # fa3_wheels_* fields are audit records: a rolling wheels release can have
    # its assets replaced in place, so it cannot be pinned by name.
    lock = {
        "release_branch": args.release_branch,
        "sglang_repo": "sgl-project/sglang",
        "sglang_branch": "sglang-miles",
        "sglang_commit": args.sglang_commit,
        "ci_image_tag": args.ci_image_tag,
        "ci_image_digest": args.ci_image_digest,
        "sglang_image_tag": args.sglang_image_tag,
        "sglang_image_digest": args.sglang_image_digest,
        "fa3_wheels_repo": args.fa3_wheels_repo,
        "fa3_wheels_tag": args.fa3_wheels_tag,
        "fa3_wheels_fingerprint": args.fa3_wheels_fingerprint,
    }
    json.dump(lock, open("release-lock.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
