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

    json.dump(
        {
            "release_branch": args.release_branch,
            # Restored inputs: CI re-installs sglang at this commit and
            # release-docker.yml bakes it.
            "sglang_repo": "sgl-project/sglang",
            "sglang_branch": "sglang-miles",
            "sglang_commit": args.sglang_commit,
            # Restored input for release CI; the digest records what the tag
            # pointed at when it was frozen.
            "ci_image_tag": args.ci_image_tag,
            "ci_image_digest": args.ci_image_digest,
            # Restored input for release-docker.yml: FROM <tag>@<digest>.
            "sglang_image_tag": args.sglang_image_tag,
            "sglang_image_digest": args.sglang_image_digest,
            # Audit only: a rolling wheels release can have assets replaced in
            # place, so it cannot be pinned by name.
            "fa3_wheels_repo": args.fa3_wheels_repo,
            "fa3_wheels_tag": args.fa3_wheels_tag,
            "fa3_wheels_fingerprint": args.fa3_wheels_fingerprint,
        },
        open("release-lock.json", "w"),
        indent=2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
