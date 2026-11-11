"""Pick the dated CI image a release branch cut freezes, and prove it matches a build.

Usage: pick_ci_image.py <docker-build-run.json> <docker-hub-tags.json>

<docker-build-run.json> is the newest successful docker-build.yml run:
    gh run list --workflow docker-build.yml --status success --limit 1 \
        --json headSha,headBranch,createdAt,updatedAt
<docker-hub-tags.json> is the Docker Hub tag listing of radixark/miles_diffusion.

Prints `key=value` lines for the workflow: dev_image_tag, dev_image_digest,
image_build_sha. Fails when the newest dated tag was not pushed by that run
(it must be a main build, and the tag's push time must fall inside the run),
because the cut then cannot say which Dockerfile the frozen image represents.
"""

import json
import re
import sys
from datetime import datetime

DATED_TAG_RE = re.compile(r"dev-cu129-sglang-main-\d{8}")


def parse_time(value: str) -> datetime:
    # Docker Hub carries fractional seconds, gh does not; fromisoformat takes both.
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> int:
    run_path, tags_path = sys.argv[1:]
    runs = json.load(open(run_path))
    tags = json.load(open(tags_path))["results"]

    dated_tags = sorted((t for t in tags if DATED_TAG_RE.fullmatch(t["name"])), key=lambda t: t["name"])
    if not dated_tags:
        print(
            "no dev-cu129-sglang-main-<YYYYMMDD> tag on Docker Hub; dispatch docker-build.yml first", file=sys.stderr
        )
        return 1
    newest_tag = dated_tags[-1]

    if not runs:
        print("no successful docker-build.yml run found", file=sys.stderr)
        return 1
    run = runs[0]
    if run["headBranch"] != "main":
        print(f"newest successful docker-build.yml run is on {run['headBranch']!r}, not main", file=sys.stderr)
        return 1
    pushed_at = parse_time(newest_tag["last_updated"])
    if not parse_time(run["createdAt"]) <= pushed_at <= parse_time(run["updatedAt"]):
        print(
            f"{newest_tag['name']} (pushed {newest_tag['last_updated']}) was not pushed by the newest successful "
            f"docker-build.yml run ({run['createdAt']} .. {run['updatedAt']}); rebuild the image on main first",
            file=sys.stderr,
        )
        return 1

    # The workflow sources these lines: accept only the exact digest / SHA shapes.
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", newest_tag["digest"] or ""):
        print(f"unexpected digest for {newest_tag['name']}: {newest_tag['digest']!r}", file=sys.stderr)
        return 1
    if not re.fullmatch(r"[0-9a-f]{40}", run["headSha"]):
        print(f"unexpected headSha: {run['headSha']!r}", file=sys.stderr)
        return 1
    print(f"dev_image_tag={newest_tag['name']}")
    print(f"dev_image_digest={newest_tag['digest']}")
    print(f"image_build_sha={run['headSha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
