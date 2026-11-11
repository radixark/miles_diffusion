---
title: Release a Version
description: Cut a versioned miles-diffusion release branch, run release CI inside a frozen image, tag an exact release, and publish the official Docker image.
---
This runbook is for maintainers publishing an official miles-diffusion version from
`radixark/miles_diffusion`. It mirrors the miles procedure
([Release a Version](https://github.com/radixark/miles/blob/main/docs/developer/ci/04-release.md))
with this repository's inputs: one source dependency (sglang's `sglang-miles` branch), one CUDA
12.9 x86_64 image, and the CI suites from [Contributing](contributor-guide.md).

## What a version freezes

| Fact | Authoritative source | Consumed as |
|---|---|---|
| Base version `X.Y.Z` | `setup.py` | release branch `release/vX.Y.Z`; checked before tagging |
| Exact version | `version` input of `release-tag.yml` | annotated git tag `v<exact>`, image tag `radixark/miles_diffusion:v<exact>` |
| sglang commit | `release-lock.json` → `sglang_commit` | CI re-installs sglang at it; `release-docker.yml` bakes it (`SGLANG_DIFFUSION_COMMIT`) |
| CI image | `release-lock.json` → `ci_image_tag` (+ `ci_image_digest`) | every GPU suite of release CI runs inside `release-vX.Y.Z-ci` |
| Base image | `release-lock.json` → `sglang_image_tag` + `sglang_image_digest` | `release-docker.yml` builds `FROM lmsysorg/sglang:<tag>@<digest>` |
| FA3 wheels release | `release-lock.json` → `fa3_wheels_*` | audit record only: the rolling release cannot be pinned by name |
| miles-diffusion code | the tagged commit | `MILES_DIFFUSION_COMMIT` of the release build |

`release-lock.json` is written once, by the first dispatch of `release-branch-cut.yml`, and
committed on the release branch. `pr-test.yml` reads it whenever no explicit input or PR-body
directive names a value, so a plain dispatch on the branch gets the locked sglang ref and the
frozen image too. `record-e2e-standards.yml` passes no image and gets the sglang pin only; do
not record standards on a release branch (they are re-recorded on `main` through the fix's PR).

The exact version may be `X.Y.Z`, `X.Y.ZrcN` or `X.Y.Z.postN`; the branch and `setup.py` always
carry the base version. `v0.2.0rc1`, `v0.2.0` and `v0.2.0.post1` all come from `release/v0.2.0`
and share its lockfile.

What a release does **not** freeze: `apt` packages, pip's transitive resolution, the `kernels`
lockfile the Dockerfile generates at build time, the PaddleOCR model download, and the CPU half
of release CI (`ubuntu-latest`, runtime installs). Two builds of the same tag are the same
sources, not byte-identical images.

## Before you start

- Authenticate `gh` to `radixark/miles_diffusion` with permission to run workflows, push release
  refs, and open pull requests. GitHub Actions in this repository cannot open pull requests, so
  the bump PR is yours.
- Choose the base version and the exact version:

```bash
BASE_VERSION=X.Y.Z
EXACT_VERSION=X.Y.Z        # or X.Y.ZrcN / X.Y.Z.postN
RELEASE_BRANCH="release/v${BASE_VERSION}"
```

- Every release workflow is dispatched on `main`: the workflow files that run are `main`'s, the
  code they test and build is the branch's or the tag's. A later change to the CI harness on
  `main` therefore changes what a `.postN` re-dispatch runs.
- `release/*` branches take only `bot-cherry-pick.yml` pushes. Do not open pull requests
  against a release branch; nothing protects `release/*` today, so that is a rule, not a guard.
- `force=true` on the tag workflow is an audited emergency bypass, never the normal path.

## 1. Merge the base-version bump

Skip this step if `setup.py` on `main` already carries `BASE_VERSION`. Otherwise:

```bash
git checkout -b "chore/bump-version-${BASE_VERSION}" origin/main
python .github/scripts/release/bump_version.py "${BASE_VERSION}"
git commit -am "chore(release): bump version to ${BASE_VERSION}"
git push -u origin HEAD
gh pr create --base main --title "chore(release): bump version to ${BASE_VERSION}" \
  --body "Sets setup.py to the base version for the ${RELEASE_BRANCH} release."
```

Pass only `X.Y.Z` here; `rcN` / `.postN` belong to `EXACT_VERSION` at tag time. Normal PR CI
runs; merge it. Do not cut until `main` contains the base version.

## 2. Preflight the image, cut the branch, run release CI

The cut freezes the newest `radixark/miles_diffusion:dev-cu129-sglang-main-<YYYYMMDD>` image
and refuses unless that tag was pushed by the newest **successful** `docker-build.yml` run on
`main` whose `docker/Dockerfile` and `requirements.txt` equal the cut point's. Check before
dispatching:

```bash
gh run list --workflow docker-build.yml --status success --limit 1 \
  --json headSha,headBranch,createdAt,updatedAt,url
git log --oneline -1 origin/main -- docker/Dockerfile requirements.txt
```

If the last image-layer commit is newer than that run's `headSha` (a push build was cancelled,
or a change merged since), build first and wait for success:

```bash
gh workflow run docker-build.yml --ref main
```

Then cut from the `main` tip (or pass `-f commit_sha=FULL_MAIN_SHA` for an older commit with
the same image layers):

```bash
gh workflow run release-branch-cut.yml -f branch_name="${RELEASE_BRANCH}"
```

The first dispatch creates `release/vX.Y.Z`, retags the preflighted image by digest as
`release-vX.Y.Z-ci`, records the sglang commit, the base-image digest and the FA3 fingerprint
in `release-lock.json`, commits the lockfile, and calls `pr-test.yml` on that commit with every
suite in full scope (`--match-all-labels --nightly --continue-on-error`) inside the frozen
image. A finalizer job stamps a `release-ci` commit status on the tested SHA. Re-dispatching on
an existing branch keeps the lockfile and the image and tests the current tip; it cancels a
still-running cut of the same branch.

After the run is green, resolve the branch tip and keep it as `RELEASE_SHA`:

```bash
git ls-remote https://github.com/radixark/miles_diffusion.git "refs/heads/${RELEASE_BRANCH}"
RELEASE_SHA=FULL_RELEASE_SHA
gh api "repos/radixark/miles_diffusion/commits/${RELEASE_SHA}/status" \
  --jq '[.statuses[] | select(.context == "release-ci")][0].state'
```

The last command must print `success`. The suites checked out this exact SHA (not the branch
name), so a push to the branch during the run does not change what the status vouches for; it
does mean the tip is no longer the tested commit.

## 3. Apply a release hotfix, if needed

Fixes land on `main` first. Cherry-pick exactly one merged PR or one commit reachable from
`main`:

```bash
gh workflow run bot-cherry-pick.yml -f pr_number=PR_NUMBER -f target_branch="${RELEASE_BRANCH}"
# Or exactly one commit:
gh workflow run bot-cherry-pick.yml -f commit_sha=FULL_MAIN_SHA -f target_branch="${RELEASE_BRANCH}"
```

The workflow rejects commits that touch `docker/Dockerfile` or `requirements.txt`: release CI
keeps running inside the frozen image, and the GPU suites use the image's installed
requirements, so neither change can be represented without a rebuild. Such a fix needs a new
base version and a fresh cut. A fix that re-records an e2e standard carries the standard in the
same `main` commit, so the pick brings it along.

After every successful cherry-pick, repeat step 2 on the branch and replace `RELEASE_SHA` with
the newly verified tip. An older green status never authorizes the new commit.

## 4. Tag the verified SHA

Re-read the remote tip immediately before dispatch and require it to equal the verified SHA:

```bash
REMOTE_SHA=$(git ls-remote https://github.com/radixark/miles_diffusion.git "refs/heads/${RELEASE_BRANCH}" | cut -f1)
[ "${REMOTE_SHA}" = "${RELEASE_SHA}" ] || { echo "release branch moved; rerun release CI on the new tip" >&2; exit 1; }
gh workflow run release-tag.yml -f version="${EXACT_VERSION}" -f ref="${RELEASE_SHA}"
```

The workflow checks the version grammar, verifies that `setup.py` contains the base version,
requires `release-ci=success` on the checked-out SHA, creates the annotated tag
`v${EXACT_VERSION}`, and dispatches `release-docker.yml`. It does not compare the branch tip;
that is the operator check above.

## 5. Verify the published image

`release-docker.yml` checks out `v${EXACT_VERSION}`, reads its `release-lock.json`, and builds
`radixark/miles_diffusion:v${EXACT_VERSION}` on the self-hosted runner with
`SGLANG_IMAGE_TAG=<tag>@<digest>`, `SGLANG_DIFFUSION_COMMIT=<locked>` and
`MILES_DIFFUSION_COMMIT=<tagged commit>`. It never moves `latest`.

```bash
docker buildx imagetools inspect "radixark/miles_diffusion:v${EXACT_VERSION}"
docker run --rm "radixark/miles_diffusion:v${EXACT_VERSION}" \
  bash -c 'git -C /root/miles_diffusion rev-parse HEAD; git -C /sgl-workspace/sglang rev-parse HEAD'
```

Both SHAs must match the tag and the lockfile. To retry a failed build, keep the two inputs
paired to the same tag:

```bash
gh workflow run release-docker.yml -f version="${EXACT_VERSION}" -f ref="v${EXACT_VERSION}"
```

Never alter a published tag; a shipped bug gets a new exact version (`.postN`) through steps
3–5.

## 6. Publish the release notes

There is no notes automation. Draft from the commit range, review, then publish:

```bash
PREV_TAG=$(git describe --tags --abbrev=0 "v${EXACT_VERSION}^" 2>/dev/null || true)
git log --oneline "${PREV_TAG:+${PREV_TAG}..}v${EXACT_VERSION}" > /tmp/notes-input.txt
gh release create "v${EXACT_VERSION}" --draft --title "v${EXACT_VERSION}" --notes-file NOTES.md
```

For the first versioned release there is no previous tag; choose the baseline commit by hand.
Mark `rcN` tags as pre-releases (`--prerelease`).

## Workflow map

| Workflow | Trigger | Does |
|---|---|---|
| `release-branch-cut.yml` | dispatch on `main` | branch + image preflight + lockfile + retag + full CI + `release-ci` status |
| `bot-cherry-pick.yml` | dispatch on `main` | one `main` PR/commit onto `release/vX.Y.Z`, image-layer guard |
| `release-tag.yml` | dispatch on `main` | grammar + `setup.py` + `release-ci` gate → tag `v<exact>` → dispatch the image build |
| `release-docker.yml` | dispatched by the tag workflow, or a hand-pushed `v*` tag, or a retry | `radixark/miles_diffusion:v<exact>` from the tag and its lockfile |
| `pr-test.yml` (`workflow_call`) | called by the cut | the normal suites on an exact commit inside the frozen image |
