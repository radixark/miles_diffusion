# Miles-Diffusion Docker

> ⚠️ This image is still **experimental**. A stable version is on the way — stay tuned.

## Build locally (no push)

```bash
docker build -f docker/Dockerfile -t radixark/miles_diffusion:local .
```

Builds the image locally without pushing, from the repo root (the same command CI runs in
`docker-build.yml`).

## PR build check (in `pr-test.yml`)

A PR touching `docker/Dockerfile` or `requirements.txt` builds and pushes
`radixark/miles_diffusion:pr-<num>` (`docker-paths` then `docker-build`), and every GPU
suite then runs inside it instead of `latest`; a failed build stops the matrix.
The fresh build outranks a `ci-image-tag:` PR-body directive.
`docker-pr-tag-cleanup.yml` deletes the tag when the PR closes. Fork PRs skip the
build and stay on `latest`.

## Release rule

Versioned images are published only by `release-docker.yml`, from an annotated `v<version>`
tag that `release-tag.yml` created after release CI passed on that exact commit:

- `radixark/miles_diffusion:v<version>` (`v0.2.0`, `v0.2.0rc1`, `v0.2.0.post1`): built from the
  tagged commit with `release-lock.json`'s inputs — the `sglang-miles` commit as
  `SGLANG_DIFFUSION_COMMIT`, the base image as `SGLANG_IMAGE_TAG=<tag>@<digest>`, the tagged
  commit as `MILES_DIFFUSION_COMMIT`. Never moves, never rebuilt in place; a shipped bug gets a
  new `.postN` tag.
- `radixark/miles_diffusion:release-v<X.Y.Z>-ci`: the dated dev image `release-branch-cut.yml`
  froze (by digest) for that release branch's CI. Release CI always runs inside it.
- `latest` and `dev-cu129-sglang-main-<YYYYMMDD>` are rolling and are not touched by a release.

A fix that changes `docker/Dockerfile` or `requirements.txt` cannot be hotfixed onto a release
branch (`bot-cherry-pick.yml` refuses it): it needs a new base version and a fresh cut. The
maintainer procedure is in [Release a Version](../docs/developer/release.md).
