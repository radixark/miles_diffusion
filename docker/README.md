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

_TBD._
