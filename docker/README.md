# Miles-Diffusion Docker

> ⚠️ This image is still **experimental**. A stable version is on the way — stay tuned.

## Build locally (no push)

```bash
just build-local
```

Builds `radixark/miles-diffusion:<version>-local` locally without pushing.
(`<version>` is read from `docker/version.txt`.)

## PR build check (in `pr-test.yml`)

A PR touching `docker/Dockerfile` or `requirements.txt` builds and pushes
`radixark/miles_diffusion:pr-<num>` (`docker-paths` then `docker-build`), and every GPU
suite then runs inside it instead of `latest`; a failed build stops the matrix.
The fresh build outranks a `ci-image-tag:` PR-body directive.
`docker-pr-tag-cleanup.yml` deletes the tag when the PR closes. Fork PRs skip the
build and stay on `latest`.

## Release rule

_TBD._
