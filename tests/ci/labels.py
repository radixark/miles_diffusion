"""Canonical CI label registry.

Tests declare a domain label set in `register_cuda_ci(..., labels=[...])` and
`register_cpu_ci(..., labels=[...])`. The PR-side trigger for each label is
`run-ci-<key>`: each entry below MUST have a matching `run-ci-<key>` label in
the GitHub repo (maintainer-managed).

Adding a new label:
1) Add an entry below.
2) Create the matching `run-ci-<key>` label in GitHub repo Settings -> Labels.
   The workflow does not need editing -- the generic stage job filters tests
   by labels at runtime.

The meta-labels `run-ci-image` / `run-ci-all` are intentionally NOT listed
here: they bypass the per-test labels filter and run the full suite via the
`--match-all-labels` flag (handled in run_suite.py).
"""

KNOWN_LABELS: dict[str, str] = {
    "sglang-diffusion": "sglang_diffusion_utils engine / monkey patch tests",
    "fsdp": "FSDP backend + config tests",
    "rollout": "Rollout sampling / filter / strategy tests",
    "ray": "Ray actor / placement_group tests",
    "router": "Router routing decision tests",
    "arguments": "Top-level argparse / validate_args tests",
    "model-scripts": "train_diffusion.py + scripts/*.sh smoke tests",
}
