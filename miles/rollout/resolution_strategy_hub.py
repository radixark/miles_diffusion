"""Library of functions that pick the output resolution of one sglang-diffusion
rollout prompt group.

Each function has signature ``(args, sample, seed) -> (height, width)``, in the
style of ``miles.rollout.step_strategy_hub``. The rollout calls it once per
prompt group, so every request of that group -- every microgroup -- renders at
one resolution; see ``generate_and_rm_group``. Point
``--diffusion-resolution-strategy-path`` at any such function; unset, the
request keeps ``--diffusion-height`` / ``--diffusion-width``.

The strategy receives the group's first ``Sample``, so a strategy can read the
resolution off the prompt or its dataset metadata instead of drawing one.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np

from miles.utils.types import Sample

__all__ = ["weighted_buckets", "parse_resolution_buckets", "parse_resolution_weights"]

# A latent token covers 16 px: qwen_image's VAE downsamples 8x and the DiT packs 2x2
# latent patches into one token. sgl-d validates a t2i height / width as positive and
# nothing else, then floors to this grid in latent space (``prepare_latent_shape``), so a
# 1000px request silently renders 992px and the response does not say so. This is the only
# place that rejection happens. TODO: let TrainPipelineConfig declare the granularity when a
# second family opts into mixed resolution -- sgl-d requires 32 for qwen_image21.
RESOLUTION_GRANULARITY = 16


def parse_resolution_buckets(raw: str | None) -> list[tuple[int, int]]:
    """Parse ``--diffusion-resolution-buckets`` ('512x512,672x384') to ``[(height, width)]``.

    Height first, like ``--diffusion-height`` / ``--diffusion-width``.
    """
    if raw is None or not str(raw).strip():
        raise ValueError("--diffusion-resolution-buckets is required: 'HxW[,HxW...]', e.g. '512x512,672x384'")
    buckets: list[tuple[int, int]] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        dims = token.lower().split("x")
        if len(dims) != 2:
            raise ValueError(f"--diffusion-resolution-buckets entry must be 'HxW', got {token!r}")
        try:
            height, width = (int(dim) for dim in dims)
        except ValueError:
            raise ValueError(f"--diffusion-resolution-buckets entry must be 'HxW' integers, got {token!r}") from None
        for name, value in (("height", height), ("width", width)):
            if value <= 0 or value % RESOLUTION_GRANULARITY != 0:
                raise ValueError(
                    f"--diffusion-resolution-buckets {token!r} has {name} {value}; "
                    f"each side must be a positive multiple of {RESOLUTION_GRANULARITY}"
                )
        buckets.append((height, width))
    if not buckets:
        raise ValueError(f"--diffusion-resolution-buckets {raw!r} lists no bucket")
    if len(set(buckets)) != len(buckets):
        raise ValueError(
            f"--diffusion-resolution-buckets {raw!r} repeats a bucket; use --diffusion-resolution-weights"
        )
    return buckets


def parse_resolution_weights(raw: str | None, num_buckets: int) -> list[float] | None:
    """Parse ``--diffusion-resolution-weights`` ('2,1,1') to probabilities; None when unset (uniform)."""
    if raw is None or not str(raw).strip():
        return None
    try:
        weights = [float(part) for part in str(raw).split(",")]
    except ValueError:
        raise ValueError(f"--diffusion-resolution-weights must be comma-separated numbers, got {raw!r}") from None
    if len(weights) != num_buckets:
        raise ValueError(
            f"--diffusion-resolution-weights has {len(weights)} entries for {num_buckets} "
            "--diffusion-resolution-buckets; pass one weight per bucket or none at all"
        )
    if any(weight < 0 for weight in weights):
        raise ValueError(f"--diffusion-resolution-weights must be non-negative, got {raw!r}")
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"--diffusion-resolution-weights must not sum to 0, got {raw!r}")
    return [weight / total for weight in weights]


def weighted_buckets(args: Namespace, sample: Sample, seed: int) -> tuple[int, int]:
    """Draw one ``--diffusion-resolution-buckets`` entry, weighted by
    ``--diffusion-resolution-weights`` (uniform when unset)."""
    buckets = parse_resolution_buckets(args.diffusion_resolution_buckets)
    probabilities = parse_resolution_weights(args.diffusion_resolution_weights, len(buckets))
    rng = np.random.default_rng(seed)
    index = int(rng.choice(len(buckets), p=probabilities))
    return buckets[index]
