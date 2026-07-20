import json
import os

import wandb

from . import wandb_utils


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def log(args, metrics, step_key, is_media=False):
    # E2E metrics tee (tests/ci/e2e_metrics_registry.py): mirror every metric
    # dict to MILES_METRICS_JSONL; O_APPEND keeps concurrent writers line-atomic.
    jsonl_path = os.environ.get("MILES_METRICS_JSONL")
    if jsonl_path and not is_media:
        with open(jsonl_path, "a") as f:
            f.write(json.dumps({"step_key": step_key, **metrics}, sort_keys=True) + "\n")
    del step_key
    if args.use_wandb:
        # No step=: interleaved rollout/train logs aren't globally monotonic; the
        # x-axis comes from per-metric step_metric (wandb_utils._init_wandb_common).
        wandb.log(metrics)
