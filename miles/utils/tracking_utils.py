import wandb
from miles.utils.tensorboard_utils import _TensorboardAdapter

from . import wandb_utils


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        # Do NOT pass step=... to wandb.log: wandb requires the explicit step
        # argument to be monotonically increasing across all log calls, but
        # miles interleaves per-rollout logs (step=rollout_id, +1 per rollout)
        # with per-optimizer-step training logs (step=global_step, +2 per
        # rollout), so rollout calls end up < current internal step and get
        # dropped / bumped — users see rollout/* x-axis jumping like 1, 6, 10.
        # The right pattern with wandb.define_metric(step_metric=...) is to
        # leave wandb's internal commit counter auto-incrementing and let
        # define_metric pull the x-axis value out of each metric dict
        # (``rollout/step`` / ``train/step`` / ``eval/step``) — that's the
        # step_metric wiring set up in wandb_utils._init_wandb_common().
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
