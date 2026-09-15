"""One-rollout overlap. Resume discards prefetch and starts from the saved EMA."""

import sys
import time

import ray

from miles.utils import tracking_utils
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.misc import should_run_periodic_action


def train_loop(args, actor_model, rollout_manager, num_rollout_per_epoch):
    if args.start_rollout_id >= args.num_rollout:
        return

    current_batch = ray.get(rollout_manager.generate.remote(args.start_rollout_id))
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        save_checkpoint = should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        # This serial actor saves the current cursor before the next generate advances it.
        cursor_save = (
            rollout_manager.save.remote(rollout_id) if save_checkpoint and args.rollout_global_dataset else None
        )
        next_future = rollout_manager.generate.remote(rollout_id + 1) if rollout_id + 1 < args.num_rollout else None

        ray.get(actor_model.async_train(rollout_id, current_batch))

        # Measure the exposed generation wait before checkpoint I/O can hide it.
        drain_start = time.monotonic()
        if next_future is not None:
            current_batch = ray.get(next_future)
        drain_wait = time.monotonic() - drain_start

        if save_checkpoint:
            if cursor_save is not None:
                ray.get(cursor_save)
            actor_model.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)

        # No generation is in flight while the engines install the new weights.
        actor_model.update_weights()
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))
        tracking_utils.log(
            args,
            {"perf/drain_wait_time": drain_wait, "rollout/step": compute_rollout_step(args, rollout_id)},
            step_key="rollout/step",
        )


def train(args):
    from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
    from miles.utils.logging_utils import configure_logger
    from miles.utils.tracking_utils import init_tracking

    configure_logger()
    if args.colocate or args.offload_train or args.offload_rollout:
        raise ValueError("async training requires separate resident train/rollout GPU pools")
    args.train_async = True

    pgs = create_placement_groups(args)
    init_tracking(args)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model = create_training_models(args, pgs, rollout_manager)

    # Publish initial/restored weights without advancing EMA or its decay schedule.
    actor_model.update_weights()
    if args.eval_interval is not None:
        if args.num_rollout == 0:
            ray.get(rollout_manager.eval.remote(rollout_id=0))
        elif not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(args.start_rollout_id))

    train_loop(args, actor_model, rollout_manager, num_rollout_per_epoch)
    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    from miles.utils.arguments import parse_args

    sys.stdout.reconfigure(line_buffering=True)
    train(parse_args())
