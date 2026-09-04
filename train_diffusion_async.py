import logging
import sys
import time

import ray

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils import tracking_utils
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking


def train(args):
    configure_logger()
    logger = logging.getLogger(__name__)
    assert not args.colocate, "async training overlaps train and rollout; drop --colocate"
    assert not args.offload_train and not args.offload_rollout, "async training keeps both pools resident"

    logger.info("train_async: creating placement groups")
    pgs = create_placement_groups(args)
    init_tracking(args)

    logger.info("train_async: creating rollout manager")
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    logger.info("train_async: creating training model")
    actor_model = create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def save(rollout_id):
        actor_model.save_model(
            rollout_id,
            force_sync=rollout_id == args.num_rollout - 1,
        )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    def log_drain_wait(rollout_id, drain_wait):
        # The actor already logs perf/train_time, perf/step_time and perf/wait_time_ratio;
        # drain_wait (time spent waiting for the prefetched rollout at the barrier) is the
        # only phase invisible to it.
        log_dict = {
            "perf/drain_wait_time": drain_wait,
            "rollout/step": compute_rollout_step(args, rollout_id),
        }
        tracking_utils.log(args, log_dict, step_key="rollout/step")

    if args.eval_interval is not None and not args.skip_eval_before_train and args.num_rollout > 0:
        ray.get(rollout_manager.eval.remote(args.start_rollout_id))

    # one-step overlap: generate(rollout_id + 1) runs while train(rollout_id) runs, so the
    # trained batch is exactly one weight version stale. Weights are only pushed at the
    # barrier below, after the in-flight generation drains, so every rollout sees a single
    # weight version.
    generate_future = None
    if args.start_rollout_id < args.num_rollout:
        generate_future = rollout_manager.generate.remote(args.start_rollout_id)

    rollout_data_ref = None
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if generate_future is not None:
            rollout_data_ref = ray.get(generate_future)
        if rollout_id + 1 < args.num_rollout:
            generate_future = rollout_manager.generate.remote(rollout_id + 1)
        else:
            generate_future = None

        logger.info(f"train_async: rollout {rollout_id} actor train start")
        train_start = time.time()
        ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
        train_wall = time.time() - train_start

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        # drain the in-flight generation before updating weights so no rollout runs
        # under a mid-generation weight swap
        drain_start = time.time()
        if generate_future is not None:
            rollout_data_ref = ray.get(generate_future)
            generate_future = None
        drain_wait = time.time() - drain_start

        update_start = time.time()
        actor_model.update_weights()
        update_wall = time.time() - update_start

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

        log_drain_wait(rollout_id, drain_wait)
        logger.info(
            f"train_async: rollout {rollout_id} done "
            f"train_wall={train_wall:.1f}s drain_wait={drain_wait:.1f}s update={update_wall:.1f}s"
        )

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    # Ensure stdout is line-buffered so nohup logs show progress immediately.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    train(args)
