"""Train-only entry point: rollout i+1 is generated while rollout i trains.

An SFT rollout encodes a dataset batch and does not depend on the trained weights, so only the
first rollout is waited for. train_diffusion.py keeps the RL loop, which must sync weights before
each generate.
"""

import logging
import sys

import ray

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking


def train(args):
    assert args.train_only, "train_sft.py runs no rollout engines; pass --train-only (the SFT recipes do)"
    configure_logger()
    logger = logging.getLogger(__name__)
    pgs = create_placement_groups(args)
    init_tracking(args)

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model = create_training_models(args, pgs, rollout_manager)

    def is_save_rollout(rollout_id):
        return should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout)

    # colocated encoders wait for the overlapped train step before an encode burst
    rollout_shares_train_gpus = pgs["rollout"][0] is pgs["actor"][0]

    next_rollout_data = None
    if args.start_rollout_id < args.num_rollout:
        next_rollout_data = rollout_manager.generate.remote(args.start_rollout_id)

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        rollout_data_ref = ray.get(next_rollout_data)
        logger.info(f"train: rollout {rollout_id} generate done")
        if is_save_rollout(rollout_id) and args.rollout_global_dataset:
            # a resume at rollout_id + 1 needs the cursor from before the next generate moves it
            ray.get(rollout_manager.save.remote(rollout_id))

        logger.info(f"train: rollout {rollout_id} actor train start")
        train_refs = actor_model.async_train(rollout_id, rollout_data_ref)
        next_rollout_data = None
        if rollout_id + 1 < args.num_rollout:
            next_rollout_data = rollout_manager.generate.remote(
                rollout_id + 1, train_in_flight=train_refs if rollout_shares_train_gpus else None
            )
        ray.get(train_refs)
        logger.info(f"train: rollout {rollout_id} actor train done")

        if is_save_rollout(rollout_id):
            actor_model.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)

        if args.offload_train:
            actor_model.offload()
        else:
            actor_model.clear_memory()

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    # Ensure stdout is line-buffered so nohup logs show progress immediately.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    train(args)
