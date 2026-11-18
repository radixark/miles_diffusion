"""DMD2: the critic group learns the student's samples and scores them; the actor follows the scores.

Each rollout trains the two groups in the order --dmd2-critic-scoring asks for, K = --dmd2-critic-steps-per-rollout:

    reuse-critic-step   critic trains its K steps, the last one scoring ──► actor trains on the scores
    original            critic trains steps 0..K-2 ──► critic scores, forward only ──► actor trains on the scores
                        ──► critic trains step K-1

A group saves after its last train of the rollout and, under --offload-train, sleeps before another group wakes.
"""

import logging
import sys

import ray

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking

logger = logging.getLogger(__name__)

SCORING_HOOKS = "critic-scoring"


def train_rollout(args, rollout_id, rollout_data_ref, train_groups, should_save):
    """Train the critic and the actor on the rollout, in the order of --dmd2-critic-scoring."""
    critic, actor = train_groups["critic"], train_groups["actor"]
    critic_steps = args.dmd2_critic_steps_per_rollout

    def train(group, role, external_data=None, optim_steps=None):
        logger.info(f"train: rollout {rollout_id} {role} train start")
        # Each rank returns a Box around its output's ObjectRef; the driver keeps the boxes (not the payload)
        # and hands them to the later group, as RolloutManager.generate does for the shards.
        output = ray.get(group.async_train(rollout_id, rollout_data_ref, external_data, optim_steps))
        logger.info(f"train: rollout {rollout_id} {role} train done")
        return output

    def done(group):
        """The group's last train of the rollout is over: save, and sleep before another group wakes."""
        if should_save:
            group.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)
        if args.offload_train:
            group.offload()

    if args.dmd2_critic_scoring == "reuse-critic-step":
        scores = train(critic, "critic")
        done(critic)
        train(actor, "actor", {"critic": scores})
        done(actor)
        return

    if critic_steps > 1:
        train(critic, "critic", optim_steps=(0, critic_steps - 1))
    logger.info(f"train: rollout {rollout_id} critic scoring start")
    scores = ray.get(critic.async_compute(rollout_id, rollout_data_ref, {}, SCORING_HOOKS))
    logger.info(f"train: rollout {rollout_id} critic scoring done")
    if args.offload_train:
        critic.offload()
    train(actor, "actor", {"critic": scores})
    done(actor)
    train(critic, "critic", optim_steps=(critic_steps - 1, critic_steps))
    done(critic)


def train(args):
    configure_logger()
    if args.loss_type != "dmd2":
        raise ValueError("train_dmd2.py runs --loss-type dmd2; the other loss types run through train_diffusion.py")
    # allocate the GPUs
    logger.info("train: creating placement groups")
    pgs = create_placement_groups(args)
    logger.info("train: placement groups ready")
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    logger.info("train: creating rollout manager")
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    logger.info("train: rollout manager ready")

    logger.info("train: creating training model")
    train_groups = create_training_models(args, pgs, rollout_manager)
    actor_model = train_groups["actor"]
    logger.info("train: training model ready")

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        logger.info(f"train: rollout {rollout_id} generate start")
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        # generating rollout data
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        logger.info(f"train: rollout {rollout_id} generate done")

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        should_save = should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        train_rollout(args, rollout_id, rollout_data_ref, train_groups, should_save)
        ray.get(rollout_manager.release.remote(rollout_id))

        if should_save and args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))
        if not args.offload_train:
            for train_group in train_groups.values():
                train_group.clear_memory()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    # Ensure stdout is line-buffered so nohup logs show progress immediately.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    train(args)
