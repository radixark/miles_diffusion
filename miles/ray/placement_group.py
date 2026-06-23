import logging
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .actor_group import RayTrainGroup
from .rollout import RolloutManager

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    logger.info("Waiting for placement group to be ready...")
    ray.get(pg.ready())
    logger.info("Placement group is ready.")
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


def create_placement_groups(args):
    """Create placement groups for actor and rollout engines.

    Two topologies:
    - Colocate (or --debug-{train,rollout}-only): one combined placement
      group; both roles see the same bundle list.
    - Disaggregate (the else branch): two separate placement groups so
      train and rollout each own a disjoint GPU pool — avoids bundle
      overlap / scheduling deadlock when running side-by-side.
    """
    if not args.colocate and not args.debug_train_only and not args.debug_rollout_only:
        logger.info("Creating placement groups (separate actor/rollout)...")
        actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_gpus = args.rollout_num_gpus
        actor_pg = _create_placement_group(actor_gpus) if actor_gpus > 0 else None
        rollout_pg = _create_placement_group(rollout_gpus) if rollout_gpus > 0 else None
        return {
            "actor": actor_pg,
            "rollout": rollout_pg,
        }

    if args.debug_rollout_only:
        num_gpus = args.rollout_num_gpus
    else:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, all_reordered_bundle_indices, all_reordered_gpu_ids = _create_placement_group(num_gpus)

    actor_pg_reordered_bundle_indices = all_reordered_bundle_indices
    actor_pg_reordered_gpu_ids = all_reordered_gpu_ids
    rollout_pg_reordered_bundle_indices = all_reordered_bundle_indices if not args.debug_train_only else []
    rollout_pg_reordered_gpu_ids = all_reordered_gpu_ids if not args.debug_train_only else []

    return {
        "actor": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }


def allocate_train_group(args, num_nodes, num_gpus_per_node, pg):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        # Diffusion training is GPU-heavy; avoid fractional-GPU scheduling stalls.
        num_gpus_per_actor=0.8,
    )


def create_training_models(args, pgs, rollout_manager):
    logger.info("Initializing actor model...")
    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
    )
    start_rollout_ids = ray.get(actor_model.async_init(args, role="actor", with_ref=False))
    logger.info("Actor model initialized.")

    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    actor_model.set_rollout_manager(rollout_manager)
    if args.rollout_global_dataset:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    return actor_model


def create_rollout_manager(args, pg):
    logger.info("Creating rollout manager (num_gpus=%s)", 0)
    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0,
    ).remote(args, pg)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        logger.info("Fetching num_rollout_per_epoch from rollout manager...")
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0
        logger.info("Computed num_rollout=%s (num_rollout_per_epoch=%s)", args.num_rollout, num_rollout_per_epoch)

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    logger.info("Rollout manager created.")
    return rollout_manager, num_rollout_per_epoch
