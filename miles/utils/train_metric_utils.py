import logging
from argparse import Namespace
from copy import deepcopy

import torch

from miles.utils import tracking_utils
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.timer import Timer

logger = logging.getLogger(__name__)


def log_model_output_debug_metrics(
    log_stats: dict[str, list[torch.Tensor]],
    *,
    noise_pred_flat: torch.Tensor,
    grids: dict,
    sample_indices: torch.Tensor,
    tstep_indices: torch.Tensor,
    tile_sample_count: int,
    tile_tstep_count: int,
) -> None:
    """Compare train-side noise_pred against rollout debug model outputs."""
    rollout_mo_window = grids.get("rollout_model_outputs")
    if rollout_mo_window is None:
        return

    rollout_mo_tile = rollout_mo_window[sample_indices][:, tstep_indices]
    rollout_mo_flat = rollout_mo_tile.reshape(
        tile_sample_count * tile_tstep_count, *rollout_mo_tile.shape[2:]
    )
    diff = (noise_pred_flat.float() - rollout_mo_flat.float()).abs()
    ref_max = rollout_mo_flat.float().abs().max() + 1e-30
    log_stats["model_output_max_abs_diff"].append(diff.max().detach())
    log_stats["model_output_mean_abs_diff"].append(diff.mean().detach())
    log_stats["model_output_rel_max"].append((diff.max() / ref_max).detach())
    flat_train = noise_pred_flat.float().reshape(noise_pred_flat.shape[0], -1)
    flat_rollout = rollout_mo_flat.float().reshape(rollout_mo_flat.shape[0], -1)
    log_stats["model_output_cosine_sim"].append(
        torch.nn.functional.cosine_similarity(flat_train, flat_rollout, dim=1).mean().detach()
    )


def log_perf_data_raw(rollout_id: int, args: Namespace, is_primary_rank: bool) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    logger.info(f"perf {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")
