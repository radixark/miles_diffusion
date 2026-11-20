"""Worker for test_h3_image_head_gradients: two ranks, each with a pair from another denoising step."""

import os
from argparse import Namespace

import torch
import torch.distributed as dist
from tests.fast.backends.fsdp_utils.configs.test_h3_image import SIGMA, _rows_and_cond, _t2i_layout, _tiny_h3_image
from torch.distributed.tensor import distribute_tensor

from miles.backends.fsdp_utils.configs.h3_image import H3ImageTrainPipelineConfig
from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend
from miles.backends.fsdp_utils.model_loader import apply_fsdp2

# the denoising step of each rank's pair; step 1's head runs on neither rank
STEP_BY_RANK = (0, 2)


def backward_through_step(model: torch.nn.Module, step: int) -> None:
    """One pair's loss through the head of ``step``; the pair itself is the same on every rank."""
    visual, cond = _rows_and_cond(_t2i_layout())
    prediction = H3ImageTrainPipelineConfig().compute_noise_pred(
        model=model,
        latents_input={"visual": visual.cuda()},
        timesteps_input={"visual": torch.tensor([SIGMA * 1000.0], device="cuda")},
        denoising_step_indices=torch.tensor([step]),
        pos_cond=cond,
        neg_cond=None,
        joint_cond=None,
        use_cfg=False,
        cfg_batching=False,
        guidance_scale=1.0,
        true_cfg_scale=None,
    )["visual"]
    prediction.square().sum().backward()


def mean_gradients_of_the_ranks() -> dict[str, torch.Tensor]:
    """What each parameter's gradient must reduce to: the mean of the ranks' own, zero where a rank has none."""
    mean_gradients: dict[str, torch.Tensor] = {}
    for step in STEP_BY_RANK:
        plain = _tiny_h3_image().cuda()
        backward_through_step(plain, step)
        for name, parameter in plain.named_parameters():
            gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
            mean_gradients[name] = mean_gradients.get(name, 0) + gradient / len(STEP_BY_RANK)
    return mean_gradients


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    assert dist.get_world_size() == len(STEP_BY_RANK)

    # the same weights on every rank, wrapped and postprocessed as the loader does it
    model = _tiny_h3_image().cuda()
    train_pipeline_config = H3ImageTrainPipelineConfig()
    apply_fsdp2(
        model,
        DiffusersModelBackend(train_pipeline_config).fsdp_parallel_plan(model),
        args=Namespace(diffusion_forward_dtype="fp32", fsdp_reduce_dtype="fp32"),
    )
    train_pipeline_config.postprocess_model_after_materialize(model)

    backward_through_step(model, STEP_BY_RANK[rank])

    mean_gradients = mean_gradients_of_the_ranks()
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        assert gradient is not None, f"{name} has no gradient on rank {rank}"
        # src_data_rank=None cuts this rank's shard locally: no collective, so a failing rank does not hang the other
        mean_gradient_shard = distribute_tensor(
            mean_gradients[name], gradient.device_mesh, gradient.placements, src_data_rank=None
        )
        torch.testing.assert_close(
            gradient.to_local(),
            mean_gradient_shard.to_local(),
            msg=lambda mismatch, name=name: f"{name} on rank {rank}: {mismatch}",
        )

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("OK", flush=True)


if __name__ == "__main__":
    main()
