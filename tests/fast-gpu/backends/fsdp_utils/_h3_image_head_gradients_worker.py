"""Worker for test_h3_image_head_gradients: two ranks, each with a pair from another denoising step."""

import os

import diffusers
import torch
import torch.distributed as dist
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence_ref2va_blocks,
)
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import distribute_tensor

from miles.backends.fsdp_utils.configs.h3_image import H3ImageTrainPipelineConfig

TEXT_LEN, LATENT_H, LATENT_W = 5, 8, 12
NUM_IMAGE_ROWS = LATENT_H * LATENT_W // 4
# the denoising step of each rank's pair; step 1's head runs on neither rank
STEP_BY_RANK = (0, 2)


def tiny_h3_image() -> torch.nn.Module:
    torch.manual_seed(7)
    return diffusers.MiniMaxH3ImageTransformer2DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=24,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=24,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=24,
        time_embed_dim=16,
        rope_freq_dim=2,
        num_pdd_heads=3,
    ).cuda()


def backward_through_step(model: torch.nn.Module, step: int) -> None:
    """One pair's loss through the head of ``step``; the pair itself is the same on every rank."""
    layout = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=TEXT_LEN,
        latent_t=1,
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        audio_t=0,
        ref_blocks=[],
        seq_len=((TEXT_LEN + NUM_IMAGE_ROWS + 15) // 16) * 16 + 16,
    )
    torch.manual_seed(11)
    cond = {
        "encoder_hidden_states": torch.randn(1, TEXT_LEN, 8, device="cuda"),
        "h3_packed_layout": layout,
        "h3_token_tags": layout["token_tags"],
    }
    prediction = H3ImageTrainPipelineConfig().compute_noise_pred(
        model=model,
        latents_input={"visual": torch.randn(1, NUM_IMAGE_ROWS, 96, device="cuda")},
        timesteps_input={"visual": torch.tensor([750.0], device="cuda")},
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
        plain = tiny_h3_image()
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

    # the same weights on every rank, wrapped as the loader wraps the DiT: each block, then the root with the heads
    model = tiny_h3_image()
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    for block in model.transformer_blocks:
        fully_shard(block, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
    H3ImageTrainPipelineConfig().postprocess_model_after_materialize(model)

    backward_through_step(model, STEP_BY_RANK[rank])

    mean_gradients = mean_gradients_of_the_ranks()
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        assert gradient is not None, f"{name} has no gradient on rank {rank}"
        # this rank's shard of the mean, cut the way FSDP cut the parameter; no collective, so a rank that
        # fails does not hang the other
        mean_gradient_shard = distribute_tensor(mean_gradients[name], gradient.device_mesh, gradient.placements)
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
