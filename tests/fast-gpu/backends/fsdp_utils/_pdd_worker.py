"""Tiny-Qwen FSDP parity and strict checkpoint round-trip for the Miles PDD path.

Run with torchrun --standalone --nproc-per-node=1 (or 2). Uses a random tiny model;
this verifies FSDP numerics and resume, not pretrained image quality.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from diffusers import QwenImage21Transformer2DModel
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh

from miles.backends.fsdp_utils import checkpoint
from miles.backends.fsdp_utils.configs.qwen_image21 import QwenImage21TrainPipelineConfig
from miles.backends.fsdp_utils.loss_hub.pdd import pdd_loss_formula, prepare_pdd_batch
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext
from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend
from miles.backends.fsdp_utils.model_loader import load_fsdp_models
from miles.distillation.integration import install_student
from miles.distillation.pdd import pdd_loss


def tiny(dtype=torch.float32):
    torch.manual_seed(72)
    return QwenImage21Transformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=16,
        axes_dims_rope=(4, 6, 6),
        mlp_ratio=2,
    ).to(dtype=dtype)


class Backend(DiffusersModelBackend):
    def load_component(self, component, *, checkpoint_path, master_dtype, materialize_weights):
        return tiny(master_dtype)


class Metrics:
    def emit_mean(self, *args, **kwargs):
        pass


def full(model):
    return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("fsdp",))
    parallel = SimpleNamespace(get_mesh=lambda name: mesh, get_optional_mesh=lambda name: None)
    args = SimpleNamespace(
        loss_type="pdd",
        fsdp_master_dtype="fp32",
        fsdp_reduce_dtype="fp32",
        diffusion_forward_dtype="bf16",
        update_weight_target_modules=["transformer"],
        fsdp_attention_backend=None,
        gradient_checkpointing=True,
        use_lora=False,
        pdd_num_intervals=32,
        pdd_block_size=4,
        offload_train=False,
        diffusion_height=64,
        diffusion_width=64,
        seed=42,
        save=str(cli.output),
        load=None,
        no_save_optim=False,
        no_load_optim=False,
        no_load_rng=False,
        ckpt_step=None,
        start_rollout_id=0,
    )
    config = QwenImage21TrainPipelineConfig()
    config.configure(args)
    config.prepare_cond_kwargs = lambda cond, device: cond  # synthetic conditioning, not a rollout env
    backend = Backend(config)
    models = load_fsdp_models(args, backend, config, parallel, checkpoint_path="tiny", trainable=True)
    teachers = load_fsdp_models(args, backend, config, parallel, checkpoint_path="tiny", trainable=False)
    model = models["transformer"]
    teacher = teachers["transformer"]
    reference = tiny().to(device)
    install_student(reference, args)
    reference.load_state_dict(full(model))
    refteacher = tiny(torch.bfloat16).to(device).eval().requires_grad_(False)
    refteacher.load_state_dict(full(teacher))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, foreach=False)
    refopt = torch.optim.AdamW(reference.parameters(), lr=1e-4, weight_decay=0, foreach=False)
    grid = torch.linspace(1, 0, 33, device=device)
    torch.manual_seed(100 + rank)
    cond = {
        "encoder_hidden_states": torch.randn(1, 3, 16, device=device),
        "encoder_hidden_states_mask": None,
        "img_mask": torch.tensor([[False] * 3 + [True] * 4], device=device),
        "img_shapes": [[(1, 4, 4)]],
    }
    # A rollout trajectory: the eight block states plus the endpoint, BF16-rounded like the engine's noise.
    trajectory = torch.randn(9, 16, 4).bfloat16().float()
    env = SimpleNamespace(pos_cond_kwargs=cond)
    ctx = DiffusionLossContext(
        models=models,
        teacher_models=teachers,
        train_pipeline_config=config,
        sde_backend=None,
        args=args,
        forward_dtype=torch.bfloat16,
        device=device,
        algorithm_state={"grid": grid},
    )
    max_output = 0.0
    max_grad = 0.0
    for start in range(0, 32, 4):
        optimizer.zero_grad()
        refopt.zero_grad()
        pair = {"sample_index": rank, "pdd_start": start, "pdd_trajectory": trajectory, "denoising_env": env}
        prepared = prepare_pdd_batch(ctx, [pair])
        x = prepared.latents["visual"]
        torch.testing.assert_close(x[0].cpu(), trajectory[start // 4], rtol=0, atol=0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = config.compute_noise_pred(
                model=model,
                latents_input={"visual": x},
                timesteps_input=prepared.timesteps_for_model,
                pos_cond=cond,
                neg_cond=None,
                joint_cond=None,
                use_cfg=False,
                cfg_batching=False,
                guidance_scale=1.0,
                true_cfg_scale=1.0,
            )
            # Unsharded student: keep the image-token tail, split into [head, batch, token, channel].
            model_sigma = prepared.timesteps_for_model["visual"]
            refpred = reference(hidden_states=x, timestep=model_sigma, return_dict=False, **cond)[0]
            refpred = refpred[:, -16:].reshape(1, 16, 32, 4).permute(2, 0, 1, 3)
            loss = pdd_loss_formula(
                ctx, [pair], prepared, new_pred=pred, old_pred=None, ref_pred=None, metrics=Metrics()
            )

            def tf(state, sigma):
                c = {
                    k: v.bfloat16() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                    for k, v in cond.items()
                }
                return refteacher(hidden_states=state.bfloat16(), timestep=sigma.expand(1), return_dict=False, **c)[0][
                    :, -16:
                ].float()

            refloss = pdd_loss(refpred, x, grid, start, prepared.extras["selected"], tf).sum()
        rawref = refpred.permute(1, 2, 0, 3).flatten(2)
        err = (pred["visual"] - rawref).abs().max().item()
        max_output = max(max_output, err)
        torch.testing.assert_close(pred["visual"], rawref, rtol=0, atol=0)
        torch.testing.assert_close(loss, refloss, rtol=0, atol=0)
        loss.backward()
        refloss.backward()
        for (name, p), (rname, r) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
            assert name == rname and (p.grad is None) == (r.grad is None), name
            if p.grad is not None:
                dist.all_reduce(r.grad)
                r.grad.div_(dist.get_world_size())
                grad = p.grad.full_tensor()
                max_grad = max(max_grad, (grad - r.grad).abs().max().item())
                torch.testing.assert_close(grad, r.grad, rtol=2e-5, atol=2e-6, msg=name)
        optimizer.step()
        refopt.step()
    assert all(p.grad is None for p in teacher.parameters())
    for name, p in full(model).items():
        torch.testing.assert_close(p, reference.state_dict()[name], rtol=2e-5, atol=2e-6, msg=name)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    actor = SimpleNamespace(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        ema_optimizer=None,
        train_pipeline_config=config,
        global_step=8,
        micro_step=0,
        pdd_signature={"test": "tiny-fsdp", "version": 1},
    )
    if rank == 0:
        (cli.output / "rollout").mkdir(parents=True, exist_ok=True)
        torch.save({"sample_index": 2}, cli.output / "rollout/global_dataset_state_dict_0.pt")
    dist.barrier()
    checkpoint.save(actor, 0)
    saved = {k: v.clone() for k, v in full(model).items()}
    saved_rng = torch.get_rng_state().clone()
    saved_cuda = torch.cuda.get_rng_state_all()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    args.load = args.save
    payload = checkpoint.load(actor)
    checkpoint.finalize_load(actor, payload)
    assert actor.global_step == 8 and args.start_rollout_id == 1
    assert torch.equal(torch.get_rng_state(), saved_rng)
    assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), saved_cuda, strict=True))
    for name, p in full(model).items():
        torch.testing.assert_close(p, saved[name], rtol=0, atol=0)
    # Optimizer moments must survive too; compare the next update with the uninterrupted reference.
    optimizer.zero_grad()
    refopt.zero_grad()
    for p, r in zip(model.parameters(), reference.parameters(), strict=True):
        p.grad = torch.ones_like(p)
        r.grad = torch.ones_like(r)
    optimizer.step()
    refopt.step()
    for name, p in full(model).items():
        torch.testing.assert_close(p, reference.state_dict()[name], rtol=2e-5, atol=2e-6, msg=name)
    if rank == 0:
        report = {
            "state": "passed",
            "world_size": dist.get_world_size(),
            "updates": 8,
            "max_forward_abs_diff": max_output,
            "max_gradient_abs_diff_rank0": max_grad,
            "checkpoint_weights_exact": True,
            "per_rank_rng_exact": True,
            "optimizer_next_update_matches": True,
            "teacher_frozen": True,
            "model": "random tiny Qwen-Image 2.1",
        }
        (cli.output / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
