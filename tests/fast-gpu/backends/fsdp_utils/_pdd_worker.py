"""Tiny-Qwen FSDP parity for the Miles PDD path, loaded through PDD's generic loader hooks.

Run with torchrun --standalone --nproc-per-node=1 (or 2). Uses a random tiny model;
this verifies FSDP numerics, not pretrained image quality.
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

from miles.backends.fsdp_utils.configs.qwen_image21 import QwenImage21TrainPipelineConfig
from miles.backends.fsdp_utils.loss_hub.pdd import pdd_loss_formula, prepare_pdd_batch
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext
from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend
from miles.backends.fsdp_utils.model_loader import load_fsdp_models
from miles.distillation import integration
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
        diffusion_model_family="qwen_image21",
        fsdp_master_dtype="fp32",
        fsdp_param_dtype="fp32",
        fsdp_reduce_dtype="fp32",
        custom_train_model_transform_path="miles.distillation.integration.install_student",
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
        hf_checkpoint="tiny",
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
    refteacher = tiny().to(device).eval().requires_grad_(False)
    refteacher.load_state_dict(full(teacher))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, foreach=False)
    refopt = torch.optim.AdamW(reference.parameters(), lr=1e-4, weight_decay=0, foreach=False)
    grid = torch.linspace(1, 0, 33, device=device)
    integration.make_grid = lambda *args: grid  # the tiny model has no scheduler config
    torch.manual_seed(100 + rank)
    cond = {
        "encoder_hidden_states": torch.randn(1, 3, 16, device=device),
        "encoder_hidden_states_mask": None,
        "img_mask": torch.tensor([[False] * 3 + [True] * 4], device=device),
        "img_shapes": [[(1, 4, 4)]],
    }
    # A rollout trajectory: every interval start plus the endpoint, BF16-rounded like the engine's noise.
    trajectory = torch.randn(33, 16, 4).bfloat16().float()
    env = SimpleNamespace(pos_cond_kwargs=cond)
    ctx = DiffusionLossContext(
        models=models,
        train_pipeline_config=config,
        sde_backend=None,
        args=args,
        forward_dtype=torch.bfloat16,
        device=device,
    )
    max_output = 0.0
    max_grad = 0.0
    for start in range(0, 32, 4):
        optimizer.zero_grad()
        refopt.zero_grad()
        pair = {
            "sample_index": rank,
            "pdd_start": start,
            "pdd_trajectory": trajectory,
            "denoising_env": env,
        }
        prepared = prepare_pdd_batch(ctx, [pair], microbatch_id=0, optim_step_idx=0)
        x = prepared.latents["visual"]
        torch.testing.assert_close(x[0].cpu(), trajectory[start], rtol=0, atol=0)
        with torch.autocast("cuda", dtype=torch.bfloat16):

            def teacher_forward(latents, timesteps):
                # The actor's teacher forward: the sharded frozen teacher on this batch's conditioning.
                return config.compute_noise_pred(
                    model=teacher,
                    latents_input=latents,
                    timesteps_input=timesteps,
                    pos_cond=cond,
                    neg_cond=None,
                    joint_cond=None,
                    use_cfg=False,
                    cfg_batching=False,
                    guidance_scale=1.0,
                    true_cfg_scale=None,
                )

            with torch.no_grad():
                target = prepared.teacher_solver(teacher_forward)
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
                ctx,
                [pair],
                prepared,
                new_pred=pred,
                old_pred=None,
                ref_pred=None,
                teacher_pred=target,
                metrics=Metrics(),
                optim_step_idx=0,
                train_output_inputs=[],
            )

            def tf(state, sigma):
                return refteacher(hidden_states=state, timestep=sigma.expand(1), return_dict=False, **cond)[0][
                    :, -16:
                ].float()

            selected = prepared.extras["selected"]
            x_selected = trajectory[selected].to(device).unsqueeze(0)
            refloss = pdd_loss(refpred, x, grid, start, selected, tf, x_selected=x_selected).sum()
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
                # Losses match bitwise; the BF16-autocast backward sums in a different order under FSDP
                # (observed up to 1.3e-3 of the largest entry, well under one BF16 ulp).
                scale = r.grad.abs().max().item()
                torch.testing.assert_close(grad, r.grad, rtol=2e-5, atol=2e-6 + 5e-3 * scale, msg=name)
        optimizer.step()
        refopt.step()
    assert all(p.grad is None for p in teacher.parameters())
    for name, p in full(model).items():
        torch.testing.assert_close(p, reference.state_dict()[name], rtol=2e-5, atol=2e-6, msg=name)
    if rank == 0:
        report = {
            "state": "passed",
            "world_size": dist.get_world_size(),
            "updates": 8,
            "max_forward_abs_diff": max_output,
            "max_gradient_abs_diff_rank0": max_grad,
            "teacher_frozen": True,
            "model": "random tiny Qwen-Image 2.1",
        }
        (cli.output / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
