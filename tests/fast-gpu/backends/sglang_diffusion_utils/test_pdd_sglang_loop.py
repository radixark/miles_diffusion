"""Native SGLang stage -> PDD updates on FSDP -> weight sync -> native stage.

Random tiny weights and synthetic embeddings, no pretrained download. The worker
subprocess isolates SGLang globals/patches from other tests in this suite.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-5-gpu-h200", labels=["fsdp"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/SGLang")
def test_online_stage_train_sync(tmp_path):
    subprocess.run(
        [sys.executable, __file__, "--worker", str(tmp_path)],
        check=True,
        timeout=180,
        env={**os.environ, "PYTHONPATH": "."},
    )


def worker(output):
    import dataclasses
    import socket

    from diffusers import FlowMatchEulerDiscreteScheduler as HFScheduler
    from diffusers import QwenImage21Transformer2DModel as HFModel
    from peft import LoraConfig, get_peft_model
    from sglang.multimodal_gen.configs.models.dits.qwenimage21 import QwenImage21ArchConfig, QwenImage21DitConfig
    from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import QwenImage21PipelineConfig
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import ComponentResidencyManager
    from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import QwenImage21Transformer2DModel as Native
    from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import build_layout
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
        QwenImage21DenoisingStage,
        prepare_qwen21_mu,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import TimestepPreparationStage
    from sglang.multimodal_gen.runtime.post_training.weights_updater import WeightsUpdater
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from miles.backends.fsdp_utils.configs.qwen_image21 import QwenImage21TrainPipelineConfig
    from miles.backends.fsdp_utils.hf_weight_iterator import FSDPHfWeightIterator
    from miles.backends.fsdp_utils.loss_hub.pdd import pdd_loss_formula, prepare_pdd_batch
    from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext
    from miles.backends.fsdp_utils.metrics import SCHEMA
    from miles.backends.sglang_diffusion_utils.monkey_patches import patch_pdd_qwen_image21
    from miles.distillation.integration import expand_train_pairs, install_student
    from miles.distillation.pdd import block_update
    from miles.rollout.pdd_rollout import convert_samples_to_train_data
    from miles.utils.diffusion_rollout_response import _parse_denoising_env, _parse_dit_trajectory
    from miles.utils.types import Sample

    torch.cuda.set_device(0)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
    # The real head width: native Q/K norm + RoPE fast paths only fire at 128.
    cfg = dict(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=128,
        context_in_dim=16,
        mlp_ratio=2,
        axes_dims_rope=(16, 56, 56),
    )
    model_path = output / "model"
    HFScheduler(use_dynamic_shifting=True).save_pretrained(model_path / "scheduler")
    config = QwenImage21DitConfig(arch_config=QwenImage21ArchConfig(**cfg))
    server = ServerArgs(
        model_path=str(model_path),
        num_gpus=1,
        attention_backend="torch_sdpa",
        pipeline_config=QwenImage21PipelineConfig(dit_config=config, dit_precision="fp32"),
    )
    set_global_server_args(server)
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    patch_pdd_qwen_image21.apply()
    patch_pdd_qwen_image21.apply()  # idempotent in scheduler workers

    # Trainer: FSDP LoRA student with 32 full heads and a frozen FP32-master teacher, both under BF16 autocast.
    torch.manual_seed(72)
    student = HFModel(**cfg).cuda()
    teacher = HFModel(**cfg).cuda().eval().requires_grad_(False)
    args = SimpleNamespace(
        diffusion_model_family="qwen_image21",
        pdd_num_intervals=32,
        pdd_block_size=4,
        hf_checkpoint=str(model_path),
        diffusion_height=1024,
        diffusion_width=1024,
        seed=42,
        update_weight_buffer_size=4096,
    )
    # The loader's order: LoRA on the backbone, then the heads (trained in full) on the unwrapped model.
    student = get_peft_model(
        student, LoraConfig(r=4, lora_alpha=8, target_modules=QwenImage21TrainPipelineConfig.lora_target_modules)
    )
    install_student(student.get_base_model(), args)
    train_config = QwenImage21TrainPipelineConfig()
    train_config.configure(args)
    for model in (student, teacher):
        train_config.postprocess_model_after_materialize(model)
    fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False)
    fully_shard(student, mesh=init_device_mesh("cuda", (1,)), mp_policy=fp32)
    trained = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trained, lr=1e-3, weight_decay=0, foreach=False)

    # Engine: native transformer and stages, patched for PDD.
    native = Native(config, {}).cuda().eval()
    native.post_load_weights()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path / "scheduler")
    time_stage = TimestepPreparationStage(scheduler, [prepare_qwen21_mu])
    stage = QwenImage21DenoisingStage(transformer=native, scheduler=scheduler)
    pipeline = SimpleNamespace(
        modules={"transformer": native}, component_residency_strategies={}, _stage_name_mapping={"denoising": stage}
    )
    manager = ComponentResidencyManager(pipeline, server)
    manager.refresh_pipeline(pipeline)
    stage.set_component_residency_manager(manager)
    updater = WeightsUpdater(pipeline)

    def sync():
        seen = set()
        for bucket in FSDPHfWeightIterator(args, student, diffusion_model_family="qwen_image21").iter_hf_weights():
            ok, message = updater.update_weights_from_tensor({"transformer": bucket}, target_modules=["transformer"])
            assert ok, message
            for name, value in bucket:
                torch.testing.assert_close(dict(native.named_parameters())[name], value, rtol=0, atol=0)
                seen.add(name)
        assert seen == set(dict(native.named_parameters()))

    embeddings = torch.randn(1, 3, 16, device="cuda")
    noise = torch.randn(1, 4096, 4, device="cuda").bfloat16().float()
    layout = build_layout([False] * 3, [(1, 64, 64)], config.axes_dims_rope, "cuda")

    @torch.no_grad()
    def rollout():
        batch = Req(
            prompt="tiny PDD",
            height=1024,
            width=1024,
            num_inference_steps=8,
            guidance_scale=1.0,
            true_cfg_scale=1.0,
            do_classifier_free_guidance=False,
            prompt_embeds=[embeddings],
            prompt_embeds_mask=[torch.ones(1, 3, device="cuda", dtype=torch.bool)],
            latents=noise.clone(),
            raw_latent_shape=list(noise.shape),
            rollout=True,
            rollout_sde_type="ode",
            rollout_noise_level=0.0,
            rollout_log_prob_no_const=True,
            rollout_return_dit_trajectory=True,
            rollout_return_denoising_env=True,
            extra={"qwen21_positive": dict(layouts=[layout], prefix_caches=None), "qwen21_mu": 1.15},
        )
        batch = time_stage.forward(batch, server)
        manager.begin_request([stage], batch, server)
        data = stage.forward(batch, server).rollout_trajectory_data
        trace = dataclasses.asdict(data.dit_trajectory)
        trace["latents"] = trace["latents"][0]
        sample = Sample(index=0, prompt="tiny PDD", seed=31)
        sample.dit_trajectory = _parse_dit_trajectory(trace, deserialize_func=lambda x: x)
        sample.denoising_env = _parse_denoising_env(
            dataclasses.asdict(data.denoising_env), deserialize_func=lambda x: x
        )
        return convert_samples_to_train_data(args, [sample])["train_data"][0]

    class Metrics:
        def emit_mean(self, name, *, total, count):
            assert name in SCHEMA, f"{name} is not declared in the FSDP metric schema"

    def predict(prepared):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return train_config.compute_noise_pred(
                model=student,
                latents_input=prepared.latents,
                timesteps_input=prepared.timesteps_for_model,
                pos_cond=prepared.pos_cond,
                neg_cond=None,
                joint_cond=None,
                use_cfg=False,
                cfg_batching=False,
                guidance_scale=1.0,
                true_cfg_scale=1.0,
            )

    @torch.no_grad()
    def teacher_target(prepared):
        # The actor's part: run the prepare hook's solver with the frozen teacher's forward.
        def forward(latents, timesteps):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return train_config.compute_noise_pred(
                    model=teacher,
                    latents_input=latents,
                    timesteps_input=timesteps,
                    pos_cond=prepared.pos_cond,
                    neg_cond=None,
                    joint_cond=None,
                    use_cfg=False,
                    cfg_batching=False,
                    guidance_scale=1.0,
                    true_cfg_scale=None,
                )

        return prepared.teacher_solver(forward)

    ctx = DiffusionLossContext(
        models={"transformer": student},
        train_pipeline_config=train_config,
        sde_backend=None,
        args=args,
        forward_dtype=torch.bfloat16,
        device=torch.device("cuda"),
        rollout_id=0,
    )

    # One rollout, eight block-ordered updates, then sync back to the engine.
    sync()
    first = rollout()
    for block in expand_train_pairs(ctx, [first]):
        prepared = prepare_pdd_batch(ctx, [block], microbatch_id=0, optim_step_idx=0)
        optimizer.zero_grad()
        target = teacher_target(prepared)
        loss = pdd_loss_formula(
            ctx,
            [block],
            prepared,
            new_pred=predict(prepared),
            old_pred=None,
            ref_pred=None,
            teacher_pred=target,
            metrics=Metrics(),
            optim_step_idx=0,
            train_output_inputs=[],
        )
        loss.backward()
        assert all(p.grad is None for p in teacher.parameters())
        optimizer.step()
    sync()

    # The trained heads now drive the engine; every interval state must match the trainer's heads.
    second = rollout()
    assert not torch.equal(first["pdd_trajectory"], second["pdd_trajectory"])
    with torch.no_grad():
        for block in expand_train_pairs(ctx, [second]):
            prepared = prepare_pdd_batch(ctx, [block], microbatch_id=0, optim_step_idx=0)
            x, start = prepared.latents["visual"], block["pdd_start"]
            heads = predict(prepared)["visual"].reshape(*x.shape[:-1], 32, x.shape[-1]).permute(2, 0, 1, 3)
            for end in range(start + 1, start + 5):
                following = block_update(x, heads, prepared.extras["grid"], start, end)
                expected = second["pdd_trajectory"][end].cuda().unsqueeze(0)
                # Mismatched BF16 timestep rounding alone gave 1.5-3% on some blocks.
                relative = (following - expected).abs().mean() / (expected - x).abs().mean()
                assert relative < 5e-3, (start, end, float(relative))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(Path(sys.argv[2]))
    else:
        sys.exit(pytest.main([__file__, "-v"]))
