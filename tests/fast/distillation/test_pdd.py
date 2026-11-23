"""PDD on CPU: the interval math, the engine's cached-head steps, and eight block updates from rollout
states against an independent optimizer loop. The tests play the actor where it would run the teacher."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from miles.backends.fsdp_utils.loss_hub.pdd import pdd_loss_formula, prepare_pdd_batch
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext
from miles.backends.sglang_diffusion_utils.monkey_patches.patch_pdd_qwen_image21 import PDDProjection
from miles.distillation import integration
from miles.distillation.integration import (
    embedded_sigmas,
    engine_timesteps,
    expand_train_pairs,
    install_student,
    set_pdd_defaults,
    validate_pdd_args,
)
from miles.distillation.pdd import ParallelLinear, block_update, interval_loss, teacher_mean_velocity
from miles.rollout import pdd_rollout

GRID = torch.linspace(1, 0, 33)


@pytest.fixture(autouse=True)
def linear_grid(monkeypatch):
    monkeypatch.setattr(integration, "make_grid", lambda *args: GRID)


# --- interval math -------------------------------------------------------------------------------------


@pytest.mark.parametrize("descending", [False, True])
def test_block_updates_follow_an_exact_linear_ode(descending):
    """Analytic oracle dx/dt = a*x on a nonuniform grid: exact per-interval mean velocities integrate exactly."""
    grid = torch.tensor([0, 0.03, 0.11, 0.24, 0.42, 0.58, 0.73, 0.9, 1], dtype=torch.float64)
    if descending:
        grid = grid.flip(0)
    a, x0 = torch.tensor(-0.7, dtype=torch.float64), torch.tensor([[1.3, -2.1]], dtype=torch.float64)

    def heads(state, time):
        left, right = torch.exp(a * (grid[:-1] - time)), torch.exp(a * (grid[1:] - time))
        return ((right - left) / grid.diff())[:, None, None] * state

    x = x0
    for start in range(0, 8, 4):
        x = block_update(x, heads(x, grid[start]), grid, start, start + 4)
    torch.testing.assert_close(x, x0 * torch.exp(a * (grid[-1] - grid[0])), rtol=1e-12, atol=1e-12)


def test_teacher_target_is_a_detached_midpoint_step():
    weight = nn.Parameter(torch.tensor(0.5))
    calls = []

    def teacher(state, time):
        calls.append(torch.is_grad_enabled())
        return weight * state + time

    x, t, h = torch.tensor([[1.0, 2.0]], requires_grad=True), torch.tensor(0.7), torch.tensor(-0.2)
    half = x.detach() + 0.5 * h * (0.5 * x.detach() + t)
    target = teacher_mean_velocity(teacher, x, t, h)
    torch.testing.assert_close(target, 0.5 * half + t + 0.5 * h)
    assert calls == [False, False] and not target.requires_grad


# --- rollout side --------------------------------------------------------------------------------------


def test_engine_steps_on_cached_heads_reproduce_every_interval():
    """One DiT forward per block; the block's other intervals step on the cached heads."""
    torch.manual_seed(8)
    projection = PDDProjection(nn.Linear(3, 2))
    grid = GRID.square()
    hidden, x = torch.randn(1, 5, 3), torch.randn(1, 5, 2)
    for interval in range(32):
        start = interval - interval % 4
        if interval == start:
            torch.testing.assert_close(projection(hidden), projection.parallel(hidden)[0])
            block_state = x
        x = x + projection.block[interval] * (grid[interval + 1] - grid[interval])
        expected = block_update(block_state, projection.parallel(hidden), grid, start, interval + 1)
        torch.testing.assert_close(x, expected, rtol=1e-6, atol=1e-6)


def test_engine_timesteps_reproduce_the_trainer_bf16_sigma():
    # The native DiT embeds bf16(bf16(t) / 1000); the Diffusers trainer embeds bf16(sigma).
    sigmas = torch.linspace(1, 0.01, 2000)
    naive = embedded_sigmas(sigmas * 1000) != sigmas.bfloat16().float()
    fixed = embedded_sigmas(engine_timesteps(sigmas)) != sigmas.bfloat16().float()
    assert naive.float().mean() > 0.2 and fixed.float().mean() < 0.05 and not (fixed & ~naive).any()


def test_converter_accepts_only_the_full_trajectory_on_the_training_grid():
    sample = SimpleNamespace(
        index=17,
        denoising_env=SimpleNamespace(pos_cond_kwargs={}),
        dit_trajectory=SimpleNamespace(sigmas=GRID, latents=torch.randn(33, 4, 2), latent_step_indices=None),
    )
    args = SimpleNamespace(hf_checkpoint="unused", pdd_num_intervals=32)
    pair = pdd_rollout.convert_samples_to_train_data(args, [sample])["train_data"][0]
    torch.testing.assert_close(pair["pdd_trajectory"], sample.dit_trajectory.latents)
    for field, value in [
        ("sigmas", GRID + 0.01),
        ("latents", torch.randn(9, 4, 2)),
        ("latents", torch.full((33, 4, 2), float("nan"))),
    ]:
        bad = copy.deepcopy(sample)
        setattr(bad.dit_trajectory, field, value)
        with pytest.raises(ValueError):
            pdd_rollout.convert_samples_to_train_data(args, [bad])


# --- trainer side --------------------------------------------------------------------------------------


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.head = ParallelLinear(nn.Linear(2, 2), 32)

    def forward(self, x, sigma):  # [N, B, S, C]
        return self.head(self.base(x) + sigma.reshape(-1, 1, 1))


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.inputs = []

    def forward(self, x, sigma):
        self.inputs.append(x.detach().clone())
        return self.scale * x + sigma.reshape(-1, 1, 1)


class Config:
    def prepare_cond_kwargs(self, cond, device):
        return cond


class Metrics:
    def emit_mean(self, *args, **kwargs):
        pass


def context(student, rollout=0):
    args = SimpleNamespace(hf_checkpoint="unused", pdd_num_intervals=32, pdd_block_size=4, seed=42)
    return DiffusionLossContext(
        models={"transformer": student},
        train_pipeline_config=Config(),
        sde_backend=None,
        device=torch.device("cpu"),
        forward_dtype=torch.float32,
        args=args,
        rollout_id=rollout,
    )


def rollout_pair(index):
    return dict(
        sample_index=index, pdd_trajectory=torch.randn(33, 4, 2), denoising_env=SimpleNamespace(pos_cond_kwargs={})
    )


def test_eight_block_updates_from_rollout_states_match_an_independent_loop():
    torch.manual_seed(12)
    student, teacher = Student(), Teacher()
    reference = copy.deepcopy(student)
    opt, refopt = (torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0) for m in (student, reference))
    ctx = context(student)
    pairs = [rollout_pair(i) for i in (2, 9)]
    expanded = expand_train_pairs(ctx, pairs)
    assert [p["pdd_start"] for p in expanded] == [s for s in range(0, 32, 4) for _ in pairs]
    for step in range(8):
        opt.zero_grad()
        refopt.zero_grad()
        for pair in expanded[step * 2 : (step + 1) * 2]:
            prepared = prepare_pdd_batch(ctx, [pair], microbatch_id=0, optim_step_idx=0)
            k = prepared.extras["selected"]
            x, sigma = prepared.latents["visual"], prepared.timesteps_for_model["visual"]
            assert prepared.extras["start"] == 4 * step <= k < 4 * step + 4
            torch.testing.assert_close(x[0], pair["pdd_trajectory"][4 * step], rtol=0, atol=0)
            # What the actor does: run the frozen teacher through the prepare hook's solver, without grad.
            with torch.no_grad():
                target = prepared.teacher_solver(lambda lat, ts: {"visual": teacher(lat["visual"], ts["visual"])})
            # The teacher starts from the rollout's state at the sampled interval.
            torch.testing.assert_close(teacher.inputs[-2][0], pair["pdd_trajectory"][k], rtol=0, atol=0)
            loss = pdd_loss_formula(
                ctx,
                [pair],
                prepared,
                new_pred={"visual": student(x, sigma).permute(1, 2, 0, 3).flatten(2)},
                old_pred=None,
                ref_pred=None,
                teacher_pred=target,
                metrics=Metrics(),
                optim_step_idx=0,
                train_output_inputs=[],
            )
            x_k = pair["pdd_trajectory"][k].unsqueeze(0)
            expected = teacher_mean_velocity(lambda s, t: teacher(s, t.expand(1)), x_k, GRID[k], GRID.diff()[k])
            refloss = interval_loss(reference(x, sigma), k, expected).sum()
            torch.testing.assert_close(loss, refloss, rtol=0, atol=0)
            (loss / 2).backward()
            (refloss / 2).backward()
        opt.step()
        refopt.step()
    assert teacher.scale.grad is None
    for a, b in zip(student.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_interval_choice_depends_only_on_rollout_lane_and_block():
    pair = dict(rollout_pair(99), pdd_start=0)

    def choose(rollout):
        return prepare_pdd_batch(context(Student(), rollout), [pair], microbatch_id=0, optim_step_idx=0).extras[
            "selected"
        ]

    chosen = choose(7)
    torch.randn(500)  # process RNG must not change a resumed rollout's choice
    assert choose(7) == chosen and {choose(r) for r in range(32)} == {0, 1, 2, 3}


def test_lora_student_trains_adapters_and_full_heads_and_checkpoints_both():
    from peft import LoraConfig, get_peft_model

    from miles.backends.fsdp_utils.checkpoint import ModelState

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(4, 4)
            self.proj_out = nn.Linear(4, 2)

    with torch.device("meta"):  # non-zero ranks build the student on meta before FSDP sharding
        model = get_peft_model(Tiny(), LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"]))
    # The loader's order: LoRA first, then the model-transform hook on the unwrapped model.
    install_student(model.get_base_model(), SimpleNamespace(pdd_num_intervals=32))
    assert all(p.is_meta for p in model.parameters())
    assert model.base_model.model.proj_out.parallel.weight.shape == (32, 2, 4)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {
        "base_model.model.to_q.lora_A.default.weight",
        "base_model.model.to_q.lora_B.default.weight",
        "base_model.model.proj_out.parallel.weight",
        "base_model.model.proj_out.parallel.bias",
    }
    state = ModelState(model, lora_only=True)
    assert {name for name in dict(model.named_parameters()) if state._saved(name)} == trainable


# --- arguments -----------------------------------------------------------------------------------------


def configured(**overrides):
    """set_default_diffusion_args -> user overrides stand -> miles_validate_args, as in parse_args."""
    args = SimpleNamespace(
        **dict.fromkeys(
            (
                "train_only use_ema ref_load teacher_lora_adapter_paths lora_adapter_paths micro_batch_size_sample "
                "diffusion_recompute_old_log_prob diffusion_kl_beta teacher_load diffusion_step_strategy_path "
                "custom_generate_function_path group_rm rollout_patch_group custom_convert_samples_to_train_data_path "
                "custom_rm_path custom_train_model_transform_path fsdp_param_dtype num_steps_per_rollout "
                "global_batch_size rollout_return_full_trajectory"
            ).split()
        ),
        diffusion_model_family="qwen_image21",
        diffusion_task="t2i",
        train_backend="fsdp",
        update_weight_target_modules=["transformer"],
        pdd_num_intervals=32,
        pdd_block_size=4,
        diffusion_height=1024,
        diffusion_width=1024,
        micro_batch_size=1,
        sequence_parallel_size=1,
        use_lora=True,
        colocate=True,
        lora_ipc_weight_sync=True,
        sglang_lora_merge_mode="dynamic",
        ref_mode="none",
        diffusion_guidance_scale=1.0,
        diffusion_true_cfg_scale=1.0,
        prompt_data="train.jsonl",
        rollout_global_dataset=True,
        n_samples_per_prompt=1,
        fsdp_master_dtype="fp32",
        diffusion_forward_dtype="bf16",
        hf_checkpoint="model",
        diffusion_num_steps=8,
        diffusion_eval_num_steps=8,
        diffusion_sde_type="ode",
        diffusion_noise_level=0,
        rollout_num_gpus_per_engine=1,
        rollout_microgroup_size=1,
        sglang_dit_precision="fp32",
        rollout_function_path="miles.rollout.sglang_rollout.generate_rollout",
        rollout_batch_size=8,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    set_pdd_defaults(args)
    args.rollout_patch_groups = [name for name in (args.rollout_patch_group or "").split(",") if name]
    validate_pdd_args(args)
    return args


def test_defaults_select_the_sglang_rollout_and_eight_updates_per_rollout():
    from miles.backends.fsdp_utils.loss_hub import LOSS_HOOKS

    args = configured()
    assert (args.global_batch_size, args.num_steps_per_rollout) == (8, 8)
    assert args.teacher_load == args.hf_checkpoint and args.fsdp_param_dtype == "fp32"
    assert args.rollout_patch_groups == ["pdd_qwen_image21"] and args.rollout_return_full_trajectory
    assert args.custom_train_model_transform_path.endswith("integration.install_student")
    hooks = LOSS_HOOKS[("pdd", "actor")]
    assert (hooks.prepare, hooks.formula, hooks.train_input) == (
        prepare_pdd_batch,
        pdd_loss_formula,
        expand_train_pairs,
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("use_lora", False),
        ("sglang_lora_merge_mode", "merge"),
        ("lora_ipc_weight_sync", False),
        ("pdd_block_size", 8),
        ("global_batch_size", 1),
        ("diffusion_guidance_scale", 4),
        ("diffusion_height", 512),
        ("sglang_dit_precision", "bf16"),
    ],
)
def test_unvalidated_modes_fail_before_gpu_load(name, value):
    with pytest.raises(ValueError):
        configured(**{name: value})
