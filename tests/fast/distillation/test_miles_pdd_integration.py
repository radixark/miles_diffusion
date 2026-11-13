"""Trainer side: eight block updates from rollout states match an independent optimizer loop.

The test plays the actor: it runs the prepare hook's teacher solver with its own teacher forward,
then hands the result to the formula as ``teacher_pred``.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from miles.backends.fsdp_utils.loss_hub.pdd import pdd_loss_formula, prepare_pdd_batch
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext
from miles.distillation import integration
from miles.distillation.integration import expand_train_pairs, expand_trajectories, install_student
from miles.distillation.pdd import ParallelLinear, pdd_loss


@pytest.fixture(autouse=True)
def linear_grid(monkeypatch):
    monkeypatch.setattr(integration, "make_grid", lambda *args: torch.linspace(1, 0, 33))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_channels=2, patch_size=1)
        self.base = nn.Linear(2, 2)
        self.head = ParallelLinear(nn.Linear(2, 2), 32)

    def forward(self, x, sigma):
        return self.head(self.base(x) + sigma.reshape(-1, 1, 1))


class Config:
    def compute_noise_pred(self, *, model, latents_input, timesteps_input, **kw):
        return {"visual": model(latents_input["visual"].float(), timesteps_input["visual"])}

    def prepare_cond_kwargs(self, cond, device):
        return cond


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.calls = []

    def forward(self, x, sigma):
        self.calls.append((x.detach().clone(), sigma.detach().clone()))
        return self.scale * x + sigma.reshape(-1, 1, 1)


class Metrics:
    def emit_mean(self, *args, **kwargs):
        pass


def context(student, rollout=0):
    return DiffusionLossContext(
        models={"transformer": student},
        train_pipeline_config=Config(),
        sde_backend=None,
        device=torch.device("cpu"),
        forward_dtype=torch.float32,
        args=SimpleNamespace(
            diffusion_model_family="qwen_image21",
            hf_checkpoint="unused",
            pdd_num_intervals=32,
            pdd_block_size=4,
            diffusion_height=32,
            seed=42,
        ),
        rollout_id=rollout,
    )


def teacher_target(teacher, prepared):
    """What the actor does with ``prepared.teacher_solver``, without grad."""
    with torch.no_grad():
        return prepared.teacher_solver(
            lambda latents, timesteps: {"visual": teacher(latents["visual"], timesteps["visual"])}
        )


def raw_forward(model, prepared):
    pred = model(prepared.latents["visual"], prepared.timesteps_for_model["visual"])
    return {"visual": pred.permute(1, 2, 0, 3).flatten(2)}


def rollout_pair(index):
    return dict(
        sample_index=index,
        pdd_trajectory={"visual": torch.randn(33, 4, 2)},
        denoising_env=SimpleNamespace(pos_cond_kwargs={}),
    )


def test_eight_block_updates_from_rollout_states_match_reference():
    torch.manual_seed(12)
    student = Model()
    reference = copy.deepcopy(student)
    teacher = Teacher()
    ctx = context(student)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0)
    refopt = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=0)
    pairs = [rollout_pair(i) for i in (2, 9)]
    expanded = expand_train_pairs(ctx, pairs)
    for step in range(8):
        opt.zero_grad()
        refopt.zero_grad()
        for pair in expanded[step * 2 : (step + 1) * 2]:
            prepared = prepare_pdd_batch(ctx, [pair], microbatch_id=0, optim_step_idx=0)
            start, k, grids = (prepared.extras[v] for v in ("start", "selected", "grids"))
            x = prepared.latents["visual"]
            torch.testing.assert_close(x[0], pair["pdd_trajectory"]["visual"][4 * step], rtol=0, atol=0)
            assert start == 4 * step and not x.requires_grad
            predictions = reference(x, prepared.timesteps_for_model["visual"])
            # The teacher starts from the rollout's state at the sampled interval.
            x_selected = pair["pdd_trajectory"]["visual"][k].unsqueeze(0)
            expected = pdd_loss(
                predictions,
                x,
                grids["visual"],
                start,
                k,
                lambda state, sigma: teacher(state, sigma.expand(1)),
                x_selected=x_selected,
            )
            calls = len(teacher.calls)
            target = teacher_target(teacher, prepared)
            torch.testing.assert_close(teacher.calls[calls][0], x_selected, rtol=0, atol=0)
            loss = pdd_loss_formula(
                ctx,
                [pair],
                prepared,
                new_pred=raw_forward(student, prepared),
                old_pred=None,
                ref_pred=None,
                teacher_pred=target,
                metrics=Metrics(),
                optim_step_idx=0,
                train_output_inputs=[],
            )
            torch.testing.assert_close(loss, expected.sum(), rtol=0, atol=0)
            (loss / 2).backward()
            (expected.sum() / 2).backward()
        for a, b in zip(student.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        opt.step()
        refopt.step()
    assert teacher.scale.grad is None
    for a, b in zip(student.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_interval_choice_depends_only_on_rollout_lane_and_block():
    student = Model()
    pair = dict(rollout_pair(99), pdd_start=0)
    chosen = prepare_pdd_batch(context(student, 7), [pair], microbatch_id=0, optim_step_idx=0).extras["selected"]
    torch.randn(500)  # unrelated process RNG must not affect a resumed rollout
    assert (
        prepare_pdd_batch(context(student, 7), [pair], microbatch_id=0, optim_step_idx=0).extras["selected"] == chosen
    )
    choices = {
        prepare_pdd_batch(context(student, r), [pair], microbatch_id=0, optim_step_idx=0).extras["selected"]
        for r in range(32)
    }
    assert choices == {0, 1, 2, 3}


def test_heads_installed_on_meta_before_sharding():
    with torch.device("meta"):
        model = nn.Module()
        model.proj_out = nn.Linear(4, 2)
    model.config = SimpleNamespace(patch_size=1, in_channels=2)
    install_student(model, SimpleNamespace(diffusion_model_family="qwen_image21", pdd_num_intervals=32))
    assert all(p.is_meta for p in model.parameters())
    assert model.proj_out.parallel.weight.shape == (32, 2, 4)


def test_dp_split_precedes_trajectory_expansion():
    from miles.utils.train_data_utils import TrainDataDPSplitter

    pairs = [{"sample_index": i} for i in range(8)]
    shards = TrainDataDPSplitter().split_by_dp({"train_data": pairs}, 2)
    for shard in shards:
        expanded = expand_trajectories(shard["train_data"], intervals=32, block_size=4)
        for start in range(8):
            window = expanded[start * 4 : (start + 1) * 4]
            assert [p["sample_index"] for p in window] == [p["sample_index"] for p in shard["train_data"]]
            assert all(p["pdd_start"] == start * 4 for p in window)
    with pytest.raises(ValueError, match="distinct lane"):
        expand_trajectories([pairs[0], pairs[0]], intervals=32, block_size=4)


def valid_args():
    return SimpleNamespace(
        diffusion_model_family="qwen_image21",
        diffusion_task="t2i",
        train_only=False,
        train_backend="fsdp",
        update_weight_target_modules=["transformer"],
        pdd_num_intervals=32,
        pdd_block_size=4,
        diffusion_height=1024,
        diffusion_width=1024,
        micro_batch_size=1,
        micro_batch_size_sample=None,
        sequence_parallel_size=1,
        use_lora=True,
        colocate=True,
        lora_ipc_weight_sync=True,
        sglang_lora_merge_mode="dynamic",
        use_ema=False,
        ref_load=None,
        ref_mode="none",
        teacher_lora_adapter_paths=None,
        lora_adapter_paths=None,
        diffusion_recompute_old_log_prob=False,
        diffusion_kl_beta=0,
        diffusion_guidance_scale=1.0,
        diffusion_true_cfg_scale=1.0,
        prompt_data="train.jsonl",
        rollout_global_dataset=True,
        n_samples_per_prompt=1,
        fsdp_master_dtype="fp32",
        diffusion_forward_dtype="bf16",
        teacher_load=None,
        hf_checkpoint="model",
        diffusion_num_steps=8,
        diffusion_eval_num_steps=8,
        diffusion_sde_type="ode",
        diffusion_noise_level=0,
        rollout_num_gpus_per_engine=1,
        rollout_microgroup_size=1,
        sglang_dit_precision="fp32",
        diffusion_step_strategy_path=None,
        custom_generate_function_path=None,
        group_rm=False,
        rollout_patch_group=None,
        rollout_function_path="miles.rollout.sglang_rollout.generate_rollout",
        custom_convert_samples_to_train_data_path=None,
        custom_rm_path=None,
        custom_train_model_transform_path=None,
        fsdp_param_dtype=None,
        rollout_return_full_trajectory=False,
        num_steps_per_rollout=None,
        global_batch_size=None,
        rollout_batch_size=8,
    )


def configure(args, **overrides):
    """set_default_diffusion_args -> user overrides stand -> miles_validate_args, as in parse_args."""
    from miles.distillation.integration import set_pdd_defaults, validate_pdd_args

    for name, value in overrides.items():
        setattr(args, name, value)
    set_pdd_defaults(args)
    args.rollout_patch_groups = [name for name in (args.rollout_patch_group or "").split(",") if name]
    validate_pdd_args(args)


def test_defaults_select_sglang_rollout_and_eight_updates():
    args = valid_args()
    configure(args)
    assert args.global_batch_size == 8 and args.num_steps_per_rollout == 8
    assert args.teacher_load == args.hf_checkpoint
    assert args.rollout_function_path == "miles.rollout.sglang_diffusion_rollout.generate_rollout"
    assert args.rollout_patch_groups == ["pdd_qwen_image21"] and args.rollout_return_full_trajectory
    from miles.backends.fsdp_utils.loss_hub import LOSS_HOOKS

    hooks = LOSS_HOOKS[("pdd", "actor")]
    assert (hooks.prepare, hooks.formula, hooks.train_input) == (
        prepare_pdd_batch,
        pdd_loss_formula,
        expand_train_pairs,
    )
    assert args.custom_train_model_transform_path.endswith("integration.install_student")
    assert args.fsdp_param_dtype == "fp32"


@pytest.mark.parametrize(
    "name,value",
    [
        ("sequence_parallel_size", 2),
        ("use_lora", False),
        ("lora_ipc_weight_sync", False),
        ("sglang_lora_merge_mode", "merge"),
        ("train_only", True),
        ("pdd_block_size", 8),
        ("diffusion_guidance_scale", 4),
        ("custom_rm_path", "miles.rollout.rm_hub.other"),
        ("teacher_load", "other-model"),
        ("global_batch_size", 1),
        ("num_steps_per_rollout", 1),
        ("diffusion_height", 512),
        ("diffusion_num_steps", 40),
        ("diffusion_noise_level", 0.1),
        ("rollout_num_gpus_per_engine", 2),
        ("rollout_microgroup_size", 2),
        ("sglang_dit_precision", "bf16"),
        ("rollout_patch_group", "qwen_image21"),
    ],
)
def test_unvalidated_modes_fail_before_gpu_load(name, value):
    with pytest.raises(ValueError):
        configure(valid_args(), **{name: value})


def test_lora_student_trains_adapters_and_full_heads_and_checkpoints_both():
    from peft import LoraConfig, get_peft_model

    from miles.backends.fsdp_utils.checkpoint import ModelState

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(4, 4)
            self.proj_out = nn.Linear(4, 2)

    # The loader's order: LoRA first, then the model-transform hook on the unwrapped model.
    model = get_peft_model(Tiny(), LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"]))
    install_student(
        model.get_base_model(), SimpleNamespace(diffusion_model_family="qwen_image21", pdd_num_intervals=32)
    )
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {
        "base_model.model.to_q.lora_A.default.weight",
        "base_model.model.to_q.lora_B.default.weight",
        "base_model.model.proj_out.parallel.weight",
        "base_model.model.proj_out.parallel.bias",
    }
    state = ModelState(model, lora_only=True)
    assert {name for name in dict(model.named_parameters()) if state._saved(name)} == trainable
