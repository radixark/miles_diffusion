"""Trainer side: eight block updates from rollout states match an independent optimizer loop."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from miles.backends.fsdp_utils.loss_hub.pdd import pdd_loss_formula, prepare_pdd_batch
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext
from miles.distillation.integration import expand_trajectories, install_student
from miles.distillation.pdd import ParallelLinear, pdd_loss


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


def context(student, teacher, rollout=0):
    return DiffusionLossContext(
        models={"transformer": student},
        teacher_models={"transformer": teacher},
        train_pipeline_config=Config(),
        sde_backend=None,
        device=torch.device("cpu"),
        forward_dtype=torch.float32,
        args=SimpleNamespace(pdd_num_intervals=32, pdd_block_size=4, diffusion_height=32, seed=42),
        rollout_id=rollout,
        algorithm_state={"grid": torch.linspace(1, 0, 33)},
    )


def raw_forward(model, prepared):
    pred = model(prepared.latents["visual"], prepared.timesteps_for_model["visual"])
    return {"visual": pred.permute(1, 2, 0, 3).flatten(2)}


def rollout_pair(index):
    return dict(
        sample_index=index, pdd_trajectory=torch.randn(9, 4, 2), denoising_env=SimpleNamespace(pos_cond_kwargs={})
    )


def test_eight_block_updates_from_rollout_states_match_reference():
    torch.manual_seed(12)
    student = Model()
    reference = copy.deepcopy(student)
    teacher = Teacher()
    ctx = context(student, teacher)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0)
    refopt = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=0)
    pairs = [rollout_pair(i) for i in (2, 9)]
    expanded = expand_trajectories(pairs, intervals=32, block_size=4)
    for step in range(8):
        opt.zero_grad()
        refopt.zero_grad()
        for pair in expanded[step * 2 : (step + 1) * 2]:
            prepared = prepare_pdd_batch(ctx, [pair])
            start, k, grid = (prepared.extras[v] for v in ("start", "selected", "grid"))
            x = prepared.latents["visual"]
            torch.testing.assert_close(x[0], pair["pdd_trajectory"][step], rtol=0, atol=0)
            assert start == 4 * step and not x.requires_grad
            predictions = reference(x, prepared.timesteps_for_model["visual"])
            expected = pdd_loss(
                predictions, x, grid, start, k, lambda state, sigma: teacher(state.bfloat16().float(), sigma.expand(1))
            )
            loss = pdd_loss_formula(
                ctx,
                [pair],
                prepared,
                new_pred=raw_forward(student, prepared),
                old_pred=None,
                ref_pred=None,
                metrics=Metrics(),
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
    student, teacher = Model(), Teacher()
    pair = dict(rollout_pair(99), pdd_start=0)
    chosen = prepare_pdd_batch(context(student, teacher, 7), [pair]).extras["selected"]
    torch.randn(500)  # unrelated process RNG must not affect a resumed rollout
    assert prepare_pdd_batch(context(student, teacher, 7), [pair]).extras["selected"] == chosen
    choices = {prepare_pdd_batch(context(student, teacher, r), [pair]).extras["selected"] for r in range(32)}
    assert choices == {0, 1, 2, 3}


def test_heads_installed_on_meta_before_sharding():
    with torch.device("meta"):
        model = nn.Module()
        model.proj_out = nn.Linear(4, 2)
    model.config = SimpleNamespace(patch_size=1, in_channels=2)
    install_student(model, SimpleNamespace(pdd_num_intervals=32))
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
        use_lora=False,
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
        no_save_optim=False,
        no_load_optim=False,
        no_load_rng=False,
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
        rollout_patch_groups=None,
        rollout_function_path="miles.rollout.sglang_rollout.generate_rollout",
        custom_convert_samples_to_train_data_path=None,
        custom_rm_path=None,
        custom_prepare_train_batch_path=None,
        custom_loss_function_path=None,
        num_steps_per_rollout=None,
        global_batch_size=None,
        rollout_batch_size=8,
    )


def test_defaults_select_sglang_rollout_and_eight_updates():
    from miles.distillation.integration import configure_args

    args = valid_args()
    configure_args(args)
    assert args.global_batch_size == 8 and args.num_steps_per_rollout == 8
    assert args.teacher_load == args.hf_checkpoint
    assert args.rollout_function_path == "miles.rollout.sglang_diffusion_rollout.generate_rollout"
    assert args.rollout_patch_groups == ["pdd_qwen_image21"] and args.rollout_return_full_trajectory
    assert args.custom_loss_function_path.endswith("pdd.pdd_loss_formula")


@pytest.mark.parametrize(
    "name,value",
    [
        ("sequence_parallel_size", 2),
        ("use_lora", True),
        ("train_only", True),
        ("pdd_block_size", 8),
        ("diffusion_guidance_scale", 4),
        ("no_load_optim", True),
        ("global_batch_size", 1),
        ("num_steps_per_rollout", 1),
        ("diffusion_height", 512),
        ("diffusion_num_steps", 40),
        ("diffusion_noise_level", 0.1),
        ("rollout_num_gpus_per_engine", 2),
        ("rollout_microgroup_size", 2),
        ("sglang_dit_precision", "bf16"),
        ("rollout_patch_groups", ["qwen_image21"]),
    ],
)
def test_unvalidated_modes_fail_before_gpu_load(name, value):
    from miles.distillation.integration import configure_args

    args = valid_args()
    setattr(args, name, value)
    with pytest.raises(ValueError):
        configure_args(args)
