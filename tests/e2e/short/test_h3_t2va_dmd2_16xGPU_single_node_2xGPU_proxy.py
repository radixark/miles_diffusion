"""E2E: MiniMax H3 t2va 4-step DMD2 (data-free), 2-GPU colocate (FSDP DP=2 over the
full-parameter student, critic and teacher, one tp2 sglang rollout engine, no reward),
2 rollouts of 4 prompts: 2 critic steps of 2 samples, then 1 student step on the last 2
— runs the recipe script itself with its single-node knobs and checks its metric series
against the registered standard (tests/ci/fixtures/e2e_standards/). The three 31B models
do not fit two GPUs with their optimizers, so the recipe keeps parameters, gradients and
optimizer states in host memory (--fsdp-cpu-offload, about 1.1 TB) and steps on the CPU;
the master stays fp32, as H3's fp32-pinned modules rule out a bf16 master for
full-parameter training. Runs with --deterministic-mode, so every metric is compared
strictly, bit for bit. The only e2e exercising the multi-trainer-group driver, the
critic -> actor hand-off and the rollout stopped at a random step. Most of its time is the
three 62 GB checkpoints materializing into pinned host shards and the CPU optimizer steps,
so est_time lifts run_suite's 1800 s per-file cap (effective timeout = 1.25 x est_time)."""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=3300,
    suite="stage-c-3-gpu-h200",
    labels=["e2e"],
    script="scripts/run_diffusion_dmd2_h3_t2va_16gpu_multinode.py",
    args=[
        "--num-nodes",
        "1",
        "--num-gpus-per-node",
        "2",
        "--num-rollout",
        "2",
        "--rollout-batch-size",
        "4",
        "--critic-steps-per-rollout",
        "2",
        "--eval-interval",
        "0",
        "--save-interval",
        "0",
        "--duration-seconds",
        "4",
        "--fsdp-cpu-offload",
        "--extra-args",
        "--deterministic-mode",
    ],
    metrics=[
        "train/critic-loss",
        "train/critic-grad_norm",
        "train/grad_norm",
        "train/dmd2_grad_abs_mean",
        "train/dmd2_score_gap_abs_mean",
        "train/dmd2_teacher_dist_abs_mean",
        "train/dmd2_x0_mean",
        "train/dmd2_x0_std",
        "train/x0_mean_abs_diff",
    ],
)
