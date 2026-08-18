"""E2E: Qwen-Image PickScore GRPO (flow_grpo-aligned), 5-GPU (4 colocated FSDP
DP=4 + sglang rollout engines, 1 dedicated pickscore GPU) — runs the example
script's real configuration and checks its metric series against the registered
standard (tests/ci/fixtures/e2e_standards/). Runs with --deterministic-mode, so
every metric is compared strictly, bit for bit.

Only --num-rollout is cut down, 400 -> 2: one weight-sync round trip is enough
to catch drift in the post-update rollout and the second optimizer step.

The train/log_prob_* and train/model_output_* series record the
train<->rollout residual under the qwen_image patch group +
--sglang-lora-merge-mode dynamic (the recipe runs without
--diffusion-recompute-old-log-prob, so old log-probs come from the engine).
Exact zero is not yet expected for Qwen-Image; the strict comparison pins the
residual at its recorded magnitude, so a numerics change on either side —
engine kernels, patch group, LoRA path — moves these series and fails loudly.
"""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=1200,
    suite="stage-c-5-gpu-h200",
    script="scripts/run_diffusion_grpo_pickscore_5gpu_flowgrpo_aligned.py",
    args=["--num-rollout", "2"],
    labels=["e2e"],
    metrics=[
        "rollout/reward/raw_num_samples",
        "rollout/reward/raw_mean",
        "rollout/reward/raw_median",
        "rollout/reward/raw_std",
        "train/log_prob_old_idx_0",
        "train/log_prob_new_idx_0",
        "train/log_prob_mean_abs_diff",
        "train/model_output_mean_abs_diff",
        "train/model_output_rel_max",
        "train/grad_norm",
    ],
)
