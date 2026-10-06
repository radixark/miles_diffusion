"""E2E: MiniMax H3 t2va LoRA SFT, 2-GPU FSDP DP=2 with the colocated encode pool,
2 rollouts of 4 WISA clips (2 optimizer steps each) — runs the example script itself
and checks its metric series against the registered standard
(tests/ci/fixtures/e2e_standards/). Supervises both streams, so the loss covers the
soundtrack's audio rows as well as the video; the recipe itself supervises video only.
The bf16 master leaves room on each GPU for the encoders that share it. Runs with
--deterministic-mode, so every metric is compared strictly, bit for bit. The only
e2e exercising the SFT encode -> cache -> train path."""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=3000,
    suite="stage-c-3-gpu-h200",
    labels=["e2e"],
    script="scripts/run_diffusion_sft_h3_t2va.py",
    args=[
        "--num-gpus",
        "2",
        "--extra-args",
        "--num-rollout 2 --rollout-batch-size 4 --num-steps-per-rollout 2 "
        "--fsdp-supervised-streams visual,audio "
        "--fsdp-master-dtype bf16 --fsdp-reduce-dtype bf16 "
        "--deterministic-mode",
    ],
    metrics=[
        "train/loss",
        "train/grad_norm",
    ],
)
