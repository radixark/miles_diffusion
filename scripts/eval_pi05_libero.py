"""Evaluate an exported π₀.₅ bundle against a running SGLang action server."""

import argparse
import hashlib
import json
import os
import platform
import time
from collections import deque
from pathlib import Path

import numpy as np

from miles.backends.fsdp_utils.models.pi05.checkpoint import read_bundle
from miles.rollout.pi05_libero import (
    SGLangLiberoPolicy,
    configure_libero,
    libero_observation,
    load_libero_init_states,
    reset_libero_env,
)

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def evaluate(args):
    if args.libero_checkout is not None:
        configure_libero(args.libero_checkout, args.output.parent / "libero-config")
    os.environ.setdefault("MUJOCO_GL", "cgl" if platform.system() == "Darwin" else "egl")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    if args.trials <= 0 or args.replan_steps <= 0 or args.wait_steps < 0:
        raise ValueError("Trials/replan steps must be positive; wait steps must be non-negative")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=False)
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    np.random.seed(args.seed)
    policy = SGLangLiberoPolicy(args.bundle, args.endpoint, seed=args.seed, num_steps=args.num_steps)
    if args.replan_steps > policy.model["action_horizon"]:
        policy.close()
        raise ValueError("--replan-steps exceeds the checkpoint's action horizon")
    metadata = read_bundle(args.bundle)
    protocol = {
        "suite": args.suite,
        "task_id": args.task_id,
        "trials": args.trials,
        "seed": args.seed,
        "seed_scheme": "seed + 1000 * task_id + trial",
        "max_steps": args.max_steps or MAX_STEPS[args.suite],
        "replan_steps": args.replan_steps,
        "wait_steps": args.wait_steps,
        "num_steps": args.num_steps,
    }
    progress_path = args.output.with_suffix(".progress.json")
    episodes = []
    if progress_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume for {progress_path}")
        saved = json.loads(progress_path.read_text())
        if saved["protocol"] != protocol or saved["bundle_metadata"] != metadata:
            raise ValueError("Cannot resume evaluation with different protocol or weights")
        episodes = saved["episodes"]
    completed = {(item["task_id"], item["trial"]) for item in episodes}
    try:
        suite = benchmark.get_benchmark_dict()[args.suite]()
        task_ids = range(suite.n_tasks) if args.task_id is None else [args.task_id]
        if args.task_id is not None and not 0 <= args.task_id < suite.n_tasks:
            raise ValueError(f"Task id must be in [0, {suite.n_tasks})")
        for task_id in task_ids:
            task = suite.get_task(task_id)
            states = load_libero_init_states(suite, task_id)
            if args.trials > len(states):
                raise ValueError(f"Task {task_id} only has {len(states)} initial states")
            env = OffScreenRenderEnv(
                bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                camera_heights=256,
                camera_widths=256,
            )
            try:
                env.seed(args.seed)
                for trial in range(args.trials):
                    if (task_id, trial) in completed:
                        continue
                    episode_seed = args.seed + 1000 * task_id + trial
                    np.random.seed(episode_seed)
                    env.seed(episode_seed)
                    policy.rng = np.random.default_rng(episode_seed)
                    reset_libero_env(env)
                    obs = env.set_init_state(states[trial])
                    for _ in range(args.wait_steps):
                        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                    plan = deque()
                    success = False
                    requests = 0
                    inference_seconds = 0.0
                    writer = None
                    if args.video_dir is not None:
                        import imageio.v2 as imageio

                        writer = imageio.get_writer(args.video_dir / f"task-{task_id}-trial-{trial}.mp4", fps=20)
                    try:
                        for _step in range(args.max_steps or MAX_STEPS[args.suite]):
                            if writer is not None:
                                writer.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                            if not plan:
                                started = time.monotonic()
                                actions = policy.infer(libero_observation(obs, task.language))
                                inference_seconds += time.monotonic() - started
                                requests += 1
                                plan.extend(actions[: args.replan_steps])
                            obs, _, success, _ = env.step(plan.popleft().tolist())
                            if success:
                                break
                    finally:
                        if writer is not None:
                            writer.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                            writer.close()
                    episodes.append(
                        {
                            "task_id": task_id,
                            "trial": trial,
                            "seed": episode_seed,
                            "task": task.language,
                            "initial_state_sha256": hashlib.sha256(np.asarray(states[trial]).tobytes()).hexdigest(),
                            "success": bool(success),
                            "steps": _step + 1,
                            "inference_requests": requests,
                            "inference_seconds": inference_seconds,
                        }
                    )
                    save_json(progress_path, {"protocol": protocol, "bundle_metadata": metadata, "episodes": episodes})
                    print(f"task={task_id} trial={trial} success={bool(success)}", flush=True)
            finally:
                env.close()
    finally:
        policy.close()
    # HTTP/model/simulation errors fail the run; they must not become fake policy failures.
    if not episodes:
        raise ValueError("LIBERO suite produced no episodes")
    report = {
        "bundle": str(args.bundle.resolve()),
        "endpoint": args.endpoint,
        "initialization": read_bundle(args.bundle).get("initialization"),
        "training_checkpoint": read_bundle(args.bundle).get("training_checkpoint"),
        "suite": args.suite,
        "protocol": protocol,
        "complete": True,
        "evaluation_scope": (
            "smoke"
            if args.max_steps is not None
            else "single_task" if args.task_id is not None else "limited_trials" if args.trials < 50 else "full_suite"
        ),
        "max_steps": args.max_steps or MAX_STEPS[args.suite],
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "num_steps": args.num_steps,
        "success_rate": sum(item["success"] for item in episodes) / len(episodes),
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Success rate: {report['success_rate']:.4f}; results: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--libero-checkout", type=Path, help="Create an isolated, noninteractive LIBERO config")
    parser.add_argument("--suite", choices=list(MAX_STEPS), default="libero_spatial")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--task-id", type=int, help="Restrict to one task for a smoke run")
    parser.add_argument("--max-steps", type=int, help="Override episode limit (marks the report as a smoke run)")
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume completed episodes from matching progress JSON")
    parser.add_argument("--video-dir", type=Path, help="Save each episode as an MP4 in a new directory")
    evaluate(parser.parse_args())
