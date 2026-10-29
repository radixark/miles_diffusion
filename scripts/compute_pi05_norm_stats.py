"""Compute OpenPI LIBERO statistics from all local demonstration states/actions.

Read only numeric Parquet columns, avoiding image decoding. LIBERO already stores
delta actions. Chunk indices and episode-end padding match LeRobot's 10 Hz sampler.
All frames are included, including the final incomplete statistics batch.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def action_windows(actions, horizon):
    if horizon <= 0 or len(actions) == 0:
        raise ValueError("Positive horizon and a nonempty episode are required")
    return actions[np.minimum(np.arange(len(actions))[:, None] + np.arange(horizon), len(actions) - 1)]


def main(args):
    import pyarrow.parquet as pq
    from openpi.shared import normalize

    if args.output.exists():
        raise FileExistsError(args.output)
    info = json.loads((args.dataset / "meta/info.json").read_text())
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    episodes = [json.loads(line) for line in (args.dataset / "meta/episodes.jsonl").read_text().splitlines()]
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Incomplete episode metadata")
    frames = 0
    for i, episode in enumerate(episodes):
        index = episode["episode_index"]
        path = args.dataset / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
        table = pq.read_table(path, columns=["state", "actions", "episode_index", "frame_index"])
        if table.num_rows != episode["length"]:
            raise ValueError(f"Incomplete episode: {path}")
        states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        if states.shape != (table.num_rows, 8) or actions.shape != (table.num_rows, 7):
            raise ValueError(f"Unexpected LIBERO dimensions: {path}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Non-finite demonstration values: {path}")
        np.testing.assert_array_equal(table["episode_index"].to_numpy(), index)
        np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(table.num_rows))
        stats["state"].update(states)
        stats["actions"].update(action_windows(actions, args.horizon))
        frames += table.num_rows
        if (i + 1) % 100 == 0:
            print(f"Statistics: {i + 1}/{len(episodes)} episodes, {frames} frames", flush=True)
    if frames != info["total_frames"]:
        raise ValueError("Frame count differs from dataset metadata")
    normalize.save(args.output, {key: value.get_statistics() for key, value in stats.items()})
    (args.output / "coverage.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset.resolve()),
                "episodes": len(episodes),
                "frames": frames,
                "tasks": info["total_tasks"],
                "action_horizon": args.horizon,
                "method": "OpenPI RunningStats over all frames and LeRobot episode-padded action chunks",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote full-dataset statistics: {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    main(parser.parse_args())
