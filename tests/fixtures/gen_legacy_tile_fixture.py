"""Golden fixture for the REAL 2-GPU OCR pipeline -> legacy_ocr_tile_grouping.json.

Distinct from gen_legacy_tile_2d_fixture: this is the ONLY golden that exercises the DP
split. It runs the full legacy dispatch end-to-end -- stride rank partition
(range(rank, N, dp)) + per-optim-step slice + tiling -- for the one real OCR config
(512 samples, dp=2, window=2, sample_mb=4, tstep_mb=2), replayed by
test_legacy_tile_grouping_golden.py via TrainDataDPSplitter("stride") +
build_microbatch_schedule (the 1D path). The 2D fixture instead cross-checks the
build_tiled_microbatch_schedule function across tiling shapes but never splits by rank.

Tiles come from executing origin/main (a48476c) _run_optim_window verbatim (_forward_tile
-> recorder, debug_skip_optimizer_step set); DP stride (rollout.py:433) and optim-step
slice reproduced from the same source.
Regenerate: python tests/fixtures/gen_legacy_tile_fixture.py
"""

from __future__ import annotations

import ast
import json
import subprocess
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

# --------------------------------------------------------------------------- #
# Real OCR 2-GPU baseline config
# --------------------------------------------------------------------------- #
LEGACY_REF = "origin/main"  # radixark/miles_diffusion @ a48476c
LEGACY_ACTOR_PATH = "miles/backends/fsdp_utils/actor.py"

DP_SIZE = 2
ROLLOUT_BATCH_SIZE = 32  # prompts
MICROGROUP_SIZE = 16  # samples per prompt (GRPO group)
NUM_SAMPLES = ROLLOUT_BATCH_SIZE * MICROGROUP_SIZE  # 512
NUM_STEPS_PER_ROLLOUT = 2
SDE_WINDOW_SIZE = 2
SAMPLE_MICROBATCH = 4
TSTEP_MICROBATCH = 2
ITER_ORDER = "sample_major"
SDE_STEP_INDICES = [3, 4]  # sde_window_range=3,5 -> [3,4]
MICRO_BATCH_SIZE = SAMPLE_MICROBATCH * TSTEP_MICROBATCH  # 8  (new-side knob)

# Representative tiles to pin: both ranks x both optim steps x first/last tile.
SELECTED_TILES = [(0, 0, 0), (0, 0, 31), (1, 0, 0), (0, 1, 0), (1, 1, 31)]


# --------------------------------------------------------------------------- #
# _chunked_indices: VERBATIM from origin/main actor.py:774 (needed by the
# live-exec'd legacy _run_optim_window namespace).
# --------------------------------------------------------------------------- #
def _chunked_indices(total: int, chunk_size: int, device: torch.device) -> list[torch.Tensor]:
    """Split range(total) into 1-D LongTensor chunks of size <= chunk_size."""
    if total <= 0:
        return []
    chunk_size = max(1, chunk_size)
    return [
        torch.arange(start, min(start + chunk_size, total), device=device, dtype=torch.long)
        for start in range(0, total, chunk_size)
    ]


def _load_legacy_run_optim_window():
    """Pull the *actual* legacy `_run_optim_window` source from origin/main and
    exec it so the real chunk/tile iteration runs (no re-implementation)."""
    src = subprocess.check_output(["git", "show", f"{LEGACY_REF}:{LEGACY_ACTOR_PATH}"], text=True)
    tree = ast.parse(src)
    func_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_optim_window":
            func_src = ast.get_source_segment(src, node)
            break
    if func_src is None:
        raise RuntimeError(f"_run_optim_window not found in {LEGACY_REF}:{LEGACY_ACTOR_PATH}")
    ns = {
        "torch": torch,
        "defaultdict": defaultdict,
        "nullcontext": nullcontext,
        "_chunked_indices": _chunked_indices,
    }
    exec(compile(func_src, "<legacy _run_optim_window>", "exec"), ns)
    return ns["_run_optim_window"]


class _TileRecorder:
    """Stands in for the actor: skips the optimizer step (so no model/scaler is
    touched) and records the (sample_indices, tstep_indices) of every tile."""

    def __init__(self):
        self.args = SimpleNamespace(debug_skip_optimizer_step=True)
        self.tiles: list[tuple[list[int], list[int]]] = []

    def _forward_tile(self, *, sample_indices, tstep_indices, **_kwargs):
        self.tiles.append((sample_indices.tolist(), tstep_indices.tolist()))
        return torch.zeros(())


def _legacy_tiles() -> dict[tuple[int, int, int], list[list[int]]]:
    run_optim_window = _load_legacy_run_optim_window()
    grouping: dict[tuple[int, int, int], list[list[int]]] = {}
    for rank in range(DP_SIZE):
        # DP stride split (rollout.py:433): rank r owns samples r, r+dp, r+2dp, ...
        rank_samples = list(range(rank, NUM_SAMPLES, DP_SIZE))
        num_rollout_samples = len(rank_samples)
        num_samples_per_optim_step = num_rollout_samples // NUM_STEPS_PER_ROLLOUT
        for step_id in range(NUM_STEPS_PER_ROLLOUT):
            traj_start = step_id * num_samples_per_optim_step
            traj_end = min(num_rollout_samples, traj_start + num_samples_per_optim_step)
            window_global = rank_samples[traj_start:traj_end]
            num_samples_in_window = len(window_global)

            sample_microbatch = min(SAMPLE_MICROBATCH, num_samples_in_window)
            tstep_microbatch = min(TSTEP_MICROBATCH, SDE_WINDOW_SIZE)

            rec = _TileRecorder()
            grids = {
                "latents": torch.zeros(num_samples_in_window, SDE_WINDOW_SIZE, 1),
                "num_samples_in_window": num_samples_in_window,
                "sde_window_size": SDE_WINDOW_SIZE,
            }
            run_optim_window(
                rec,
                grids=grids,
                sample_microbatch=sample_microbatch,
                tstep_microbatch=tstep_microbatch,
                iter_order=ITER_ORDER,
                use_cfg=False,
                guidance_scale=0.0,
                true_cfg_scale=None,
                clip_range=0.0,
                kl_beta=0.0,
                noise_level=0.0,
                num_train_timesteps=1000,
            )
            for tile_idx, (sample_pos, tstep_pos) in enumerate(rec.tiles):
                # _forward_tile reshape is sample-major: s outer, t inner.
                cells = [[window_global[sp], SDE_STEP_INDICES[tp]] for sp in sample_pos for tp in tstep_pos]
                grouping[(rank, step_id, tile_idx)] = cells
    return grouping


def main() -> None:
    grouping = _legacy_tiles()

    tiles_per_window = {
        len([k for k in grouping if k[0] == r and k[1] == s])
        for r in range(DP_SIZE)
        for s in range(NUM_STEPS_PER_ROLLOUT)
    }
    assert len(tiles_per_window) == 1, tiles_per_window
    expected_microbatches_per_optim_step = tiles_per_window.pop()

    tiles = []
    for rank, step, tile_idx in SELECTED_TILES:
        if (rank, step, tile_idx) not in grouping:
            raise KeyError(f"selected tile {(rank, step, tile_idx)} not produced by legacy run")
        tiles.append(
            {
                "rank": rank,
                "optim_step": step,
                "tile_index": tile_idx,
                "cells": grouping[(rank, step, tile_idx)],
            }
        )

    out = {
        "meta": {
            "description": (
                "Real legacy TrainRayActor grid-tile group-batch membership, produced by "
                "executing origin/main _run_optim_window verbatim with the 2-GPU OCR baseline "
                "config. Each cell is [sample_index, sde_step]. The refactored compat pipeline "
                "(stride DP split + build_microbatch_schedule, micro_batch_size="
                f"{MICRO_BATCH_SIZE}) must reproduce every tile exactly."
            ),
            "legacy_provenance": {
                "ref": "radixark/miles_diffusion@a48476c (origin/main)",
                "dp_stride": "miles/ray/rollout.py:433",
                "tile_iteration": "miles/backends/fsdp_utils/actor.py:_run_optim_window (executed live)",
                "cell_order": "miles/backends/fsdp_utils/actor.py:_forward_tile reshape (sample-major)",
            },
            "config": {
                "dp_size": DP_SIZE,
                "num_samples": NUM_SAMPLES,
                "num_steps_per_rollout": NUM_STEPS_PER_ROLLOUT,
                "sde_window_size": SDE_WINDOW_SIZE,
                "sde_step_indices": SDE_STEP_INDICES,
                "sample_microbatch": SAMPLE_MICROBATCH,
                "tstep_microbatch": TSTEP_MICROBATCH,
                "iter_order": ITER_ORDER,
                "micro_batch_size": MICRO_BATCH_SIZE,
            },
            "expected_microbatches_per_optim_step": expected_microbatches_per_optim_step,
        },
        "tiles": tiles,
    }

    out_path = Path(__file__).parent / "legacy_ocr_tile_grouping.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}  ({len(tiles)} tiles, {expected_microbatches_per_optim_step} mb/optim-step)")


if __name__ == "__main__":
    main()
