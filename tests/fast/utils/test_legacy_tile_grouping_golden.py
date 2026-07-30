"""Golden CI: baseline_stride + 1D schedule reproduce the REAL legacy OCR tiles.

The distinctive axis here is the DP split: this replays the one real 2-GPU OCR config
(legacy_ocr_tile_grouping.json, see gen_legacy_tile_fixture.py) through
TrainDataDPSplitter("baseline_stride")  # == legacy range(rank, N, dp)
    -> build_microbatch_schedule(...)    # contiguous micro-batches
and asserts every legacy tile is reproduced cell-for-cell. The 2D tiling function is
covered separately by test_tiled_microbatch_schedule.py (no DP split there).
Pure-tensor CPU, no model forward (immune to bf16 backward non-determinism).
"""

from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import json
from pathlib import Path

from miles.utils.train_data_utils import TrainDataDPSplitter, build_microbatch_schedule

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "legacy_ocr_tile_grouping.json"


def _load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text())


def _refactored_grouping(cfg: dict):
    """Run the refactored compat pipeline and return:

    - grouping:  {(rank, optim_step, micro_batch_index): [[sample_index, sde_step], ...]}
    - mb_counts: {(rank, optim_step): num_micro_batches}
    """
    # Flat, sample-major train pairs tagged with their real (sample, sde-step)
    # identity — exactly what Flow-GRPO train-pair conversion emits, minus the heavy
    # per-cell tensors (irrelevant to *grouping*).
    pairs = [
        {"sample_index": s, "sde_step": sde} for s in range(cfg["num_samples"]) for sde in cfg["sde_step_indices"]
    ]
    shards = TrainDataDPSplitter().split_by_dp({"train_data": pairs}, cfg["dp_size"], mode="baseline_stride")

    grouping: dict[tuple[int, int, int], list[list[int]]] = {}
    mb_counts: dict[tuple[int, int], int] = {}
    for rank in range(cfg["dp_size"]):
        rank_pairs = shards[rank]["train_data"]
        num_pairs = len(rank_pairs)
        assert num_pairs % cfg["num_steps_per_rollout"] == 0, num_pairs
        schedule = build_microbatch_schedule(
            num_pairs_per_optim_step=num_pairs // cfg["num_steps_per_rollout"],
            num_optim_steps_per_rollout=cfg["num_steps_per_rollout"],
            micro_batch_size=cfg["micro_batch_size"],
        )
        for step, step_ranges in enumerate(schedule):
            mb_counts[(rank, step)] = len(step_ranges)
            for mb_idx, (lo, hi) in enumerate(step_ranges):
                grouping[(rank, step, mb_idx)] = [[p["sample_index"], p["sde_step"]] for p in rank_pairs[lo:hi]]
    return grouping, mb_counts


def test_refactored_microbatches_match_real_legacy_tiles():
    fx = _load_fixture()
    cfg = fx["meta"]["config"]
    grouping, _ = _refactored_grouping(cfg)

    assert fx["tiles"], "fixture has no tiles"
    for tile in fx["tiles"]:
        key = (tile["rank"], tile["optim_step"], tile["tile_index"])
        assert key in grouping, f"refactored pipeline produced no micro-batch at {key}"
        assert grouping[key] == tile["cells"], (
            f"group-batch mismatch at rank={key[0]} optim_step={key[1]} tile={key[2]}\n"
            f"  legacy : {tile['cells']}\n"
            f"  refactor: {grouping[key]}"
        )


def test_refactored_microbatch_count_matches_legacy_window():
    """Beyond the pinned tiles: every (rank, optim_step) must split into exactly
    the same number of micro-batches as the legacy window had tiles."""
    fx = _load_fixture()
    cfg = fx["meta"]["config"]
    expected = fx["meta"]["expected_microbatches_per_optim_step"]
    _, mb_counts = _refactored_grouping(cfg)

    assert mb_counts, "no micro-batches produced"
    for (rank, step), n in mb_counts.items():
        assert n == expected, f"rank={rank} optim_step={step}: {n} micro-batches, expected {expected}"


def test_baseline_stride_is_what_makes_it_match():
    """Sanity: with the *default* contiguous DP split (parity knob off) the very
    first pinned tile no longer matches — proving the match is due to
    baseline_stride, not a trivial identity."""
    fx = _load_fixture()
    cfg = fx["meta"]["config"]
    tile = fx["tiles"][0]

    pairs = [
        {"sample_index": s, "sde_step": sde} for s in range(cfg["num_samples"]) for sde in cfg["sde_step_indices"]
    ]
    shards = TrainDataDPSplitter().split_by_dp({"train_data": pairs}, cfg["dp_size"], mode="contiguous")
    rank_pairs = shards[tile["rank"]]["train_data"]
    schedule = build_microbatch_schedule(
        num_pairs_per_optim_step=len(rank_pairs) // cfg["num_steps_per_rollout"],
        num_optim_steps_per_rollout=cfg["num_steps_per_rollout"],
        micro_batch_size=cfg["micro_batch_size"],
    )
    lo, hi = schedule[tile["optim_step"]][tile["tile_index"]]
    contiguous_cells = [[p["sample_index"], p["sde_step"]] for p in rank_pairs[lo:hi]]
    assert contiguous_cells != tile["cells"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
