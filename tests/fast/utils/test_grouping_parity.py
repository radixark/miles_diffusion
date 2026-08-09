"""Strong, deterministic parity tests for the refactored train-data grouping.

These pin the refactored group-batch pipeline against the *legacy*
TrainRayActor grouping at the **data-assembly layer** (everything before the DiT
forward). They are pure-tensor, CPU-only, and bit-exact — deliberately avoiding
any model forward/backward so they are NOT polluted by the bf16 attention-backward
non-determinism that makes end-to-end training comparison unreliable
(see determinism/BACKWARD_NONDETERMINISM.md).

Contract under test: with the parity flags
``--train-dp-split-mode stride`` + ``--diffusion-train-cond-pad-window``,
the refactored pipeline feeds each DP rank / each micro-batch the **same tensors in
the same order** as the legacy grid-based path.

Layers:
  L1  DP split            — stride == legacy range(rank, N, dp), and only
                            contiguous keeps a rollout microgroup whole on one rank
  L2  Converter           — flat train-pairs == direct sde-indexed trajectory data
  L3  Cond window padding — per-microbatch collate(pad_to_len=window_max)
                            == legacy window-collate-then-tile-slice

Run:  python -m pytest test_grouping_parity.py -q     (or  python test_grouping_parity.py)
"""

from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import torch

from miles.ray.data_conversion_hub.flow_grpo import expand_samples_to_train_pairs
from miles.utils.train_data_utils import TrainDataDPSplitter, scheduler_meta_from_rollout


# --------------------------------------------------------------------------------------
# L1 — DP split parity: stride reproduces legacy range(rank, N, dp_size)
# --------------------------------------------------------------------------------------
def _legacy_dp_partition(num_samples: int, dp_size: int):
    return [list(range(r, num_samples, dp_size)) for r in range(dp_size)]


def _make_flat_pairs(num_samples: int, pairs_per_sample: int):
    return {
        "train_data": [{"sample_index": s, "tag": (s, t)} for s in range(num_samples) for t in range(pairs_per_sample)]
    }


def _rank_sample_order(shard):
    seen = []
    for pair in shard["train_data"]:
        if not seen or seen[-1] != pair["sample_index"]:
            seen.append(pair["sample_index"])
    return seen


def test_l1_dp_split_stride_matches_legacy():
    splitter = TrainDataDPSplitter()
    for num_samples, ppp, dp in [(256, 2, 2), (256, 2, 4), (16, 1, 2), (12, 3, 3)]:
        data = _make_flat_pairs(num_samples, ppp)
        shards = splitter.split_by_dp(data, dp, mode="stride")
        expected = _legacy_dp_partition(num_samples, dp)
        for r in range(dp):
            assert _rank_sample_order(shards[r]) == expected[r], (num_samples, dp, r)
        sizes = {len(s["train_data"]) for s in shards}
        assert len(sizes) == 1  # equal shards
        total = sum(len(s["train_data"]) for s in shards)
        assert total == num_samples * ppp


def test_l1_contiguous_differs_from_stride():
    splitter = TrainDataDPSplitter()
    data = _make_flat_pairs(256, 2)
    cont = splitter.split_by_dp(data, 2, mode="contiguous")
    strd = splitter.split_by_dp(data, 2, mode="stride")
    assert _rank_sample_order(cont[0]) != _rank_sample_order(strd[0])


def test_l1_only_contiguous_keeps_rollout_microgroups_whole():
    """A micro-batch can only reproduce a rollout forward if its samples reached one rank.

    8 samples generated as 4 microgroups of 2, split over 2 DP ranks:

        microgroups   [0 1] [2 3] [4 5] [6 7]
        contiguous    rank0 = 0 1 2 3   -> microgroups {0,1} intact
        stride        rank0 = 0 2 4 6   -> one sample from every microgroup
    """
    splitter = TrainDataDPSplitter()
    data = _make_flat_pairs(num_samples=8, pairs_per_sample=1)
    microgroups = [{0, 1}, {2, 3}, {4, 5}, {6, 7}]

    contiguous = set(_rank_sample_order(splitter.split_by_dp(data, 2, mode="contiguous")[0]))
    stride = set(_rank_sample_order(splitter.split_by_dp(data, 2, mode="stride")[0]))

    assert sum(group <= contiguous for group in microgroups) == 2
    assert sum(group <= stride for group in microgroups) == 0


# --------------------------------------------------------------------------------------
# L2 — Converter: flat train-pairs == direct sde-indexed trajectory data
# --------------------------------------------------------------------------------------
def _mk_sample(index: int, num_steps: int, sde_idx, chan: int = 4, with_debug=True, with_sigmas=True):
    g = torch.Generator().manual_seed(100 + index)
    latents = torch.randn(num_steps + 1, chan, generator=g)  # (T+1, C)
    timesteps = torch.arange(num_steps, dtype=torch.float32) + 0.5  # (T,)
    # Scheduler sigmas are SHARED across all samples in a batch (one scheduler), so use a
    # fixed seed (not the per-sample one) -- the converter now verifies this; a per-sample
    # seed here would (correctly) raise.
    sigmas = torch.randn(num_steps + 1, generator=torch.Generator().manual_seed(7)) if with_sigmas else None
    traj = SimpleNamespace(latents=latents, timesteps=timesteps, sigmas=sigmas)
    rollout_log_probs = torch.randn(num_steps, generator=g)  # (T,)
    dbg = None
    if with_debug:
        dbg = SimpleNamespace(
            rollout_variance_noises=torch.randn(num_steps, chan, generator=g),
            rollout_prev_sample_means=torch.randn(num_steps, chan, generator=g),
            rollout_noise_std_devs=torch.randn(num_steps, chan, generator=g),
            rollout_model_outputs=torch.randn(num_steps, chan, generator=g),
        )
    return SimpleNamespace(
        index=index,
        prompt=f"prompt-{index}",
        dit_trajectory=traj,
        denoising_env=SimpleNamespace(tag=index),
        rollout_log_probs=rollout_log_probs,
        train_metadata={"sde_step_indices": list(sde_idx)},
        rollout_debug_tensors=dbg,
    )


def test_l2_converter_pairs_match_direct_indexing():
    T, sde = 6, [1, 3, 4]
    samples = [_mk_sample(i, T, sde) for i in range(3)]
    rewards = [0.1, 0.2, 0.3]
    raw_rewards = [0.4, 0.5, 0.6]

    out = expand_samples_to_train_pairs(None, samples, rewards, raw_rewards)
    pairs = out["train_data"]

    # count + sample-major ordering + scheduler meta from the first trajectory
    assert len(pairs) == len(samples) * len(sde)
    assert torch.equal(out["scheduler_timesteps"], samples[0].dit_trajectory.timesteps.float())
    assert torch.equal(out["scheduler_sigmas"], samples[0].dit_trajectory.sigmas.float())

    k = 0
    for si, s in enumerate(samples):
        all_lat = s.dit_trajectory.latents.float()
        lat, nxt = all_lat[:-1], all_lat[1:]
        ts = s.dit_trajectory.timesteps.float()
        rlp = s.rollout_log_probs.float()
        for _t_pos, idx in enumerate(sde):
            p = pairs[k]
            k += 1
            # per-sample fields
            assert p["sample_index"] == s.index
            assert p["prompt"] == s.prompt
            assert p["advantage"] == float(rewards[si])
            assert p["raw_reward"] == float(raw_rewards[si])
            assert p["denoising_env"] is s.denoising_env
            # per-timestep fields == direct sde-indexed trajectory rows
            assert torch.equal(p["latent"], lat[idx])
            assert torch.equal(p["next_latent"], nxt[idx])
            assert torch.equal(p["timestep"], ts[idx])
            assert torch.equal(p["log_prob_old"], rlp[idx])
            # debug tensors sliced to the same sde index
            d = p["rollout_debug_tensors"]
            assert torch.equal(d["rollout_step_model_output"], s.rollout_debug_tensors.rollout_model_outputs[idx])
            assert torch.equal(d["rollout_step_variance_noise"], s.rollout_debug_tensors.rollout_variance_noises[idx])
            assert torch.equal(
                d["rollout_step_prev_sample_mean"], s.rollout_debug_tensors.rollout_prev_sample_means[idx]
            )
            assert torch.equal(d["rollout_step_noise_std_dev"], s.rollout_debug_tensors.rollout_noise_std_devs[idx])


def test_l2_converter_requires_sigmas():
    """A trajectory without the rollout sigmas snapshot must raise (no derived fallback)."""
    T, sde = 4, [0, 2]
    samples = [_mk_sample(i, T, sde, with_sigmas=False, with_debug=False) for i in range(2)]
    try:
        expand_samples_to_train_pairs(None, samples, [1.0, 2.0], [1.0, 2.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing dit_trajectory.sigmas")


def test_l2_converter_rejects_mismatched_scheduler_timesteps():
    """One scheduler_meta is returned for the whole batch (from sample 0); a sample
    carrying a different schedule must raise, not silently inherit sample 0's."""
    samples = [_mk_sample(i, 6, [1, 3, 4]) for i in range(3)]
    samples[2].dit_trajectory.timesteps = samples[2].dit_trajectory.timesteps + 1.0  # tamper
    try:
        expand_samples_to_train_pairs(None, samples, [0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched scheduler_timesteps")


def test_l2_converter_rejects_mismatched_scheduler_sigmas():
    samples = [_mk_sample(i, 4, [0, 2]) for i in range(2)]  # with_sigmas=True
    samples[1].dit_trajectory.sigmas = samples[1].dit_trajectory.sigmas + 1.0  # tamper
    try:
        expand_samples_to_train_pairs(None, samples, [1.0, 2.0], [1.0, 2.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched scheduler_sigmas")


def test_scheduler_meta_from_rollout_requires_sigmas():
    """The train actor consumes the rollout sigmas snapshot verbatim; no timesteps/N fallback."""
    ts = torch.tensor([999.0, 500.0, 1.0])
    sig = torch.tensor([1.0, 0.5, 0.001, 0.0])
    out_ts, out_sig = scheduler_meta_from_rollout(
        {"scheduler_timesteps": ts, "scheduler_sigmas": sig}, device=torch.device("cpu")
    )
    assert torch.equal(out_ts, ts) and torch.equal(out_sig, sig)
    try:
        scheduler_meta_from_rollout({"scheduler_timesteps": ts}, device=torch.device("cpu"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing scheduler_sigmas")


# --------------------------------------------------------------------------------------
# L3 — Cond window padding: per-microbatch collate(pad_to_len) == legacy window-slice
# --------------------------------------------------------------------------------------
def _collate():
    # call the (self-free) collate as an unbound function to avoid config instantiation
    from miles.backends.fsdp_utils.configs.qwen_image import QwenImageTrainPipelineConfig

    return QwenImageTrainPipelineConfig.collate_cond_for_sample_batch


def _cond(seq_len: int, dim: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        "encoder_hidden_states": torch.randn(1, seq_len, dim, generator=g),
        "txt_seq_lens": [seq_len],
        "img_shapes": [(1, 2, 2)],
    }


def _legacy_window_then_tile(collate, window_conds, tile_sample_idx, tstep, window_max):
    """Mirror legacy _build_train_grids window-collate + _tile_collated_cond slice."""
    win = collate(None, window_conds, "cpu", pad_to_len=window_max)
    rows = torch.tensor(tile_sample_idx)

    def tile(v):
        if isinstance(v, torch.Tensor):
            return v.index_select(0, rows).repeat_interleave(tstep, dim=0)
        if isinstance(v, list):
            return [v[i] for i in tile_sample_idx for _ in range(tstep)]
        return v

    return {k: tile(v) for k, v in win.items()}


def _assert_cond_equal(a, b):
    assert set(a) == set(b)
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert torch.equal(a[k], b[k]), f"tensor mismatch: {k}"
        else:
            assert a[k] == b[k], f"value mismatch: {k}"


def test_l3_cond_pad_window_matches_legacy_tile_slice():
    collate = _collate()
    dim, tstep = 8, 2
    lens = [5, 9, 3, 7]  # window of 4 samples, varying text lengths
    window = [_cond(L, dim, seed=i) for i, L in enumerate(lens)]
    window_max = max(lens)
    tile_sample_idx = [0, 2]  # this micro-batch covers samples 0 and 2

    legacy = _legacy_window_then_tile(collate, window, tile_sample_idx, tstep, window_max)

    # refactor: per-pair conds in sample-major order [s0, s0, s2, s2], padded to window_max
    per_pair = [window[si] for si in tile_sample_idx for _ in range(tstep)]
    refac = collate(None, per_pair, "cpu", pad_to_len=window_max)

    _assert_cond_equal(legacy, refac)
    # the contract: shared window seq_len, not the micro-batch-local max
    assert refac["encoder_hidden_states"].shape[1] == window_max
    assert refac["encoder_hidden_states_mask"].shape[1] == window_max


def test_l3_without_window_pad_would_diverge():
    """Sanity: per-microbatch-local padding (no pad_to_len) yields a *different* seq_len
    than the legacy window padding — i.e. the --diffusion-train-cond-pad-window flag is
    what closes the gap, not a no-op."""
    collate = _collate()
    dim, tstep = 8, 2
    lens = [5, 9, 3, 7]
    window = [_cond(L, dim, seed=i) for i, L in enumerate(lens)]
    window_max = max(lens)  # 9 (driven by sample 1, NOT in this tile)
    tile_sample_idx = [0, 2]  # local max = max(5, 3) = 5

    per_pair = [window[si] for si in tile_sample_idx for _ in range(tstep)]
    local = collate(None, per_pair, "cpu")  # no pad_to_len
    windowed = collate(None, per_pair, "cpu", pad_to_len=window_max)

    assert local["encoder_hidden_states"].shape[1] == max(lens[0], lens[2])  # 5
    assert windowed["encoder_hidden_states"].shape[1] == window_max  # 9
    assert local["encoder_hidden_states"].shape[1] != windowed["encoder_hidden_states"].shape[1]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
