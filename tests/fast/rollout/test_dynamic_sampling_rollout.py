from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

# Dynamic sampling must not leak generation work past the end of a rollout:
#
#   fetched groups  ==  kept groups  +  dropped groups      (no extra generation, no skipped prompts)
#   in-flight groups after generate_rollout returns  ==  0  (engines are idle before offload / weight sync)

import asyncio
import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.rollout import sglang_diffusion_rollout
from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.rollout.sglang_diffusion_rollout import GenerateState, generate_rollout

ROLLOUT_BATCH_SIZE = 8
N_SAMPLES_PER_PROMPT = 2

_inflight: set[str] = set()
_started: list[str] = []
_failing: set[str] = set()
# when set, every group takes the same time and these prompt ids are dropped instead of every 4th
_dropped_ids: set[int] | None = None


def _prompt_id(sample) -> int:
    return int(sample.prompt[1:])


async def fake_generate(args, microgroup, sampling_params, evaluation=False):
    prompt = microgroup[0].prompt
    _inflight.add(prompt)
    _started.append(prompt)
    for sample in microgroup:
        sample.request_id = f"req-{prompt}"
    try:
        if _dropped_ids is not None:
            await asyncio.sleep(0.05)
        else:
            # uneven latency so groups finish out of submission order
            await asyncio.sleep(0.01 * ((_prompt_id(microgroup[0]) * 7) % 5 + 1))
        if prompt in _failing:
            raise RuntimeError(f"engine failed on {prompt}")
    finally:
        _inflight.discard(prompt)
    return microgroup


async def fake_group_rm(args, samples):
    # every 4th prompt gets all-equal rewards, so check_reward_nonzero_std drops it
    dropped = _prompt_id(samples[0]) in _dropped_ids if _dropped_ids is not None else _prompt_id(samples[0]) % 4 == 3
    if dropped:
        return [0.0] * len(samples)
    return [float(i) for i in range(len(samples))]


def _args(tmp_path):
    prompt_data = tmp_path / "prompts.jsonl"
    prompt_data.write_text("".join(json.dumps({"text": f"p{i}"}) + "\n" for i in range(200)))
    return Namespace(
        # data source
        rollout_global_dataset=True,
        prompt_data=str(prompt_data),
        input_key="text",
        metadata_key="metadata",
        rollout_seed=42,
        rollout_shuffle=False,
        buffer_filter_path=None,
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
        # rollout
        rollout_batch_size=ROLLOUT_BATCH_SIZE,
        over_sampling_batch_size=ROLLOUT_BATCH_SIZE,
        rollout_microgroup_size=N_SAMPLES_PER_PROMPT,
        dynamic_sampling_filter_path="miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
        rollout_sample_filter_path=None,
        custom_generate_function_path=f"{__name__}.fake_generate",
        custom_rm_path=f"{__name__}.fake_group_rm",
        group_rm=True,
        reward_key=None,
        # GenerateState
        sglang_server_concurrency=64,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        rollout_parser_num_workers=0,
        diffusion_step_strategy_path=None,
        # sampling params
        diffusion_negative_prompt=None,
        diffusion_generator_device=None,
        diffusion_eval_num_steps=None,
        diffusion_num_steps=10,
        diffusion_width=64,
        diffusion_height=64,
        diffusion_guidance_scale=4.0,
        diffusion_guidance_scale_2=None,
        diffusion_true_cfg_scale=None,
        diffusion_output_num_frames=None,
        diffusion_fps=None,
        diffusion_sde_type="sde",
        diffusion_noise_level=0.7,
        diffusion_log_prob_no_const=False,
        diffusion_debug_mode=False,
        rollout_video_dtype="keep",
        train_pipeline_config_path=None,
    )


@pytest.fixture
def rollout_env(tmp_path):
    global _dropped_ids
    GenerateState.clear_instances()
    _inflight.clear()
    _started.clear()
    _failing.clear()
    _dropped_ids = None
    fake_ctx = SimpleNamespace(get_node_id=lambda: "node-0")
    with patch.object(sglang_diffusion_rollout.ray, "get_runtime_context", return_value=fake_ctx):
        yield _args(tmp_path)
    _dropped_ids = None
    GenerateState.clear_instances()


def test_dynamic_filter_leaves_no_work_behind(rollout_env):
    args = rollout_env
    data_source = RolloutDataSourceWithBuffer(args)

    trained, dropped = set(), set()
    for rollout_id in range(3):
        n_started = len(_started)
        output = generate_rollout(args, rollout_id, data_source)
        assert not _inflight, f"rollout {rollout_id}: {sorted(_inflight)} still generating after return"

        groups = output.samples
        assert len(groups) == ROLLOUT_BATCH_SIZE
        assert all(len(group) == N_SAMPLES_PER_PROMPT for group in groups)
        kept = {group[0].prompt for group in groups}
        assert all(_prompt_id(group[0]) % 4 != 3 for group in groups)
        trained |= kept
        dropped |= {p for p in _started[n_started:] if int(p[1:]) % 4 == 3}

        # each started group is either trained on or filtered out, never generated for nothing
        assert len(_started) - n_started == len(kept) + output.metrics.get(
            "rollout/dynamic_filter/drop_zero_std_0.0", 0
        )

    # every prompt the data source handed out (and will checkpoint past) was trained on or filtered out
    fetched = {f"p{i}" for i in range(data_source.sample_offset)} | {g[0].prompt for g in data_source.buffer}
    assert fetched == trained | dropped
    assert not data_source.buffer


def test_failed_group_cancels_the_rest(rollout_env):
    args = rollout_env
    data_source = RolloutDataSourceWithBuffer(args)

    _failing.add("p0")  # p0 finishes first, while the slower groups are still generating
    with pytest.raises(RuntimeError, match="engine failed on p0"):
        generate_rollout(args, 0, data_source)
    assert not _inflight, f"{sorted(_inflight)} still generating after the rollout raised"

    # the next rollout starts from a clean state instead of awaiting the previous rollout's tasks
    _failing.clear()
    output = generate_rollout(args, 1, data_source)
    assert len(output.samples) == ROLLOUT_BATCH_SIZE
    assert not _inflight


def test_drop_refills_every_idle_engine(rollout_env):
    # 4 engines, 4 groups wanted, equal latency (one wave = one group time):
    #   wave 0: p0-p3,  p2 p3 dropped              -> 4 engines idle, submit 4
    #   wave 1: p4-p7,  p5 p6 p7 dropped           -> 4 engines idle, submit 4
    #   wave 2: p8-p11 all kept and finish together; whichever is seen first meets the target, the other 3 are extras
    # refilling only the dropped count instead would run p6, p7, p8 one after another on one engine.
    global _dropped_ids
    _dropped_ids = {2, 3, 5, 6, 7}
    args = rollout_env
    args.rollout_batch_size = args.over_sampling_batch_size = 4
    args.rollout_num_gpus = 4
    data_source = RolloutDataSourceWithBuffer(args)

    output = generate_rollout(args, 0, data_source)
    assert not _inflight

    kept = sorted(_prompt_id(group[0]) for group in output.samples)
    extras = sorted(_prompt_id(group[0]) for group in data_source.buffer)
    assert kept[:3] == [0, 1, 4]
    assert sorted(kept[3:] + extras) == [8, 9, 10, 11]
    # 12 groups in 3 waves of 4, never fewer groups in flight than engines
    assert len(_started) == 12
    # extras go back untouched, so the next rollout generates them from scratch
    assert all(sample.request_id is None and sample.reward is None for group in data_source.buffer for sample in group)

    _dropped_ids = set()
    output = generate_rollout(args, 1, data_source)
    assert sorted(_prompt_id(group[0]) for group in output.samples) == extras + [12]
    assert not data_source.buffer
