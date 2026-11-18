"""A rollout's generated output through the output store, from the engine's marker to its delete.

Mental model (one microgroup, store on; the rollout process never reads an output it does not log):

    engine response        {"__store__": "generated_output.<b>"}
        |  apply_rollout_image_response                 (1) a StoredTensor, layout left as stored
        v
    sample.generated_output = StoredTensor(ref, "generated_output.<b>")
        |  slot released; the reward is given the refs  (2) at every call site: microgroup, custom generate,
        v                                                   group reward
    reward pool actor = _StoredOutputRewardActor        (3) with the store on, not with it off
        |  read_generated_outputs                       (4) == the inline parse; the buffer is registered once,
        |                                                   and again only to grow
        |                                               (5) reads from several threads get their own tensors
        |  scorer.score_batch(tensors)                  (6) refs read, inline tensors passed through
        |  PickScore: pickscore_rm hands a stored video over unsliced   (7)
        |             the actor picks its frames of the whole video     (8) == the frames picked inline
        v
    release_generated_outputs after the reward          (9) only output fields deleted, outputs None, inline
        |                                                   untouched, nothing read when no media is logged
        |  keep the first N for media as tensors        (10) at most N across concurrent microgroups
        |                                               (11) a failing read or delete does not fail the rollout
        v
    generate_rollout_async sets the media counter       (12) only on rollouts that log media
    rollout manager logs the samples holding a tensor   (13)
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import asyncio
import concurrent.futures
import contextlib
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from radix_transfer_plugins import OutputStore
from radix_transfer_plugins.stashes.tensor_stash import TensorStash
from tests.fast.utils.test_output_store_materialize import MemoryBackend

import miles.ray.rollout as rollout_module
import miles.rollout.rm_hub.core as core_module
import miles.rollout.rm_hub.pickscore as pickscore_module
import miles.rollout.sglang_diffusion_rollout as rollout_fn_module
from miles.ray.rollout import RolloutManager
from miles.rollout.rm_hub.core import _StoredOutputRewardActor
from miles.rollout.sglang_diffusion_rollout import (
    generate_and_rm_group,
    generate_and_rm_microgroup,
    generate_rollout_async,
    release_generated_outputs,
)
from miles.utils.diffusion_rollout_response import (
    _normalize_generated_output,
    apply_rollout_image_response,
    read_generated_outputs,
)
from miles.utils.output_store import RegisteredTensorReader, StoredTensor
from miles.utils.processing_utils import sample_frame_indices
from miles.utils.types import Sample

_CHW = torch.arange(3 * 5 * 6, dtype=torch.float32).reshape(3, 5, 6) / 90
_HWC = _CHW.permute(1, 2, 0).contiguous()
_LATENTS = "dit_trajectory.latents.0.0"
_RolloutManager = RolloutManager.__ray_metadata__.modified_class


class _PartsBackend(MemoryBackend):
    """Deletes the named parts only and reads byte ranges, as Mooncake does; counts registrations and range reads."""

    def __init__(self):
        super().__init__()
        self.registrations = 0
        self.range_reads = 0

    def register(self, memory):
        self.registrations += 1

    def get_ranges(self, reads, *, into):
        self.range_reads += 1
        for read in reads:
            data = self.objects[read.ref.key][read.field]
            into[read.dst_offset : read.dst_offset + read.nbytes] = data[
                read.src_offset : read.src_offset + read.nbytes
            ]

    def delete_many(self, keys, *, parts):
        for key, names in zip(keys, parts, strict=True):
            for name in names:
                self.objects[key].pop(name, None)


def _stored(tensors: dict[str, torch.Tensor]):
    backend = _PartsBackend()
    store = OutputStore(backend=backend)
    (ref,) = store.put_many([TensorStash(tensors)])
    return store, backend, ref


def _stored_samples(ref, indices) -> list[Sample]:
    return [Sample(index=b, generated_output=StoredTensor(ref, f"generated_output.{b}")) for b in indices]


def _state(store, *, media_outputs_left=0):
    return SimpleNamespace(
        semaphore=asyncio.Semaphore(1),
        dp_rank_context=contextlib.nullcontext,
        output_store=store,
        output_reader=RegisteredTensorReader(store) if store is not None else None,
        media_outputs_left=media_outputs_left,
    )


def test_parse_keeps_a_stored_output_as_a_store_ref():
    """Canonicalizing a StoredTensor at parse time would fail; its layout is fixed where it is read."""
    store, _, ref = _stored({"generated_output.0": _HWC})
    body = {"output_store_ref": ref.to_builtins(), "generated_output": {"__store__": "generated_output.0"}}

    sample = apply_rollout_image_response(Sample(index=0), body)

    assert sample.generated_output == StoredTensor(sample.rollout_store_ref, "generated_output.0")


@pytest.mark.asyncio
@pytest.mark.parametrize("call_site", ["microgroup", "custom_generate", "group_rm"])
async def test_every_call_site_hands_the_reward_refs_and_releases_them(monkeypatch, call_site):
    """The rollout process must not read an output on any path, generation must not wait for a reward, and a
    path that skipped the release would leave every output in the store until the train step."""
    store, backend, ref = _stored({"generated_output.0": _CHW, "generated_output.1": _CHW, _LATENTS: _CHW})
    state = _state(store)
    rewarded = []

    async def generate(args, microgroup, sampling_params, evaluation=False):
        for sample in microgroup:
            sample.generated_output = StoredTensor(ref, f"generated_output.{sample.index}")
        return microgroup

    async def reward(args, samples):
        rewarded.extend((sample.generated_output, state.semaphore.locked()) for sample in samples)
        return [1.0] * len(samples)

    monkeypatch.setattr(rollout_fn_module, "GenerateState", lambda args: state)
    monkeypatch.setattr(rollout_fn_module, "generate_microgroup", generate)
    monkeypatch.setattr(rollout_fn_module, "load_function", lambda path: generate)
    monkeypatch.setattr(rollout_fn_module, "batched_async_rm", reward)
    monkeypatch.setattr(rollout_fn_module.hooks, "record_trajectory", lambda sample: None)
    args = Namespace(
        custom_generate_function_path="custom.generate" if call_site == "custom_generate" else None,
        group_rm=call_site == "group_rm",
        n_samples_per_prompt=2,
        rollout_seed=0,
        rollout_microgroup_size=2,
    )
    group = [Sample(index=b, group_index=0) for b in (0, 1)]

    if call_site == "group_rm":
        await generate_and_rm_group(args, group, {})
    else:
        await generate_and_rm_microgroup(args, group, {})

    assert len(rewarded) == 2
    assert all(isinstance(output, StoredTensor) and not slot_held for output, slot_held in rewarded)
    assert backend.range_reads == 0 and set(backend.objects[ref.key]) == {_LATENTS}
    assert [sample.generated_output for sample in group] == [None, None]


@pytest.mark.parametrize("use_output_store", [True, False], ids=["store_on", "store_off"])
def test_pool_actors_read_the_store_only_with_it_on(monkeypatch, use_output_store):
    """Unwrapped actors would get refs they cannot score; with the store off there is no store to connect to.
    A PickScore actor without its frame count would average every frame of a stored video."""
    created = []

    class FakeRemote:
        def __init__(self, actor_cls):
            self.actor_cls = actor_cls

        def options(self, **options):
            return self

        def remote(self, **kwargs):
            created.append((self.actor_cls, kwargs))

    monkeypatch.setattr(core_module.ray, "remote", FakeRemote)
    monkeypatch.setattr(pickscore_module.SingletonMeta, "_instances", {})
    args = Namespace(
        use_output_store=use_output_store,
        pickscore_processor_path="processor",
        pickscore_model_path="model",
        pickscore_batch_size=8,
        pickscore_num_workers=1,
        pickscore_num_gpus_per_worker=0,
        pickscore_reward_colocate=False,
        pickscore_num_frames=3,
    )

    pickscore_module.AsyncPickScorePool(args)

    ((actor_cls, kwargs),) = created
    scorer_kwargs = {"processor_path": "processor", "model_path": "model", "frames_per_forward": 8, "num_frames": 3}
    if use_output_store:
        assert actor_cls is _StoredOutputRewardActor
        assert kwargs == {
            "scorer_cls": pickscore_module.PickScoreRewardActor,
            "scorer_kwargs": scorer_kwargs,
            "args": args,
        }
    else:
        assert actor_cls is pickscore_module.PickScoreRewardActor and kwargs == scorer_kwargs


def test_read_matches_the_inline_parse_and_registers_its_buffer_once():
    """A reward must score the same tensor as on the inline path, whatever layout the engine stored. Under RDMA a
    registration pins the memory, so a reader that registered on every read would be slower than the transfer."""
    store, backend, ref = _stored({"generated_output.0": _CHW, "generated_output.1": _HWC, "generated_output.2": _CHW})
    reader = RegisteredTensorReader(store)
    one_output = [StoredTensor(ref, "generated_output.0")]
    three_outputs = [StoredTensor(ref, f"generated_output.{b}") for b in range(3)]

    first = read_generated_outputs(reader, one_output)
    again = read_generated_outputs(reader, one_output)
    grown = read_generated_outputs(reader, three_outputs)

    assert torch.equal(first[0], _normalize_generated_output(_CHW)) and torch.equal(again[0], first[0])
    assert torch.equal(grown[1], _normalize_generated_output(_HWC))
    assert torch.equal(grown[2], _normalize_generated_output(_CHW))
    # the first read registers, the second reuses, the larger one registers again
    assert backend.registrations == 2 and backend.range_reads == 3


def test_concurrent_reads_return_their_own_tensors():
    """API reward actors read from several threads; a read that shared the buffer would hand one thread another's
    bytes."""
    tensors = {f"generated_output.{b}": torch.full((3, 5, 6), float(b)) for b in range(8)}
    store, _, ref = _stored(tensors)
    reader = RegisteredTensorReader(store)

    def reads_its_own_tensor(read_index):
        field = f"generated_output.{read_index % 8}"
        (tensor,) = reader.read([StoredTensor(ref, field)])
        return torch.equal(tensor, tensors[field])

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        assert all(pool.map(reads_its_own_tensor, range(200)))


def test_reward_actor_reads_refs_and_passes_tensors_through(monkeypatch):
    """The scorer must get tensors in the order it was given outputs; eval outputs arrive inline."""
    store, _, ref = _stored({"generated_output.0": _CHW, "generated_output.1": _HWC})
    monkeypatch.setattr(core_module, "connect_output_store", lambda args: store)
    received = []

    class FakeScorer:
        def score_batch(self, outputs, prompts):
            received.extend(outputs)
            return [1.0] * len(outputs)

    actor = _StoredOutputRewardActor(scorer_cls=FakeScorer, scorer_kwargs={}, args=Namespace())
    inline = torch.zeros(3, 1, 4, 5)
    outputs = [StoredTensor(ref, "generated_output.0"), inline, StoredTensor(ref, "generated_output.1")]

    assert actor.score_batch(outputs, ["a", "b", "c"]) == [1.0, 1.0, 1.0]
    assert torch.equal(received[0], _normalize_generated_output(_CHW))
    assert received[1] is inline
    assert torch.equal(received[2], _normalize_generated_output(_HWC))


@pytest.mark.asyncio
async def test_pickscore_hands_a_stored_video_over_unsliced(monkeypatch):
    """Slicing a StoredTensor in the rollout process would crash; its frames are picked where it is read."""
    _, _, ref = _stored({"generated_output.0": torch.rand(3, 7, 4, 5)})
    scored = []

    class FakePool:
        async def score(self, outputs, prompts):
            scored.extend(outputs)
            return [1.0] * len(outputs), 0

    monkeypatch.setattr(pickscore_module, "AsyncPickScorePool", lambda args: FakePool())
    stored_video = StoredTensor(ref, "generated_output.0")
    samples = [
        Sample(prompt="a", generated_output=stored_video),
        Sample(prompt="b", generated_output=torch.rand(3, 7, 4, 5)),
    ]

    await pickscore_module.pickscore_rm(Namespace(pickscore_num_frames=3), samples)

    assert scored[0] is stored_video and scored[1].shape == (3, 3, 4, 5)


@pytest.mark.parametrize(
    ("total_frames", "num_frames"),
    [(7, 3), (7, 1), (7, None), (2, 3)],
    ids=["3_of_7", "1_of_7", "all_of_7", "3_of_2"],
)
def test_pickscore_picks_the_same_frames_from_a_whole_stored_video(total_frames, num_frames):
    """A stored video reaches the actor whole; it must score as the frames the rollout process picks inline.
    The unpicked frames are on the 0-255 scale, so quantising before picking would rescale the picked ones."""
    actor = pickscore_module.PickScoreRewardActor.__new__(pickscore_module.PickScoreRewardActor)
    actor.frames_per_forward = 8
    actor.num_frames = num_frames
    actor.scorer = lambda prompts, images: [float(sum(image.getdata(0))) for image in images]
    picked_frames = sample_frame_indices(total_frames, num_frames)
    video = torch.rand(3, total_frames, 4, 6)
    for frame in set(range(total_frames)) - set(picked_frames):
        video[:, frame] = 200.0

    whole = actor.score_batch([video], ["p"])
    picked = actor.score_batch([video[:, picked_frames]], ["p"])

    assert whole == picked


@pytest.mark.asyncio
async def test_release_deletes_only_the_scored_outputs():
    """Latents stay for the trainer; the outputs go once scored, the samples drop their dangling refs, and an
    inline (eval) output is untouched. A rollout that logs no media reads nothing in the rollout process."""
    store, backend, ref = _stored({"generated_output.0": _CHW, "generated_output.1": _CHW, _LATENTS: _CHW})
    inline = torch.zeros(3, 1, 2, 2)
    samples = [*_stored_samples(ref, (0, 1)), Sample(index=2, generated_output=inline)]

    await release_generated_outputs(_state(store), samples)

    assert set(backend.objects[ref.key]) == {_LATENTS} and backend.range_reads == 0
    assert [sample.generated_output for sample in samples[:2]] == [None, None]
    assert samples[2].generated_output is inline


@pytest.mark.asyncio
async def test_logged_rollout_keeps_at_most_the_logged_count_as_tensors():
    """Media logging needs N outputs after the fields are gone; concurrent microgroups must not keep more than N."""
    store, backend, ref = _stored({f"generated_output.{b}": _CHW * b for b in range(4)})
    groups = [_stored_samples(ref, pair) for pair in ([0, 1], [2, 3])]
    state = _state(store, media_outputs_left=1)

    await asyncio.gather(*(release_generated_outputs(state, group) for group in groups))

    outputs = [sample.generated_output for group in groups for sample in group]
    assert torch.equal(outputs[0], _normalize_generated_output(_CHW * 0))
    assert outputs[1:] == [None, None, None]
    assert backend.objects[ref.key] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["read", "delete"])
async def test_failing_release_does_not_fail_the_rollout(failing):
    """A failed media read or delete must not lose a scored rollout; the rollout's release deletes the object."""
    store, backend, ref = _stored({"generated_output.0": _CHW, "generated_output.1": _CHW})

    def fail(*args, **kwargs):
        raise RuntimeError(f"Mooncake {failing} failed")

    setattr(store, "read_ranges" if failing == "read" else "delete_many", fail)
    samples = _stored_samples(ref, (0, 1))

    await release_generated_outputs(_state(store, media_outputs_left=1), samples)

    if failing == "read":
        assert samples[0].generated_output is None and backend.objects[ref.key] == {}
    else:
        assert torch.equal(samples[0].generated_output, _normalize_generated_output(_CHW))
    assert samples[1].generated_output is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("rollout_id", "media_outputs_kept"), [(10, 4), (11, 0)], ids=["logged", "not_logged"])
async def test_only_a_rollout_that_logs_media_keeps_outputs(monkeypatch, rollout_id, media_outputs_kept):
    """A rollout must keep exactly the outputs the rollout manager will log for it, and none otherwise."""

    class _Stop(Exception):
        pass

    class FakeState:
        media_outputs_left = None

        @property
        def remaining_batch_size(self):
            raise _Stop

    state = FakeState()
    monkeypatch.setattr(rollout_fn_module, "GenerateState", lambda args: state)
    args = Namespace(
        rollout_global_dataset=True,
        wandb_log_num_images=4,
        wandb_log_image_interval=10,
        dynamic_sampling_filter_path=None,
        rollout_batch_size=1,
        over_sampling_batch_size=1,
        n_samples_per_prompt=1,
    )

    with pytest.raises(_Stop):
        await generate_rollout_async(args, rollout_id, data_source=None)

    assert state.media_outputs_left == media_outputs_kept


def test_media_logging_takes_the_samples_that_kept_an_output(monkeypatch):
    """With the first samples' outputs dropped, logging the first N samples would log nothing; a store ref a
    rollout function never released would crash the logging thread."""
    _, _, ref = _stored({"generated_output.0": _CHW})
    media_sample = Sample(index=2, reward=1.0, generated_output=torch.zeros(3, 1, 2, 2))
    samples = [
        Sample(index=0, reward=1.0, generated_output=None),
        Sample(index=1, reward=1.0, generated_output=StoredTensor(ref, "generated_output.0")),
        media_sample,
    ]
    logged = []
    manager = SimpleNamespace(
        args=Namespace(
            wandb_log_num_images=1,
            wandb_log_image_interval=1,
            advantage_estimator="grpo",
            n_samples_per_prompt=1,
            reward_key=None,
        ),
        rollout_id=0,
        custom_convert_samples_to_train_data_func=None,
        custom_expand_samples_to_train_pairs_func=None,
        _post_process_rewards=lambda samples: ([1.0] * len(samples), [0.0] * len(samples)),
        _log_images_async=lambda media, **kwargs: logged.append(media["rollout_media/sample_images"]),
    )
    monkeypatch.setattr(rollout_module.tracking_utils, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(rollout_module, "flow_grpo_expand_samples_to_train_pairs", lambda *args: None)

    _RolloutManager._convert_samples_to_train_data(manager, samples)

    assert logged == [[media_sample]]
