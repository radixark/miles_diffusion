"""A rollout's generated output through the output store, from the engine's marker to the reward.

Mental model (one microgroup, store on):

    engine response        {"__store__": "generated_output.<b>"}
        |  apply_rollout_image_response               (1) a StoredTensor, layout left as stored
        v
    sample.generated_output = StoredTensor(ref, "generated_output.<b>")
        |  load_generated_outputs, inside the semaphore
        |      read_generated_outputs                 (2) == the inline-parsed tensor (CHW, HWC); one read per ref
        |      delete the output fields               (3) latents stay for the trainer; inline (eval) outputs untouched;
        |                                                 read + delete is the dashboard's store_read stage
        |                                             (4) a failing delete is left to the release; a failing read fails
        v
    sample.generated_output = tensor                  (5) every call site (microgroup, group_rm, custom generate)
        |                                                 rewards tensors, read while holding the semaphore
        v
    reward, media logging, trainer: as with the store off
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import asyncio
import contextlib
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from radix_transfer_plugins import OutputStore
from radix_transfer_plugins.stashes.tensor_stash import TensorStash
from tests.fast.utils.test_output_store_materialize import MemoryBackend

import miles.rollout.sglang_diffusion_rollout as rollout_fn_module
from miles.dashboard.events import STAGE_KINDS
from miles.rollout.sglang_diffusion_rollout import (
    generate_and_rm_group,
    generate_and_rm_microgroup,
    load_generated_outputs,
)
from miles.utils.diffusion_rollout_response import (
    _normalize_generated_output,
    apply_rollout_image_response,
    read_generated_outputs,
)
from miles.utils.output_store import StoredTensor
from miles.utils.types import Sample

_CHW = torch.arange(3 * 5 * 6, dtype=torch.float32).reshape(3, 5, 6) / 90
_HWC = _CHW.permute(1, 2, 0).contiguous()
_LATENTS = "dit_trajectory.latents.0.0"


class _PartsBackend(MemoryBackend):
    """Deletes the named parts only and ignores missing ones, as Mooncake does."""

    def delete_many(self, keys, *, parts):
        for key, names in zip(keys, parts, strict=True):
            for name in names:
                self.objects[key].pop(name, None)


def _stored(tensors: dict[str, torch.Tensor]):
    backend = _PartsBackend()
    store = OutputStore(backend=backend)
    (ref,) = store.put_many([TensorStash(tensors)])
    return store, backend, ref


def test_parse_keeps_a_stored_output_as_a_store_ref():
    """Canonicalizing a StoredTensor at parse time would fail; its layout is fixed where it is read."""
    store, _, ref = _stored({"generated_output.0": _HWC})
    body = {"output_store_ref": ref.to_builtins(), "generated_output": {"__store__": "generated_output.0"}}

    sample = apply_rollout_image_response(Sample(index=0), body)

    assert sample.generated_output == StoredTensor(sample.rollout_store_ref, "generated_output.0")


def test_read_matches_the_inline_parse_with_one_read_per_ref():
    """A reward must score the same tensor as on the inline path, whatever layout the engine stored."""
    store, backend, ref = _stored({"generated_output.0": _CHW, "generated_output.1": _HWC})

    outputs = read_generated_outputs(
        store, [StoredTensor(ref, "generated_output.0"), StoredTensor(ref, "generated_output.1")]
    )

    assert torch.equal(outputs[0], _normalize_generated_output(_CHW))
    assert torch.equal(outputs[1], _normalize_generated_output(_HWC))
    assert backend.reads == [[(ref.key, ["generated_output.0", "generated_output.1"])]]


@pytest.mark.asyncio
async def test_load_reads_the_outputs_and_deletes_only_their_fields():
    """Latents stay for the trainer; the outputs leave the store once read; an inline (eval) output is untouched.
    The dashboard drops a stage name it does not know, so the read time must use a known one."""
    store, backend, ref = _stored({"generated_output.0": _CHW, "generated_output.1": _HWC, _LATENTS: _CHW})
    inline = torch.zeros(3, 1, 2, 2)
    samples = [Sample(index=b, generated_output=StoredTensor(ref, f"generated_output.{b}")) for b in (0, 1)]
    samples.append(Sample(index=2, generated_output=inline))

    await load_generated_outputs(SimpleNamespace(output_store=store), samples)

    assert torch.equal(samples[0].generated_output, _normalize_generated_output(_CHW))
    assert torch.equal(samples[1].generated_output, _normalize_generated_output(_HWC))
    assert samples[2].generated_output is inline
    assert set(backend.objects[ref.key]) == {_LATENTS}
    stages = [
        [segment["stage"] for segment in (sample.metadata or {}).get("lifecycle_stages", [])] for sample in samples
    ]
    assert stages == [["store_read"], ["store_read"], []] and "store_read" in STAGE_KINDS


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["read", "delete"])
async def test_a_failing_delete_is_left_to_the_release_but_a_failing_read_fails(failing):
    """The reward cannot run without the output; a field left behind is deleted with the object by the release."""
    store, _, ref = _stored({"generated_output.0": _CHW})

    def fail(*args, **kwargs):
        raise RuntimeError(f"Mooncake {failing} failed")

    setattr(store, "read_many" if failing == "read" else "delete_many", fail)
    sample = Sample(index=0, generated_output=StoredTensor(ref, "generated_output.0"))

    if failing == "read":
        with pytest.raises(RuntimeError, match="read failed"):
            await load_generated_outputs(SimpleNamespace(output_store=store), [sample])
    else:
        await load_generated_outputs(SimpleNamespace(output_store=store), [sample])
        assert torch.equal(sample.generated_output, _normalize_generated_output(_CHW))


@pytest.mark.asyncio
@pytest.mark.parametrize("call_site", ["microgroup", "group_rm", "custom_generate"])
async def test_every_call_site_rewards_tensors_read_inside_the_semaphore(monkeypatch, call_site):
    """A call site without the load would hand rewards store refs; a load outside the semaphore would let the
    store fill with outputs of more microgroups than there are generation slots."""
    store, backend, ref = _stored({"generated_output.0": _CHW, "generated_output.1": _CHW, _LATENTS: _CHW})
    state = SimpleNamespace(semaphore=asyncio.Semaphore(1), dp_rank_context=contextlib.nullcontext, output_store=store)
    read_inside_semaphore = []
    read_many = store.read_many

    def recording_read_many(*args, **kwargs):
        read_inside_semaphore.append(state.semaphore.locked())
        return read_many(*args, **kwargs)

    store.read_many = recording_read_many
    rewarded = []

    async def generate(args, microgroup, sampling_params, evaluation=False):
        for sample in microgroup:
            sample.generated_output = StoredTensor(ref, f"generated_output.{sample.index}")
        return microgroup

    async def reward(args, samples):
        rewarded.extend(sample.generated_output for sample in samples)
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
    assert all(torch.equal(output, _normalize_generated_output(_CHW)) for output in rewarded)
    assert read_inside_semaphore == [True]
    assert set(backend.objects[ref.key]) == {_LATENTS}
