"""StoredTensorReader hands out what OutputStore.read_many would, read through one kept registered buffer."""

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from radix_transfer_plugins import OutputStore
from radix_transfer_plugins.stashes.tensor_stash import TensorStash
from tests.fast.utils.test_output_store_materialize import MemoryBackend

from miles.utils.output_store import StoredTensor, StoredTensorReader

_TENSORS = {
    "latent": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).bfloat16(),
    "pooled": torch.linspace(-1.0, 1.0, 8),
    "none": torch.empty(0, 5, dtype=torch.int64),
    "scalar": torch.tensor(True),
    "image": torch.arange(3 * 5 * 6, dtype=torch.uint8).reshape(3, 5, 6),
}


def _stored(tensors):
    backend = MemoryBackend()
    store = OutputStore(backend=backend)
    (ref,) = store.put_many([TensorStash(tensors)])
    return store, backend, ref


def test_reads_equal_read_many_and_skip_empty_fields():
    """The trainer must get the same tensors as through read_many; an empty field has no bytes to read."""
    store, backend, ref = _stored(_TENSORS)
    names = sorted(_TENSORS)
    (stash,) = store.read_many([(ref, names)], stash_type=TensorStash)

    tensors = StoredTensorReader(store).read([StoredTensor(ref, name) for name in names])

    for name, tensor in zip(names, tensors, strict=True):
        want = stash.tensors[name]
        assert tensor.dtype == want.dtype and tensor.shape == want.shape and torch.equal(tensor, want), name
    assert backend.range_reads == [[(ref.key, name) for name in names if name != "none"]]


def test_only_empty_fields_read_nothing():
    store, backend, ref = _stored({"none": _TENSORS["none"]})
    (tensor,) = StoredTensorReader(store).read([StoredTensor(ref, "none")])
    assert tensor.shape == (0, 5) and tensor.dtype == torch.int64
    assert backend.range_reads == [] and backend.registrations == 0


def test_a_field_is_read_once_and_the_tensors_are_the_callers():
    """A shard asks for a cond field once per pair; the next read overwrites the buffer, not the tensors."""
    store, backend, ref = _stored(_TENSORS)
    reader = StoredTensorReader(store)

    first, again = reader.read([StoredTensor(ref, "pooled"), StoredTensor(ref, "pooled")])
    first.zero_()
    (fresh,) = reader.read([StoredTensor(ref, "pooled")])

    assert backend.range_reads == [[(ref.key, "pooled")], [(ref.key, "pooled")]]
    assert again is first and torch.equal(fresh, _TENSORS["pooled"])


def test_a_stored_dtype_torch_cannot_load_fails_as_in_read_many():
    store, _, ref = _stored({"fp8": torch.zeros(2, dtype=torch.float8_e4m3fn)})
    with pytest.raises(ValueError, match="unknown tensor dtype"):
        store.read_many([(ref, ["fp8"])], stash_type=TensorStash)
    with pytest.raises(ValueError, match="unknown tensor dtype"):
        StoredTensorReader(store).read([StoredTensor(ref, "fp8")])


def test_a_missing_field_fails_before_reading():
    store, backend, ref = _stored(_TENSORS)
    with pytest.raises(ValueError, match=r"no field \['other'\]"):
        StoredTensorReader(store).read([StoredTensor(ref, "latent"), StoredTensor(ref, "other")])
    assert backend.range_reads == []
