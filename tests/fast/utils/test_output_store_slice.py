"""StoreSlice indexes a stored tensor the way the conversion hubs index latents; refs are tracked per rollout."""

import pytest
import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from radix_transfer_plugins import FieldSpec, StoreRef

from miles.utils.output_store import StoreSlice, record_dropped_samples, store_refs_of, take_dropped_refs
from miles.utils.types import Sample


def _ref(key="k", shape=(2, 5, 3)) -> StoreRef:
    spec = FieldSpec(name="dit_trajectory.latents", dtype="bfloat16", shape=shape, codec="identity", nbytes=0)
    return StoreRef(backend="mooncake", key=key, fields=(spec,))


def test_int_index_fixes_a_row():
    latents = StoreSlice(_ref(), "dit_trajectory.latents", rows=(1,))
    assert latents.shape == (5, 3)
    assert latents[2].rows == (1, 2)
    assert latents[-1].rows == (1, 4)
    assert latents[2].shape == (3,)
    with pytest.raises(IndexError):
        latents[5]


def test_sequence_index_then_int_resolves_through_the_selection():
    latents = StoreSlice(_ref(), "dit_trajectory.latents", rows=(0,))
    window = latents[torch.tensor([3, 1])]
    assert window.shape == (2, 3)
    assert window[0].rows == (0, 3)
    assert window[1].rows == (0, 1)
    assert window[[1, 0]].select == (1, 3)


def test_refs_are_collected_from_nested_groups_and_drops():
    with_ref = [Sample(index=0, rollout_store_ref=_ref("a")), Sample(index=1, rollout_store_ref=_ref("a"))]
    without = Sample(index=2)
    assert list(store_refs_of([with_ref, without])) == ["a"]
    record_dropped_samples([[Sample(index=3, rollout_store_ref=_ref("b"))]])
    assert list(take_dropped_refs()) == ["b"]
    assert take_dropped_refs() == {}
