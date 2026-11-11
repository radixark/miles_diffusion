"""Output store refs are tracked per rollout and per dropped sample."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from radix_transfer_plugins import FieldSpec, StoreRef

from miles.utils.output_store import record_dropped_samples, store_refs_of, take_dropped_refs
from miles.utils.types import Sample


def _ref(key="k", shape=(2, 5, 3)) -> StoreRef:
    spec = FieldSpec(name="dit_trajectory.latents", dtype="bfloat16", shape=shape, codec="identity", nbytes=0)
    return StoreRef(backend="mooncake", key=key, fields=(spec,))


def test_refs_are_collected_from_nested_groups_and_drops():
    with_ref = [Sample(index=0, rollout_store_ref=_ref("a")), Sample(index=1, rollout_store_ref=_ref("a"))]
    without = Sample(index=2)
    assert list(store_refs_of([with_ref, without])) == ["a"]
    record_dropped_samples([[Sample(index=3, rollout_store_ref=_ref("b"))]])
    assert list(take_dropped_refs()) == ["b"]
    assert take_dropped_refs() == {}
