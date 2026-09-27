from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

from datetime import timedelta
from types import SimpleNamespace

import torch

from miles.backends.fsdp_utils import diffusion_update_weight_utils as update


def test_connect_assigns_every_engine_rank_before_waiting(monkeypatch):
    calls = []
    engines = []
    for index in range(2):

        def init(index=index, **kwargs):
            calls.append((index, kwargs))
            return index

        engines.append(SimpleNamespace(init_weights_update_group=SimpleNamespace(remote=init)))

    monkeypatch.setattr(update.ray._private.services, "get_node_ip_address", lambda: "127.0.0.1")

    def join(**kwargs):
        assert len(calls) == 2
        assert kwargs["rank"] == 0
        assert kwargs["world_size"] == 7
        return "group"

    monkeypatch.setattr(update, "init_custom_process_group", join)
    monkeypatch.setattr(update.ray, "get", lambda refs: calls.append(("wait", refs)))
    group = update.connect_rollout_engines_from_distributed(engines, [2, 4], "update", timedelta(seconds=30))
    assert group == "group"
    assert [calls[i][1]["rank_offset"] for i in range(2)] == [1, 3]
    assert calls[-1] == ("wait", [0, 1])


def test_broadcast_matches_metadata_and_retains_contiguous_buffers(monkeypatch):
    events = []
    payloads = []
    tensors = [("b", torch.arange(6).reshape(2, 3).t()), ("a", torch.ones(2, dtype=torch.bfloat16))]

    def receive(**kwargs):
        payloads.append(kwargs)
        events.append("rpc")
        return len(payloads)

    engines = [SimpleNamespace(update_weights_from_distributed=SimpleNamespace(remote=receive)) for _ in range(2)]

    def broadcast(tensor, src, group, async_op):
        assert len(payloads) == 2
        assert tensor.is_contiguous()
        index = events.count("broadcast")
        torch.testing.assert_close(tensor, tensors[index][1])
        events.append("broadcast")
        return SimpleNamespace(wait=lambda: events.append("wait"))

    monkeypatch.setattr(update.dist, "broadcast", broadcast)
    monkeypatch.setattr(update.ray, "get", lambda refs: events.append("ack"))
    update.broadcast_bucket(engines, "group", "update", tensors, "transformer")
    assert payloads[0]["names"] == ["b", "a"]
    assert payloads[0]["dtypes"] == ["int64", "bfloat16"]
    assert payloads[0]["shapes"] == [[3, 2], [2]]
    assert events == ["rpc", "rpc", "broadcast", "broadcast", "wait", "wait", "ack"]


def test_reconnect_destroys_old_group_before_joining(monkeypatch):
    args = SimpleNamespace(rollout_num_gpus_per_engine=2, distributed_timeout_minutes=1)
    updater = update.DiffusionUpdateWeightFromDistributed(args, {})
    updater._model_update_group = "old"
    events = []
    engine = SimpleNamespace(
        destroy_weights_update_group=SimpleNamespace(remote=lambda name: events.append("remote destroy"))
    )
    monkeypatch.setattr(update.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(update.dist, "destroy_process_group", lambda group: events.append(("destroy", group)))
    monkeypatch.setattr(update.ray, "get", lambda refs: events.append("ack"))

    def connect(**kwargs):
        events.append("connect")
        return "new"

    monkeypatch.setattr(update, "connect_rollout_engines_from_distributed", connect)
    updater.connect_rollout_engines([engine], None)
    assert events == ["remote destroy", ("destroy", "old"), "ack", "connect"]
    assert updater._model_update_group == "new"
