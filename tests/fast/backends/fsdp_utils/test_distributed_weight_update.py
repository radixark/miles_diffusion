from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils import diffusion_update_weight_utils as update
from miles.utils.arguments import rollout_merges_lora, validate_lora_weight_sync_args


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
        assert kwargs["world_size"] == 5
        return "group"

    monkeypatch.setattr(update, "init_custom_process_group", join)
    monkeypatch.setattr(update.ray, "get", lambda refs: calls.append(("wait", refs)))
    group = update.connect_rollout_engines_from_distributed(engines, 2, "update", timedelta(seconds=30))
    assert group == "group"
    assert [(calls[i][1]["rank_offset"], calls[i][1]["world_size"]) for i in range(2)] == [(1, 5), (3, 5)]
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


def test_first_connect_clears_groups_left_on_engines(monkeypatch):
    # No trainer group yet, but an earlier partial init may have left the group on some engines.
    args = SimpleNamespace(rollout_num_gpus_per_engine=1, distributed_timeout_minutes=1)
    updater = update.DiffusionUpdateWeightFromDistributed(args, {})
    events = []
    engines = [
        SimpleNamespace(
            destroy_weights_update_group=SimpleNamespace(remote=lambda name, i=i: events.append(("leave", i, name)))
        )
        for i in range(2)
    ]
    monkeypatch.setattr(update.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(update.dist, "destroy_process_group", lambda group: events.append(("destroy", group)))
    monkeypatch.setattr(update.ray, "get", lambda refs: events.append("ack"))
    monkeypatch.setattr(
        update, "connect_rollout_engines_from_distributed", lambda **kwargs: events.append("connect") or "new"
    )
    updater.connect_rollout_engines(engines, None)
    group_name = updater._group_name
    assert events == [("leave", 0, group_name), ("leave", 1, group_name), "ack", "connect"]
    assert updater._model_update_group == "new"


def test_lora_fields_follow_the_adapter_and_only_ride_lora_buckets(monkeypatch):
    # The rollout merges W + (alpha / r) * B @ A with the trained adapter's own alpha and r, not the CLI flags.
    model = SimpleNamespace(peft_config={"default": SimpleNamespace(lora_alpha=64, r=32)}, active_adapter="default")
    args = SimpleNamespace(lora_alpha=1, lora_rank=1)
    updater = update.DiffusionUpdateWeightFromDistributed(args, {"transformer": model})
    updater.rollout_engines, updater._model_update_group = ["engine"], "group"
    calls = []
    monkeypatch.setattr(update.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(update, "broadcast_bucket", lambda **kwargs: calls.append(kwargs))
    tensors = [("a", torch.ones(1))]
    updater.update_bucket_weights(tensors, "transformer", weight_update_mode="lora_merge")
    updater.update_bucket_weights(tensors, "transformer")
    assert (calls[0]["weight_update_mode"], calls[0]["lora_alpha"], calls[0]["lora_rank"]) == ("lora_merge", 64, 32)
    assert not {"weight_update_mode", "lora_alpha", "lora_rank"} & calls[1].keys()


@pytest.mark.parametrize(
    "use_lora,lora_ipc_weight_sync,colocate,accepted,rollout_merges",
    [
        (False, False, True, True, False),
        (False, False, False, True, False),
        (True, False, True, True, False),  # the trainer merges and pushes full weights over IPC
        (True, False, False, True, True),  # adapters over NCCL
        (True, True, True, True, True),  # adapters over IPC
        (True, True, False, False, True),  # IPC needs --colocate
    ],
)
def test_lora_ipc_weight_sync_needs_colocate(use_lora, lora_ipc_weight_sync, colocate, accepted, rollout_merges):
    args = SimpleNamespace(
        use_lora=use_lora,
        lora_ipc_weight_sync=lora_ipc_weight_sync,
        colocate=colocate,
        lora_target_modules=["to_q"],
        train_only=False,
        debug_rollout_only=False,
    )
    assert rollout_merges_lora(args) == rollout_merges
    if accepted:
        validate_lora_weight_sync_args(args)
    else:
        with pytest.raises(ValueError, match="requires --colocate"):
            validate_lora_weight_sync_args(args)


def test_rollout_merged_lora_needs_target_modules():
    args = SimpleNamespace(
        use_lora=True,
        lora_ipc_weight_sync=False,
        colocate=False,
        lora_target_modules=None,
        train_only=False,
        debug_rollout_only=False,
    )
    with pytest.raises(ValueError, match="requires LoRA target modules"):
        validate_lora_weight_sync_args(args)


@pytest.mark.parametrize("mode", ["train_only", "debug_rollout_only"])
def test_lora_runs_without_weight_sync_skip_the_colocate_check(mode):
    args = SimpleNamespace(
        use_lora=True,
        lora_ipc_weight_sync=True,
        colocate=False,
        lora_target_modules=["to_q"],
        train_only=False,
        debug_rollout_only=False,
    )
    setattr(args, mode, True)
    validate_lora_weight_sync_args(args)
