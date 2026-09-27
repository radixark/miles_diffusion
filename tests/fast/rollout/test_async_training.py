"""Exercise the async driver with real serial Ray actors and tiny CPU weights."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="stage-a-cpu", labels=[])

import json
import tempfile
import time
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import ray
import torch
from train_diffusion_async import train_loop

from miles.backends.fsdp_utils.ema import EmaShadow
from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.utils.ray_utils import Box


@ray.remote
class RolloutProbe:
    def __init__(self, args, delay):
        self.source = RolloutDataSourceWithBuffer(args)
        self.source.load(args.start_rollout_id - 1)
        self.delay = delay
        self.weight = 0.0
        self.events = []

    def generate(self, rollout_id):
        start = time.monotonic()
        sample = self.source.get_samples(1)[0][0]
        time.sleep(self.delay)
        self.events.append((rollout_id, start, time.monotonic()))
        return [Box(ray.put(dict(rollout_id=rollout_id, weight=self.weight, sample_index=sample.index)))]

    def save(self, rollout_id):
        self.source.save(rollout_id)

    def install(self, weight):
        self.weight = weight

    def get_rollout_engines_and_lock(self):
        return [], None, 0

    def get_events(self):
        return self.events


@ray.remote
class TrainerProbe:
    def __init__(self, manager, delay, restored=None):
        from miles.backends.fsdp_utils import actor
        from miles.utils.timer import Timer

        self.actor = actor
        self.delay = delay
        self.rollout_manager = manager
        self.args = Namespace(
            offload_train=False, debug_rollout_only=False, train_only=False, ema_rollout_policy="ema"
        )
        self.parallel_state = SimpleNamespace(get_mesh=lambda name: SimpleNamespace(get_local_rank=lambda: 0))
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.ema_shadow = EmaShadow([self.param], decay=0.5, flat_steps=100, keep_previous_ema=True)
        if restored is not None:
            with torch.no_grad():
                self.param.copy_(restored["param"])
            self.ema_shadow.load_state_dict(restored["ema"])
        self.weight_updater = SimpleNamespace(update_weights=self._capture_weight)
        self.records = []
        self.saves = {}
        Timer().start("train_wait")

    def _capture_weight(self):
        self.published = self.param.item()

    def update_weights(self):
        with patch.object(self.actor, "clear_memory"), patch.object(self.actor.dist, "get_rank", return_value=0):
            self.actor.FSDPTrainRayActor.update_weights(self)
        return self.published, self.ema_shadow.step

    def _train_core(self, rollout_id, rollout_data):
        start = time.monotonic()
        with self.ema_shadow.swap_in(use_previous_ema=True):
            reference = self.param.item()
        assert rollout_data["rollout_id"] == rollout_id
        time.sleep(self.delay)
        with torch.no_grad():
            self.param.add_(1)
        self.records.append((rollout_data, start, time.monotonic(), reference))

    def train(self, rollout_id, batch):
        with patch.object(self.actor.dist, "get_rank", return_value=0), patch.object(
            self.actor.train_metric_utils, "log_perf_data_raw"
        ):
            self.actor.FSDPTrainRayActor.train(self, rollout_id, batch)

    def save_model(self, rollout_id, force_sync=False):
        from copy import deepcopy

        self.saves[rollout_id] = deepcopy({"param": self.param.detach(), "ema": self.ema_shadow.state_dict()})

    def result(self):
        return self.records, self.saves


class TrainGroupProbe:
    def __init__(self, trainer, manager):
        self.trainer = trainer
        self.manager = manager
        self.updates = []

    def async_train(self, rollout_id, batch):
        return [self.trainer.train.remote(rollout_id, batch)]

    def update_weights(self):
        requested_at = time.monotonic()
        weight, ema_step = ray.get(self.trainer.update_weights.remote())
        self.updates.append((requested_at, ema_step))
        ray.get(self.manager.install.remote(weight))

    def save_model(self, rollout_id, force_sync=False):
        ray.get(self.trainer.save_model.remote(rollout_id, force_sync=force_sync))


@pytest.fixture(scope="module", autouse=True)
def ray_runtime():
    with tempfile.TemporaryDirectory(prefix="pr232-ray-") as directory:
        ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False, _temp_dir=directory)
        yield
        ray.shutdown()


def run_loop(tmp_path, monkeypatch, *, train_delay, rollout_delay, start=0, count=3, restored=None):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"input": str(i)}) + "\n" for i in range(16)))
    args = Namespace(
        start_rollout_id=start,
        num_rollout=count,
        save_interval=2,
        eval_interval=None,
        rollout_global_dataset=True,
        prompt_data=str(prompts),
        input_key="input",
        metadata_key="metadata",
        rollout_seed=42,
        rollout_shuffle=False,
        n_samples_per_prompt=1,
        save=str(tmp_path / "ckpt"),
        load=str(tmp_path / "ckpt") if restored else None,
        buffer_filter_path=None,
        use_wandb=False,
    )
    metrics = []
    monkeypatch.setattr("train_diffusion_async.tracking_utils.log", lambda args, data, **kw: metrics.append(data))
    manager = RolloutProbe.remote(args, rollout_delay)
    trainer = TrainerProbe.remote(manager, train_delay, restored)
    group = TrainGroupProbe(trainer, manager)
    try:
        group.update_weights()
        train_loop(args, group, manager, None)
        records, saves = ray.get(trainer.result.remote())
        events = ray.get(manager.get_events.remote())
        for rid in saves:
            saved = torch.load(f"{args.save}/rollout/global_dataset_state_dict_{rid}.pt")
            assert saved["sample_index"] == rid + 1
        return records, saves, events, group.updates, metrics
    finally:
        ray.kill(trainer)
        ray.kill(manager)


@pytest.mark.parametrize("train_delay,rollout_delay", [(0.05, 0.25), (0.25, 0.05)])
def test_overlap_reference_cursor_and_update_barrier(tmp_path, monkeypatch, train_delay, rollout_delay):
    records, saves, events, updates, metrics = run_loop(
        tmp_path, monkeypatch, train_delay=train_delay, rollout_delay=rollout_delay
    )
    assert [record[0]["sample_index"] for record in records] == [0, 1, 2]
    assert [record[0]["weight"] for record in records] == [0.0, 0.0, 0.5]
    assert [record[3] for record in records] == [record[0]["weight"] for record in records]
    assert [step for _, step in updates] == [1, 2, 3, 4]
    assert saves[1]["ema"]["step"] == 2
    for i in range(2):
        assert max(records[i][1], events[i + 1][1]) < min(records[i][2], events[i + 1][2])
        assert updates[i + 1][0] >= events[i + 1][2]
    if rollout_delay > train_delay:
        # Rollout 1 saves a checkpoint; its drain wait must not disappear into save().
        assert metrics[1]["perf/drain_wait_time"] > 0.05


def test_resume_rewarms_with_restored_ema(tmp_path, monkeypatch):
    _, saves, _, _, _ = run_loop(tmp_path, monkeypatch, train_delay=0, rollout_delay=0)
    records, _, _, updates, _ = run_loop(
        tmp_path, monkeypatch, train_delay=0, rollout_delay=0, start=2, count=5, restored=saves[1]
    )
    assert [r[0]["sample_index"] for r in records] == [2, 3, 4]
    assert [r[0]["weight"] for r in records] == [1.25, 1.25, 2.125]
    # The republish replays the EMA step the checkpoint preceded; the regenerated first batch
    # samples from that EMA while its reference is still the restored one.
    assert [r[3] for r in records] == [0.5, 1.25, 2.125]
    assert [step for _, step in updates] == [3, 4, 5, 6]


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_single_rollout(tmp_path, monkeypatch, count):
    records, _, events, updates, _ = run_loop(tmp_path, monkeypatch, train_delay=0, rollout_delay=0, count=count)
    assert len(records) == len(events) == count
    assert len(updates) == count + 1
