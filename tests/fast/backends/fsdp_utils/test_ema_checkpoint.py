"""EMA checkpoint integration, including restoration of distributed shards."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="stage-a-cpu", labels=[])

import shutil
import subprocess
import sys
from argparse import Namespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from miles.backends.fsdp_utils import checkpoint
from miles.backends.fsdp_utils.ema import EmaShadow


def make_actor(tmp_path):
    model = torch.nn.Linear(3, 5, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    return Namespace(
        model=model,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0),
        ema_shadow=EmaShadow(model.parameters(), decay=0.5, flat_steps=10, keep_previous_ema=True),
        global_step=2,
        micro_step=0,
        train_pipeline_config=Namespace(optimizer_state_allowed_missing=[]),
        args=Namespace(
            save=str(tmp_path),
            load=str(tmp_path),
            ckpt_step=None,
            use_lora=False,
            no_save_optim=False,
            no_load_optim=False,
            no_load_rng=True,
            start_rollout_id=0,
        ),
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_restores_ema_and_restarts_reference(tmp_path, monkeypatch, legacy):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    try:
        original = make_actor(tmp_path)
        for _ in range(2):
            with torch.no_grad():
                original.model.weight.add_(1)
            original.ema_shadow.update()
        checkpoint.save(original, iteration=1)
        if legacy:
            shutil.rmtree(tmp_path / "iter_0000002/ema")
        restored = make_actor(tmp_path)
        payload = checkpoint.load(restored)
        # Actor initializes its EMA from the already-restored live model.
        restored.ema_shadow = EmaShadow(restored.model.parameters(), decay=0.5, flat_steps=10, keep_previous_ema=True)
        checkpoint.finalize_load(restored, payload)
        assert restored.args.start_rollout_id == 2
        assert restored.ema_shadow.step == 2
        expected = original.model.weight if legacy else original.ema_shadow.shadow[0]
        torch.testing.assert_close(restored.ema_shadow.shadow[0], expected)
        torch.testing.assert_close(restored.ema_shadow.previous_ema[0], expected)
        torch.testing.assert_close(restored.model.weight, original.model.weight)
        if not legacy:
            assert original.ema_shadow.update() == restored.ema_shadow.update()
            torch.testing.assert_close(restored.ema_shadow.shadow[0], original.ema_shadow.shadow[0])
    finally:
        dist.destroy_process_group()


def test_ema_checkpoint_reshards_to_single_process(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--module",
            "tests.fast.backends.fsdp_utils._ema_checkpoint_worker",
            str(tmp_path),
        ],
        check=True,
        timeout=180,
    )
    param = torch.nn.Parameter(torch.zeros(5, 3))
    restored = EmaShadow([param], keep_previous_ema=True)
    dcp.load({"ema": restored}, checkpoint_id=str(tmp_path / "ema"))
    assert restored.step == 2
    torch.testing.assert_close(restored.shadow[0], torch.arange(15).reshape(5, 3).float() + 1.25)
    torch.testing.assert_close(restored.previous_ema[0], restored.shadow[0])
