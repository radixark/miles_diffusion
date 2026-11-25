"""Two ranks read their own halves of a safetensors checkpoint.

    files hold the state dict   --> both ranks on meta --> rows [0:2] on rank 0, rows [2:4] on rank 1, GPU or CPU offload
                                    a buffer no file holds --> the value this rank's constructor computed
    another name in the files   --> ValueError on both ranks
    another name on rank 1 only --> the same error on both ranks: the ranks agree before either reads
    files seen by rank 1 only   --> ValueError on both ranks before either builds
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tests.fast.backends.fsdp_utils.test_reference_models import TinyPipelineConfig, make_model_loader_args
from tests.fast.backends.fsdp_utils.test_safetensors_load import FileBackend, save_checkpoint, saved_weight
from torch.distributed.device_mesh import init_device_mesh

from miles.backends.fsdp_utils.model_loader import load_fsdp_models


class RankTaggedBackend(FileBackend):
    """A buffer outside the state dict whose constructor value differs between the ranks; rank 0 sees no files
    under a checkpoint named `rank1-only`."""

    def safetensors_files(self, component, *, checkpoint_path):
        if Path(checkpoint_path).name == "rank1-only" and dist.get_rank() == 0:
            return None
        return super().safetensors_files(component, checkpoint_path=checkpoint_path)

    def load_component(self, component, **kwargs):
        model = super().load_component(component, **kwargs)
        model.register_buffer("built_on_rank", torch.tensor(float(dist.get_rank())), persistent=False)
        return model


def load(mesh, path, cpu_offload):
    parallel = SimpleNamespace(get_mesh=lambda name: mesh, get_optional_mesh=lambda name: None)
    args, backend = make_model_loader_args(), RankTaggedBackend()
    args.use_lora = False
    models = load_fsdp_models(
        args, backend, TinyPipelineConfig(), parallel, checkpoint_path=str(path), cpu_offload=cpu_offload
    )
    for model in models.values():
        shard = model.block.proj.weight.to_local()
        assert shard.shape == (2, 3) and shard.dtype == torch.float32
        assert shard.device.type == ("cpu" if cpu_offload else "cuda")
        assert model.forward_count.device.type == "cuda" and model.forward_count.item() == 0
        assert model.built_on_rank.device.type == "cuda" and model.built_on_rank.item() == dist.get_rank()
    return models, backend.materialize_requests


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    rank, root = dist.get_rank(), Path(sys.argv[1])
    if rank == 0:
        save_checkpoint(root / "state")
        save_checkpoint(root / "renamed", name="block.proj.kernel")
        save_checkpoint(root / "rank1-only")
    dist.barrier()
    inputs = torch.ones(2, 3, device="cuda")
    for cpu_offload in (False, True):
        models, materialize_requests = load(mesh, root / "state", cpu_offload)
        assert materialize_requests == [False, False]
        for component, model in models.items():
            expected = saved_weight(component).float()
            assert torch.equal(model.block.proj.weight.to_local().cpu(), expected[2 * rank : 2 * rank + 2])
            with torch.no_grad():
                torch.testing.assert_close(model(inputs).cpu(), (inputs.cpu() @ expected.T).tanh())

        for path in ("renamed", "renamed" if rank == 1 else "state"):
            # Both ranks raise, also when only rank 1's copy holds another name: rank 0 must not enter the read alone.
            try:
                load(mesh, root / path, cpu_offload)
            except ValueError as error:
                assert "do not hold the model's state dict" in str(error) and "rank 1" in str(error)
            else:
                raise AssertionError(f"{path}: expected a ValueError on every rank")
        try:
            load(mesh, root / "rank1-only", cpu_offload)
        except ValueError as error:
            assert "are seen on ranks [1] only" in str(error)
        else:
            raise AssertionError("rank1-only: expected a ValueError on every rank")
    dist.barrier()
    if rank == 0:
        print("OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
