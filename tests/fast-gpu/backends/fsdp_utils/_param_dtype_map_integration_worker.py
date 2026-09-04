"""Four-GPU integration test for root-FQN compilation through ``apply_fsdp2``.

The test deliberately places overrides in both a child FSDP wrap and the root:

    user rules from model root
    +-----------------------------------+
    | block.full_precision.* -> FP32    |
    | root_scale              -> FP32   |
    +-----------------+-----------------+
                      |
                      v
                 apply_fsdp2
                      |
          +-----------+------------+
          | compile root FQNs       |
          | install the patch       |
          +-----------+------------+
                      |
          +-----------+-------------------------+
          |                                     |
          v                                     v
    fully_shard(block)                    fully_shard(model)
    full_precision.* -> FP32              root_scale -> FP32
    low_precision.*  -> BF16

A manually cast, unsharded model is the numerical reference. The worker checks
the gathered dtypes observed during forward, then requires bitwise-equal outputs
and reconstructed full gradients. This verifies the compiler and FSDP2 wiring
together; the lower-level patch communication paths are covered by PR #100.
"""

import copy
import os
from argparse import Namespace

import torch
import torch.distributed as dist
from torch import nn

from miles.backends.fsdp_utils.actor import apply_fsdp2
from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan


PARAM_DTYPE_PATTERNS = {
    "block.full_precision.*": "fp32",
    "root_scale": "fp32",
}


class MixedParamDtypeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.low_precision = nn.Linear(8, 15)
        self.full_precision = nn.Linear(8, 15)
        self.seen_param_dtypes = None

    def forward(self, x):
        self.seen_param_dtypes = (
            self.low_precision.weight.dtype,
            self.full_precision.weight.dtype,
        )
        low_precision_output = self.low_precision(x)
        full_precision_output = self.full_precision(x.to(self.full_precision.weight.dtype))
        return low_precision_output + full_precision_output.to(low_precision_output.dtype)


class MixedParamDtypeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.root_scale = nn.Parameter(torch.ones(1))
        self.block = MixedParamDtypeBlock()
        self.seen_root_dtype = None

    def forward(self, x):
        self.seen_root_dtype = self.root_scale.dtype
        output = self.block(x)
        return output * self.root_scale.to(output.dtype)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.use_deterministic_algorithms(True)

    torch.manual_seed(42)
    model = MixedParamDtypeModel().cuda()
    reference = copy.deepcopy(model)
    reference.block.low_precision.to(torch.bfloat16)
    model = apply_fsdp2(
        model,
        FSDPParallelPlan(
            no_split_modules=("MixedParamDtypeBlock",),
            param_dtype_patterns=PARAM_DTYPE_PATTERNS,
        ),
        args=Namespace(
            diffusion_forward_dtype="bf16",
            fsdp_reduce_dtype="fp32",
            fsdp_reshard_after_forward=True,
            gradient_checkpointing=False,
        ),
    )

    torch.manual_seed(43)
    inp = torch.randn(4, 8, device="cuda")
    output = model(inp)
    reference_output = reference(inp.to(torch.bfloat16))
    assert torch.equal(output, reference_output)
    assert model.block.seen_param_dtypes == (torch.bfloat16, torch.float32)
    assert model.seen_root_dtype == torch.float32

    output.sum().backward()
    reference_output.sum().backward()
    for (name, param), (reference_name, reference_param) in zip(
        model.named_parameters(),
        reference.named_parameters(),
        strict=True,
    ):
        assert name == reference_name
        assert param.grad is not None
        assert reference_param.grad is not None
        assert param.grad.dtype == torch.float32
        assert torch.equal(
            param.grad.full_tensor(),
            reference_param.grad.to(torch.float32),
        ), f"Gradient mismatch for {name}"

    if dist.get_rank() == 0:
        print("OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
