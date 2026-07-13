"""SP determinism smoke: the SP attention path (ulysses/ring) must be bitwise
deterministic under
torch.use_deterministic_algorithms (warn_only=False also asserts no op on the
path is registered nondeterministic). Runs forward+backward twice on identical
inputs and compares every grad bitwise. --no-det runs without the flag as a
control.

torchrun --standalone --nproc_per_node=2 tests/sp/sp_determinism_smoke.py --sp 2 --ulysses 1 [--no-det]
"""

import argparse
import os

import torch
import torch.distributed as dist
from diffusers import WanTransformer3DModel

from miles.backends.fsdp_utils.configs.wan2_2 import Wan2_2TrainPipelineConfig
from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend
from miles.backends.fsdp_utils.parallel import create_fsdp_parallel_state
from miles.backends.fsdp_utils.sp_attention import apply_sequence_parallel
from miles.utils.distributed_utils import init_gloo_group

DTYPE = torch.bfloat16


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sp", type=int, default=2)
    p.add_argument("--ulysses", type=int, default=1)
    p.add_argument("--no-det", action="store_true")
    cli = p.parse_args()

    if not cli.no_det:
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "set CUBLAS_WORKSPACE_CONFIG=:4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.cuda.current_device()
    init_gloo_group()

    args = argparse.Namespace(
        sequence_parallel_size=cli.sp,
        ulysses_degree=cli.ulysses,
    )
    ps = create_fsdp_parallel_state(args)

    torch.manual_seed(0)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=8,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=1024,
        num_layers=2,
        rope_max_seq_len=1024,
    ).to(device=device, dtype=DTYPE)
    model.train()
    for prm in model.parameters():
        dist.broadcast(prm.data, src=0)

    g = torch.Generator(device=device).manual_seed(123)
    hidden = torch.randn(1, 16, 4, 8, 8, device=device, dtype=DTYPE, generator=g)
    enc = torch.randn(1, 32, 4096, device=device, dtype=DTYPE, generator=g)
    ts = torch.tensor([500], device=device)
    out_grad = torch.randn(1, 16, 4, 8, 8, device=device, dtype=DTYPE, generator=g)
    for t in (hidden, enc, out_grad):
        dist.broadcast(t, src=0)

    plan = DiffusersModelBackend(Wan2_2TrainPipelineConfig()).sequence_parallel_plan(model)
    apply_sequence_parallel(model, ps, plan)

    def run_once():
        model.zero_grad(set_to_none=True)
        out = model(hidden_states=hidden, timestep=ts, encoder_hidden_states=enc, return_dict=False)[0]
        out.backward(out_grad)
        return out.detach().clone(), {n: prm.grad.detach().clone() for n, prm in model.named_parameters()}

    out1, g1 = run_once()
    out2, g2 = run_once()

    out_eq = torch.equal(out1, out2)
    diffs = [n for n in g1 if not torch.equal(g1[n], g2[n])]
    flag = torch.tensor([0 if (out_eq and not diffs) else 1], device=device)
    dist.all_reduce(flag)
    if rank == 0:
        mode = "no-det(control)" if cli.no_det else "deterministic"
        print(
            f"[DET-PROBE] mode={mode} u{ps.ulysses_degree}r{ps.ring_degree} forward_bitwise={out_eq} grad_mismatches={len(diffs)}"
        )
        for n in diffs[:8]:
            d = (g1[n].float() - g2[n].float()).abs().max().item()
            print(f"    diff {n}: max_abs={d:.3e}")
        print(f"[DET-PROBE {'OK' if flag.item() == 0 else 'NONDETERMINISTIC'}] (all ranks)")
    if not cli.no_det:
        assert flag.item() == 0, "SP attention path is not bitwise deterministic under deterministic mode"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
