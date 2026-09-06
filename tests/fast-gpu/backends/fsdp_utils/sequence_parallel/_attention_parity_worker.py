"""Four-rank worker for USP parity against full-sequence attention.

This is launched by ``test_attention_parity.py`` rather than discovered as a
standalone CI test. Every rank constructs the same full Q/K/V reference, then
runs USP from its sequence shard and compares its local output and input grads.
``--attention-backend`` selects the kernel family for the reference, the USP
local attention and the ring steps alike, mirroring ``--fsdp-attention-backend``.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.attention import SDPBackend, sdpa_kernel

from miles.backends.fsdp_utils import flash_attention_3
from miles.backends.fsdp_utils.parallel import build_sp_groups
from miles.backends.fsdp_utils.sequence_parallel.attention import usp_attention


SP_SIZE = 4
SHAPE = (2, 128, 8, 64)  # [batch, global sequence, heads, head dim]
DTYPE = torch.bfloat16

# These bounds compare production-dtype Ring attention against the same
# full-sequence reference kernel. The observed error is at most one bf16
# quantization step for this input band; the bounds leave only a small margin.
# Pure Ulysses is a lossless permutation around per-head attention, so both its
# forward and dQ/dK/dV are required to remain bitwise identical.
TOLERANCES = {
    2: {"forward": (8e-3, 1e-3), "backward": (8e-3, 2.5e-4)},
    1: {"forward": (8e-3, 1e-3), "backward": (8e-3, 2.5e-4)},
}

_SDPA_BACKENDS = {None: SDPBackend.FLASH_ATTENTION, "_native_cudnn": SDPBackend.CUDNN_ATTENTION}
KERNEL_TAGS = {None: "", "_native_cudnn": "-cudnn", "_flash_3": "-fa3"}
_ATTENTION_BACKEND = None
_DETERMINISTIC = False


def _local_attention(query, key, value):
    """Full-sequence attention on [B, S, H, D] with the selected kernel family."""
    scale = query.shape[-1] ** -0.5
    if _ATTENTION_BACKEND == "_flash_3":
        return flash_attention_3.flash3_attention(query, key, value, scale=scale, deterministic=_DETERMINISTIC)
    with sdpa_kernel(_SDPA_BACKENDS[_ATTENTION_BACKEND]):
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
    return output.transpose(1, 2).contiguous()


def _make_full_inputs(device):
    generator = torch.Generator(device=device).manual_seed(20260721)
    query = torch.randn(SHAPE, device=device, dtype=DTYPE, generator=generator) * 0.5
    key = torch.randn(SHAPE, device=device, dtype=DTYPE, generator=generator) * 0.5
    value = torch.randn(SHAPE, device=device, dtype=DTYPE, generator=generator) * 0.5
    grad_output = torch.randn(SHAPE, device=device, dtype=DTYPE, generator=generator) * 0.1
    for tensor in (query, key, value, grad_output):
        dist.broadcast(tensor, src=0)
    return query, key, value, grad_output


def _create_usp_groups(ulysses_degree):
    sp_mesh = init_device_mesh("cuda", (SP_SIZE,), mesh_dim_names=("sp",))
    ring_degree = SP_SIZE // ulysses_degree
    return build_sp_groups(sp_mesh, ring_degree, ulysses_degree)


def _run_reference(query, key, value, grad_output):
    query = query.detach().clone().requires_grad_(True)
    key = key.detach().clone().requires_grad_(True)
    value = value.detach().clone().requires_grad_(True)
    output = _local_attention(query, key, value)
    output.backward(grad_output)
    return output.detach(), (query.grad.detach(), key.grad.detach(), value.grad.detach())


def _run_usp(query, key, value, grad_output, ulysses_group, ring_group):
    rank = dist.get_rank()
    local_sequence = SHAPE[1] // SP_SIZE
    start = rank * local_sequence
    query = query[:, start : start + local_sequence].detach().clone().requires_grad_(True)
    key = key[:, start : start + local_sequence].detach().clone().requires_grad_(True)
    value = value[:, start : start + local_sequence].detach().clone().requires_grad_(True)
    local_grad_output = grad_output[:, start : start + local_sequence].contiguous()

    output = usp_attention(
        query,
        key,
        value,
        ulysses_group=ulysses_group,
        ring_group=ring_group,
        local_attention_fn=_local_attention,
        ring_backend=_ATTENTION_BACKEND,
        deterministic=_DETERMINISTIC,
    )
    output.backward(local_grad_output)
    return output.detach(), (query.grad.detach(), key.grad.detach(), value.grad.detach())


def _local_shard(tensor):
    local_sequence = SHAPE[1] // SP_SIZE
    start = dist.get_rank() * local_sequence
    return tensor[:, start : start + local_sequence].contiguous()


def _global_stats(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    max_abs = difference.max()
    normalized = max_abs / expected.float().abs().max().clamp_min(1e-12)
    mismatches = torch.count_nonzero(actual != expected).to(torch.int64)
    total = torch.tensor(actual.numel(), device=actual.device, dtype=torch.int64)
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    dist.all_reduce(normalized, op=dist.ReduceOp.MAX)
    dist.all_reduce(mismatches, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return max_abs.item(), normalized.item(), mismatches.item(), total.item()


def _assert_bitwise(name, actual, expected):
    max_abs, normalized, mismatches, total = _global_stats(actual, expected)
    if dist.get_rank() == 0:
        print(
            f"{name}: bitwise={mismatches == 0} mismatches={mismatches}/{total} "
            f"max_abs={max_abs:.3e} normalized={normalized:.3e}",
            flush=True,
        )
    assert mismatches == 0, f"{name} must be bitwise identical; {mismatches}/{total} values differ"


def _assert_close(name, actual, expected, *, rtol, atol):
    max_abs, normalized, mismatches, total = _global_stats(actual, expected)
    if dist.get_rank() == 0:
        print(
            f"{name}: bitwise={mismatches == 0} mismatches={mismatches}/{total} "
            f"max_abs={max_abs:.3e} normalized={normalized:.3e} rtol={rtol:.1e} atol={atol:.1e}",
            flush=True,
        )
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _enable_deterministic_mode():
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def main():
    global _ATTENTION_BACKEND, _DETERMINISTIC
    parser = argparse.ArgumentParser()
    parser.add_argument("--ulysses-degree", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--attention-backend", choices=("_native_cudnn", "_flash_3"), default=None)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    _ATTENTION_BACKEND = args.attention_backend
    _DETERMINISTIC = args.deterministic
    if args.deterministic:
        _enable_deterministic_mode()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    assert dist.get_world_size() == SP_SIZE

    ulysses_group, ring_group = _create_usp_groups(args.ulysses_degree)
    full_inputs = _make_full_inputs(device)
    topology = f"sp4-u{args.ulysses_degree}r{SP_SIZE // args.ulysses_degree}{KERNEL_TAGS[_ATTENTION_BACKEND]}"

    if args.deterministic:
        output_1, grads_1 = _run_usp(*full_inputs, ulysses_group, ring_group)
        output_2, grads_2 = _run_usp(*full_inputs, ulysses_group, ring_group)
        _assert_bitwise(f"{topology} deterministic forward", output_1, output_2)
        for name, actual, expected in zip(("dQ", "dK", "dV"), grads_1, grads_2, strict=True):
            _assert_bitwise(f"{topology} deterministic {name}", actual, expected)
        dist.barrier()
        if dist.get_rank() == 0:
            print(f"{topology} deterministic: PASS", flush=True)
        dist.destroy_process_group()
        return

    reference_output, reference_grads = _run_reference(*full_inputs)
    usp_output, usp_grads = _run_usp(*full_inputs, ulysses_group, ring_group)
    local_reference_output = _local_shard(reference_output)
    if args.ulysses_degree == SP_SIZE:
        _assert_bitwise(f"{topology} forward", usp_output, local_reference_output)
    else:
        rtol, atol = TOLERANCES[args.ulysses_degree]["forward"]
        _assert_close(f"{topology} forward", usp_output, local_reference_output, rtol=rtol, atol=atol)

    for name, actual, expected in zip(("dQ", "dK", "dV"), usp_grads, reference_grads, strict=True):
        local_expected = _local_shard(expected)
        if args.ulysses_degree == SP_SIZE:
            _assert_bitwise(f"{topology} {name}", actual, local_expected)
        else:
            rtol, atol = TOLERANCES[args.ulysses_degree]["backward"]
            _assert_close(f"{topology} {name}", actual, local_expected, rtol=rtol, atol=atol)

    dist.barrier()
    if dist.get_rank() == 0:
        print(f"{topology}: PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
