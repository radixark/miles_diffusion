"""No-download FA3 / pure-Ulysses regression worker; launch with torchrun.

Exercises the actual Diffusers dispatcher and Miles USP patch, upstream FA3
autograd, and a replicated trainable projection. This is not a model/FSDP test.
Fixed-topology repeats must be bitwise equal; cross-topology and FP32-reference
comparisons use explicit numerical tolerances. No fallback kernels are allowed.
"""

import argparse
import functools
import importlib.metadata
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.attention import SDPBackend, sdpa_kernel


class _AttentionModel:
    """Only supplies the processor/module protocol used by the production patch."""

    def __init__(self):
        self.attn_processors = {"attention": SimpleNamespace()}


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    # S=128 may miss sequence-dependent backward nondeterminism; keep that
    # shape available explicitly as a smoke test, not the default regression.
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.repeats < 2 or min(args.batch_size, args.seq_len, args.heads, args.head_dim) < 1:
        parser.error("repeats must be >= 2 and tensor dimensions must be positive")
    if args.timeout_seconds < 1 or any(x is not None and x < 0 for x in (args.rtol, args.atol)):
        parser.error("timeout must be positive and tolerances must be nonnegative")
    return args


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _shard(tensor):
    return tensor.chunk(dist.get_world_size(), dim=1)[dist.get_rank()].contiguous()


def _math_attention(q, k, v):
    # All arithmetic is FP32 and the math backend is forced, so this cannot
    # silently reuse the FA3 implementation under test.
    with sdpa_kernel(SDPBackend.MATH):
        return (
            F.scaled_dot_product_attention(
                q.float().transpose(1, 2),
                k.float().transpose(1, 2),
                v.float().transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            )
            .transpose(1, 2)
            .contiguous()
        )


def _forward_backward(attention, inputs, *, fp32=False):
    qkv = [x.detach().clone().to(torch.float32 if fp32 else x.dtype).requires_grad_() for x in inputs[:3]]
    output = attention(*qkv)
    output.backward(inputs[3].to(output.dtype))
    return (output.detach(), *(x.grad.detach() for x in qkv))


def _check(name, actual, expected, report, *, rtol, atol, exact=False):
    assert actual.shape == expected.shape, f"{name}: shape mismatch {actual.shape} vs {expected.shape}"
    a, e = actual.float(), expected.float()
    finite = torch.isfinite(a) & torch.isfinite(e)
    delta = (a - e).abs()
    if exact:
        assert actual.dtype == expected.dtype
        # Check representations too, including signed zero. All tensors checked
        # exactly here are BF16/FP16 outputs or input gradients.
        failed = (actual.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)) | ~finite
    else:
        failed = (delta > atol + rtol * e.abs()) | ~finite
    errors = torch.stack((delta.nan_to_num(nan=float("inf")).max(), e.abs().max()))
    counts = torch.tensor([failed.sum().item(), actual.numel()], device=actual.device, dtype=torch.int64)
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    record = {
        "name": name,
        "max_abs": errors[0].item(),
        "max_abs_over_reference_peak": (errors[0] / errors[1].clamp_min(1e-12)).item(),
        "failed_elements": counts[0].item(),
        "elements": counts[1].item(),
        "exact": exact,
        "rtol": 0 if exact else rtol,
        "atol": 0 if exact else atol,
    }
    report["checks"].append(record)
    if dist.get_rank() == 0:
        print(json.dumps(record), flush=True)
    assert record["failed_elements"] == 0, f"{name}: {record}; do not widen tolerances without investigating"


def _projection_step(attention, x, initial_weight, dout, *, distributed, dtype):
    weight = torch.nn.Parameter(initial_weight.clone())
    optimizer = torch.optim.SGD([weight], lr=0.05)
    if distributed:
        x, dout = _shard(x), _shard(dout)
    # FP32 master weight, low-precision attention, no autocast or hidden scaler.
    projected = F.linear(x, weight).to(dtype)
    q, k, v = projected.reshape(*x.shape[:2], 3, *dout.shape[2:]).unbind(dim=2)
    output = attention(q.contiguous(), k.contiguous(), v.contiguous())
    loss = (output.float() * dout.float()).sum()
    loss.backward()
    if distributed:
        # Each rank owns a disjoint part of the global SUM objective. Summing
        # shared-parameter gradients matches the full-sequence objective.
        dist.all_reduce(weight.grad, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss.detach(), op=dist.ReduceOp.SUM)
    gradient = weight.grad.detach().clone()
    optimizer.step()
    return loss.detach(), gradient, weight.detach()


def _run(args, report):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this worker never skips or substitutes CPU/SDPA for FA3")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 9:
        raise RuntimeError(f"This regression requires an SM90+ GPU and a compatible FA3 build, got {capability}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching torchrun")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=args.timeout_seconds))
    world = dist.get_world_size()
    if world not in (1, 2, 4) or args.seq_len % world or args.heads % world:
        raise ValueError("world size must be 1, 2, or 4; seq-len and heads must be divisible by it")

    from diffusers.models import attention_dispatch

    if not callable(attention_dispatch.flash_attn_3_func):
        raise RuntimeError("FA3 must be installed and importable by Diffusers; FA2 is not sufficient")

    from miles.backends.fsdp_utils.fa3_attention import install_diffusers_fa3_attention
    from miles.backends.fsdp_utils.parallel import build_sp_groups
    from miles.backends.fsdp_utils.sequence_parallel.diffusers_dispatch import install_diffusers_usp_patch

    install_diffusers_fa3_attention(deterministic=True)
    raw_fa3 = attention_dispatch.flash_attn_3_func
    calls = []
    expected_deterministic = True

    @functools.wraps(raw_fa3)
    def traced_fa3(*call_args, **kwargs):
        assert (
            kwargs.get("deterministic") is expected_deterministic
        ), f"FA3 received deterministic={kwargs.get('deterministic')}, expected {expected_deterministic}"
        assert kwargs.get("num_splits") == 1, f"FA3 received num_splits={kwargs.get('num_splits')}"
        calls.append({"deterministic": kwargs["deterministic"], "num_splits": kwargs.get("num_splits")})
        return raw_fa3(*call_args, **kwargs)

    attention_dispatch.flash_attn_3_func = traced_fa3
    # The real patch finds the model's module and replaces this imported symbol,
    # exactly as it does for a Diffusers transformer module.
    global dispatch_attention_fn
    dispatch_attention_fn = attention_dispatch.dispatch_attention_fn
    model = _AttentionModel()
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("sp",)) if world > 1 else None
    ulysses_group, ring_group = build_sp_groups(mesh, ring_degree=1, ulysses_degree=world)
    install_diffusers_usp_patch(model, SimpleNamespace(ulysses_group=ulysses_group, ring_group=ring_group))
    config = model.attn_processors["attention"]._parallel_config

    def dispatched(q, k, v):
        return dispatch_attention_fn(q, k, v, backend="_flash_3", parallel_config=config)

    def reference(q, k, v):
        # Bypass both Miles' adapter and Diffusers' dispatcher.
        out = raw_fa3(q, k, v, causal=False, deterministic=True, num_splits=1)
        return out[0] if isinstance(out, tuple) else out

    dtype = getattr(torch, args.dtype)
    # Two relative rounding steps plus a small absolute allowance for near-zero
    # values in these bounded random inputs. These are proposed bounds, not
    # measured GPU error claims; overrides are always written to the report.
    rtol = args.rtol if args.rtol is not None else 2 * torch.finfo(dtype).eps
    atol = args.atol if args.atol is not None else torch.finfo(dtype).eps / 16
    report.update(
        {
            "world_size": world,
            "ulysses_degree": world,
            "ring_degree": 1,
            "device": torch.cuda.get_device_name(device),
            "capability": capability,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "diffusers": _version("diffusers"),
            "flash_attn_3": _version("flash_attn_3"),
            "nccl": torch.cuda.nccl.version(),
            "rtol": rtol,
            "atol": atol,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (args.batch_size, args.seq_len, args.heads, args.head_dim)
    full = [
        torch.randn(shape, device=device, dtype=dtype, generator=generator) * scale for scale in (0.5, 0.5, 0.5, 0.1)
    ]
    for tensor in full:
        dist.broadcast(tensor, src=0)
    local = [_shard(tensor) for tensor in full]
    fa3_ref = _forward_backward(reference, full)
    math_ref = _forward_backward(_math_attention, full, fp32=True)
    first = _forward_backward(dispatched, local)
    for name, actual, direct, math in zip(("out", "dQ", "dK", "dV"), first, fa3_ref, math_ref, strict=True):
        _check(f"fa3-parity/{name}", actual, _shard(direct), report, rtol=rtol, atol=atol)
        _check(f"fp32-math-parity/{name}", actual, _shard(math), report, rtol=rtol, atol=atol)
    for repeat in range(1, args.repeats):
        repeated = _forward_backward(dispatched, local)
        for name, actual, expected in zip(("out", "dQ", "dK", "dV"), repeated, first, strict=True):
            _check(f"repeat-{repeat}/{name}", actual, expected, report, rtol=0, atol=0, exact=True)

    x = torch.randn((*shape[:2], 32), device=device, generator=generator) * 0.5
    weight = torch.randn((3 * args.heads * args.head_dim, 32), device=device, generator=generator) / 32**0.5
    for tensor in (x, weight):
        dist.broadcast(tensor, src=0)
    reference_step = _projection_step(reference, x, weight, full[3], distributed=False, dtype=dtype)
    local_step = _projection_step(dispatched, x, weight, full[3], distributed=True, dtype=dtype)
    for name, actual, expected in zip(("loss", "dWeight", "updatedWeight"), local_step, reference_step, strict=True):
        _check(f"projection/{name}", actual, expected, report, rtol=rtol, atol=atol)

    # The adapter also replaces the default (mode-off) backend: verify that its
    # forward/backward works and the flag really turns off. No nondeterminism
    # or repeat-equality claim is made for this mode.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    expected_deterministic = False
    install_diffusers_fa3_attention(deterministic=False)
    mode_off = _forward_backward(dispatched, local)
    for name, actual, expected in zip(("out", "dQ", "dK", "dV"), mode_off, fa3_ref, strict=True):
        _check(f"mode-off-parity/{name}", actual, _shard(expected), report, rtol=rtol, atol=atol)
    report["mode_off_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    # Each dispatched forward must reach the traced upstream function, proving
    # that neither the repeat nor projection test silently fell back to SDPA.
    assert len(calls) == args.repeats + 2, f"Expected {args.repeats + 2} real FA3 calls, observed {len(calls)}"
    report["kernel_calls_per_rank"] = calls
    peak_memory = [None] * world
    dist.all_gather_object(peak_memory, torch.cuda.max_memory_allocated(device))
    report["peak_allocated_bytes_by_rank"] = peak_memory
    torch.cuda.synchronize(device)
    dist.barrier()


def main():
    args = _arguments()
    report = {
        "status": "running",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checks": [],
    }
    start = time.monotonic()
    rank = int(os.environ.get("RANK", "0"))
    try:
        _run(args, report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        if args.output_json is not None:
            path = args.output_json
            if rank == 0:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, indent=2) + "\n")
            if report["status"] != "passed":
                path = path.with_name(f"{path.stem}.rank{rank}.failed{path.suffix}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, indent=2) + "\n")
        if rank == 0:
            print(f"FA3 pure-Ulysses: {report['status']} ({report['elapsed_seconds']:.2f}s)", flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
