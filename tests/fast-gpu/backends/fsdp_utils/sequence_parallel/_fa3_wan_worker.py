"""Real tiny Wan + production FSDP2/Ulysses, without checkpoints or downloads.

Two ranks compare DP2/SP1 with DP1/SP2 at the same global batch, using the
production gather backward and FSDP mean. This is model integration, not a
Ray/SGLang rollout or pretrained-model quality test. CPU sanity explicitly uses
native FP32 attention; it never constitutes FA3 evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--cpu-sanity", action="store_true")
    parser.add_argument("--backend", choices=("_flash_3", "native"), default="_flash_3")
    parser.add_argument("--seq-len", type=int, choices=(128, 1024), default=1024)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--output-rel-l2", type=float, default=0.02)
    parser.add_argument("--update-rel-l2", type=float, default=0.05)
    parser.add_argument("--normalized-max", type=float, default=0.10)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.cpu_sanity != (args.backend == "native"):
        parser.error("CPU control requires both --cpu-sanity --backend native; GPU regression requires _flash_3")
    if args.repeats < 2 or args.lr <= 0:
        parser.error("repeats must be >=2 and lr must be positive")
    for name in ("output_rel_l2", "update_rel_l2", "normalized_max"):
        if not 0 < getattr(args, name) < 0.5:
            parser.error(f"{name} must be between 0 and 0.5")
    return args


def full_cpu(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach()
    if hasattr(value, "full_tensor"):
        value = value.full_tensor()
    return value.float().cpu().contiguous().clone()


def cpu_parallel_state(sp_size: int):
    """Same production meshes/groups, with an explicit CPU device for the control."""
    from miles.backends.fsdp_utils.parallel import build_fsdp_meshes, build_sp_groups
    from miles.backends.training_utils.parallel import ParallelState
    from miles.utils.distributed_utils import get_gloo_group

    meshes = build_fsdp_meshes("cpu", 2, 1, sp_size)
    sp = meshes.get("sp")
    dp = meshes["dp"]
    ulysses, ring = build_sp_groups(sp, 1, sp_size)
    return ParallelState(
        dp_rank=dp.get_local_rank(),
        dp_src_rank=0,
        dp_size=dp.size(),
        dp_group=dp.get_group(),
        sp_rank=sp.get_local_rank() if sp is not None else 0,
        sp_size=sp_size,
        sp_group=sp.get_group() if sp is not None else None,
        ulysses_degree=sp_size,
        ring_degree=1,
        ulysses_group=ulysses,
        ring_group=ring,
        dp_sp_rank=dist.get_rank(),
        dp_sp_size=2,
        dp_sp_group_gloo=get_gloo_group(),
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        meshes=meshes,
    )


def error_metrics(actual: torch.Tensor, reference: torch.Tensor, l2_floor: float, peak_floor: float) -> dict:
    if actual.shape != reference.shape or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise AssertionError("Shape mismatch or nonfinite comparison tensor")
    difference = actual.double() - reference.double()
    return {
        "relative_l2": float(difference.norm() / max(float(reference.double().norm()), l2_floor, 1e-30)),
        "normalized_max": float(difference.abs().max()) / max(float(reference.abs().max()), peak_floor, 1e-30),
        "max_abs": float(difference.abs().max()),
        "exact": torch.equal(actual.reshape(-1).view(torch.uint8), reference.reshape(-1).view(torch.uint8)),
    }


def compare_tensors(
    report: dict,
    label: str,
    actual: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    relative_limit: float,
    maximum_limit: float,
    exact: bool = False,
) -> None:
    if actual.keys() != reference.keys():
        raise AssertionError(f"{label}: tensor names differ")
    ref_flat = torch.cat([value.reshape(-1) for value in reference.values()])
    value_flat = torch.cat([value.reshape(-1) for value in actual.values()])
    # Relative error is ill-defined for nearly-zero parameter gradients. These
    # declared group-relative floors apply only to per-tensor checks; the full
    # concatenation always has a strict, unfloored relative-error check.
    l2_floor = float(ref_flat.double().norm()) * 1e-6
    peak_floor = float(ref_flat.abs().max()) * 1e-4
    pairs = [("all", value_flat, ref_flat, 0.0, 0.0)]
    pairs += [(name, actual[name], value, l2_floor, peak_floor) for name, value in reference.items()]
    for name, value, target, norm_floor, max_floor in pairs:
        metrics = error_metrics(value, target, norm_floor, max_floor)
        passed = (
            metrics["exact"]
            if exact
            else (metrics["relative_l2"] <= relative_limit and metrics["normalized_max"] <= maximum_limit)
        )
        report["checks"].append({"name": f"{label}/{name}", "passed": passed, **metrics})
        if not passed:
            raise AssertionError(f"{label}/{name}: {metrics}; limits L2={relative_limit}, max={maximum_limit}")


def compare_runs(report: dict, label: str, actual: list[dict], reference: list[dict], args, *, exact=False) -> None:
    for step, (value, target) in enumerate(zip(actual, reference, strict=True)):
        for group in ("output", "loss", "gradients", "deltas"):
            limit = args.output_rel_l2 if group in ("output", "loss") else args.update_rel_l2
            compare_tensors(
                report,
                f"{label}/step{step + 1}/{group}",
                value[group],
                target[group],
                relative_limit=limit,
                maximum_limit=args.normalized_max,
                exact=exact,
            )


class KernelTrace:
    """Observe actual FA3 calls, including forward re-entry during backward."""

    def __init__(self, report: dict):
        self.report = report
        self.case = "setup"
        self.step = 0
        self.phase = "setup"
        self.layers: list[str] = []
        self.original = None

    def install(self) -> None:
        import diffusers.models.attention_dispatch as dispatch

        self.original = dispatch.flash_attn_3_func

        @functools.wraps(self.original)
        def traced(*args, **kwargs):
            q = kwargs.get("q", args[0] if args else None)
            k = kwargs.get("k", args[1] if len(args) > 1 else None)
            v = kwargs.get("v", args[2] if len(args) > 2 else None)
            if any(value.dtype != torch.bfloat16 or value.device != q.device for value in (q, k, v)):
                raise AssertionError("Real FA3 Q/K/V must all be BF16 on the same CUDA device")
            if q.device.type != "cuda" or k.shape != v.shape:
                raise AssertionError("Real FA3 requires CUDA Q/K/V and matching K/V shapes")
            record = {
                "case": self.case,
                "step": self.step,
                "phase": self.phase,
                "layer": self.layers[-1] if self.layers else None,
                "q_shape": list(q.shape),
                "k_shape": list(k.shape),
                "dtype": str(q.dtype),
                "deterministic": kwargs.get("deterministic"),
                "num_splits": kwargs.get("num_splits"),
                "completed": False,
            }
            self.report["kernel_calls"].append(record)
            if record["layer"] is None or record["deterministic"] is not True or record["num_splits"] != 1:
                raise AssertionError(f"Missing layer context or incorrect real FA3 flags: {record}")
            result = self.original(*args, **kwargs)
            record["completed"] = True
            return result

        dispatch.flash_attn_3_func = traced

    def instrument(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            if name.endswith((".attn1", ".attn2")):

                def enter(module, inputs, layer=name):
                    self.layers.append(layer)

                def leave(module, inputs, output):
                    self.layers.pop()

                module.register_forward_pre_hook(enter)
                module.register_forward_hook(leave, always_call=True)
            elif name.endswith(".norm2"):

                def observe_norm(module, inputs, layer=name):
                    dtype = module.weight.dtype
                    self.report["norm2_calls"].append(
                        {"case": self.case, "phase": self.phase, "layer": layer, "weight_dtype": str(dtype)}
                    )
                    if dtype != torch.float32:
                        raise AssertionError(f"Wan {layer} lost its production FP32 parameter override")

                module.register_forward_pre_hook(observe_norm)

    def verify(self, case: str, sp_size: int, checkpoint: bool, seq_len: int) -> None:
        calls = [call for call in self.report["kernel_calls"] if call["case"] == case]
        for step in (1, 2):
            for layer in ("blocks.0.attn1", "blocks.0.attn2", "blocks.1.attn1", "blocks.1.attn2"):
                for phase, expected in (("forward", 1), ("backward", int(checkpoint))):
                    matching = [c for c in calls if c["step"] == step and c["layer"] == layer and c["phase"] == phase]
                    if len(matching) != expected:
                        raise AssertionError(
                            f"{case}/{step}/{layer}/{phase}: expected {expected} calls, got {len(matching)}"
                        )
                    for call in matching:
                        if phase == "forward" and not call["completed"]:
                            raise AssertionError(f"Original forward FA3 did not return successfully: {call}")
                        self_attention = layer.endswith("attn1")
                        batch = 1 if sp_size == 1 else 2
                        heads = 4 // sp_size if self_attention else 4
                        q_len = seq_len if self_attention else seq_len // sp_size
                        k_len = seq_len if self_attention else 17
                        if call["q_shape"] != [batch, q_len, heads, 64] or call["k_shape"] != [
                            batch,
                            k_len,
                            heads,
                            64,
                        ]:
                            raise AssertionError(f"Incorrect self/cross attention topology: {call}")
        if self.layers:
            raise AssertionError(f"Unbalanced attention contexts: {self.layers}")
        norms = [call for call in self.report["norm2_calls"] if call["case"] == case]
        if len(norms) != 4 * (1 + int(checkpoint)):
            raise AssertionError(f"{case}: missing live norm2 precision/recompute observations")


def build_model(args, trace, initial, config, device, sp_size, checkpoint):
    from diffusers import WanTransformer3DModel

    from miles.backends.fsdp_utils.actor import apply_fsdp2
    from miles.backends.fsdp_utils.configs.wan2_2 import Wan2_2TrainPipelineConfig
    from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend
    from miles.backends.fsdp_utils.parallel import create_fsdp_parallel_state
    from miles.backends.fsdp_utils.sequence_parallel.plan import apply_sequence_parallel

    parallel_args = argparse.Namespace(sequence_parallel_size=sp_size, ulysses_degree=sp_size, dp_replicate_size=1)
    parallel = cpu_parallel_state(sp_size) if args.cpu_sanity else create_fsdp_parallel_state(parallel_args)
    family = Wan2_2TrainPipelineConfig()
    backend = DiffusersModelBackend(family)
    backend.enable_deterministic_attention(args.backend)
    model = WanTransformer3DModel(**config).to(device)
    model.load_state_dict(initial)
    backend.set_attention_backend(model, args.backend)
    if checkpoint:
        backend.enable_gradient_checkpointing(model)
    model.train()
    precision = argparse.Namespace(
        diffusion_forward_dtype="fp32" if args.cpu_sanity else "bf16", fsdp_reduce_dtype="fp32"
    )
    model = apply_fsdp2(model, backend.fsdp_parallel_plan(model), mesh=parallel.get_mesh("fsdp"), args=precision)
    family.postprocess_model_after_materialize(model)
    if sp_size > 1:
        apply_sequence_parallel(
            model, parallel, backend.sequence_parallel_plan(model), backend.install_sequence_parallel_attention
        )
    trace.instrument(model)
    return model, parallel


def run_case(
    args,
    report: dict,
    trace: KernelTrace,
    initial: dict,
    batches: list[dict],
    config: dict,
    device: torch.device,
    sp_size: int,
    checkpoint: bool,
    repeat: int,
) -> list[dict]:
    case = f"sp{sp_size}-checkpoint{int(checkpoint)}-repeat{repeat}"
    trace.case = case
    print(f"[rank {dist.get_rank()}] {case}", flush=True)
    model, parallel = build_model(args, trace, initial, config, device, sp_size, checkpoint)
    parameters = dict(model.named_parameters())
    if any(parameter.dtype != torch.float32 for parameter in parameters.values()):
        raise AssertionError("Master parameters must remain FP32")
    starting = {name: full_cpu(parameter) for name, parameter in parameters.items()}
    if any(not torch.equal(starting[name], initial[name]) for name in starting):
        raise AssertionError("Cases did not start from identical full FP32 parameters")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    snapshots = []
    for step, batch in enumerate(batches, start=1):
        trace.step, trace.phase = step, "forward"
        optimizer.zero_grad(set_to_none=True)
        selection = slice(parallel.dp_rank, parallel.dp_rank + 1) if sp_size == 1 else slice(None)
        inputs = {name: value[selection].to(device) for name, value in batch.items()}
        previous = {name: full_cpu(parameter) for name, parameter in parameters.items()}
        compute = contextlib.nullcontext() if args.cpu_sanity else torch.autocast("cuda", dtype=torch.bfloat16)
        with compute:
            output = model(
                hidden_states=inputs["latent"],
                timestep=inputs["timestep"],
                encoder_hidden_states=inputs["text"],
                return_dict=False,
            )[0]
            # Equivalent to sft_loss_formula's per-pair FP32 MSE sum divided
            # by local pair count. SP loss must not be divided again.
            loss = (output.float() - inputs["target"]).square().mean()
        trace.phase = "backward"
        loss.backward()
        missing = [name for name, parameter in parameters.items() if parameter.grad is None]
        if missing:
            raise AssertionError(f"Missing trainable gradients: {missing}")
        gradients = {name: full_cpu(parameter.grad) for name, parameter in parameters.items()}
        if any(not torch.isfinite(value).all() for value in gradients.values()):
            raise AssertionError("Nonfinite parameter gradient")
        gradient_norm = float(torch.cat([value.flatten() for value in gradients.values()]).norm())
        if gradient_norm == 0:
            raise AssertionError("All gradients are zero")
        trace.phase = "optimizer"
        optimizer.step()
        updated = {name: full_cpu(parameter) for name, parameter in parameters.items()}
        deltas = {name: updated[name] - starting[name] for name in parameters}
        step_norm = float(torch.cat([(updated[name] - previous[name]).flatten() for name in parameters]).norm())
        if step_norm == 0 or any(not torch.isfinite(value).all() for value in updated.values()):
            raise AssertionError("Optimizer did not produce a finite nonzero update")
        output_parts = [torch.empty_like(output) for _ in range(2)]
        dist.all_gather(output_parts, output.detach())
        if sp_size == 2 and not torch.equal(output_parts[0], output_parts[1]):
            raise AssertionError("SP ranks did not gather identical full outputs")
        global_output = torch.cat(output_parts) if sp_size == 1 else output_parts[0]
        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss)
        global_loss /= 2  # Reporting only; does not change loss.backward().
        snapshots.append(
            {
                "output": {"tensor": full_cpu(global_output)},
                "loss": {"scalar": full_cpu(global_loss)},
                "gradients": gradients,
                "deltas": deltas,
            }
        )
        report["steps"].append(
            {
                "case": case,
                "step": step,
                "loss": float(global_loss),
                "gradient_norm": gradient_norm,
                "step_delta_norm": step_norm,
                "trainable_tensors": len(parameters),
            }
        )
    if not args.cpu_sanity:
        trace.verify(case, sp_size, checkpoint, args.seq_len)
    del optimizer, model, parameters
    gc.collect()
    if not args.cpu_sanity:
        torch.cuda.empty_cache()
    dist.barrier()
    return snapshots


def run_matrix(args, report, trace, initial, batches, config, device) -> None:
    references = {}
    for checkpoint in (False, True):
        for sp_size in (1, 2):
            reference = run_case(args, report, trace, initial, batches, config, device, sp_size, checkpoint, 0)
            for repeat in range(1, args.repeats):
                repeated = run_case(args, report, trace, initial, batches, config, device, sp_size, checkpoint, repeat)
                compare_runs(
                    report,
                    f"repeat/sp{sp_size}/checkpoint{int(checkpoint)}/{repeat}",
                    repeated,
                    reference,
                    args,
                    exact=True,
                )
            references[checkpoint, sp_size] = reference
        compare_runs(
            report,
            f"topology/checkpoint{int(checkpoint)}",
            references[checkpoint, 2],
            references[checkpoint, 1],
            args,
        )
    for sp_size in (1, 2):
        compare_runs(report, f"checkpoint/sp{sp_size}", references[True, sp_size], references[False, sp_size], args)
    # Assert the declared metric rejects the regressions this test targets.
    baseline = references[False, 1]
    final_delta = torch.cat([value.flatten() for value in baseline[1]["deltas"].values()])
    missed_step = torch.cat([value.flatten() for value in baseline[0]["deltas"].values()])
    report["negative_controls"] = {}
    for name, bad in (
        ("double_update", final_delta * 2),
        ("zero_update", torch.zeros_like(final_delta)),
        ("missing_second_step", missed_step),
    ):
        metrics = error_metrics(bad, final_delta, 0, 0)
        rejected = metrics["relative_l2"] > args.update_rel_l2 or metrics["normalized_max"] > args.normalized_max
        report["negative_controls"][name] = {"rejected": rejected, **metrics}
        if not rejected:
            raise AssertionError(f"Tolerance is insensitive to {name}")


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("Launch this worker with torchrun --nproc_per_node=2")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    device = torch.device("cpu" if args.cpu_sanity else f"cuda:{os.environ['LOCAL_RANK']}")
    if not args.cpu_sanity:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use explicit --cpu-sanity --backend native only for CPU control")
        torch.cuda.set_device(device)
        if torch.cuda.get_device_capability()[0] < 9:
            raise RuntimeError("This FA3 regression requires an SM90+ GPU and compatible FA3 installation")
        import diffusers.models.attention_dispatch as dispatch

        if not callable(dispatch.flash_attn_3_func):
            raise RuntimeError("Real FA3 kernel missing; no SDPA fallback is allowed")
    dist.init_process_group("gloo" if args.cpu_sanity else "nccl", timeout=timedelta(seconds=args.timeout_seconds))
    from diffusers import WanTransformer3DModel

    from miles.utils.distributed_utils import init_gloo_group

    init_gloo_group()
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    if args.cpu_sanity:
        args.output_rel_l2 = args.update_rel_l2 = 1e-4
        args.normalized_max = 1e-3
    config = dict(
        patch_size=(1, 2, 2),
        num_attention_heads=4,
        attention_head_dim=64,
        in_channels=4,
        out_channels=4,
        text_dim=32,
        freq_dim=32,
        ffn_dim=512,
        num_layers=2,
    )
    template = WanTransformer3DModel(**config)
    initial = {name: value.detach().clone() for name, value in template.state_dict().items()}
    del template
    # Two frames exercise temporal RoPE; SP2 boundaries separate the frames.
    # T2/H32/W64 -> 1024 patch tokens; T2/H16/W16 -> 128 for explicit smoke.
    height, width = (32, 64) if args.seq_len == 1024 else (16, 16)
    batches = [
        dict(
            latent=torch.randn(2, 4, 2, height, width),
            text=torch.randn(2, 17, 32),
            timestep=torch.tensor([800.0 - 100 * step, 700.0 - 100 * step]),
            target=torch.randn(2, 4, 2, height, width),
        )
        for step in range(2)
    ]
    report: dict[str, Any] = {
        "status": "running",
        "mode": "native_fp32_cpu_control" if args.cpu_sanity else "real_fa3_bf16",
        "scope": "tiny real Wan; production FSDP2 + Ulysses; DP2/SP1 vs DP1/SP2; two SGD updates",
        "not_tested": [
            "pretrained weights",
            "Ray actor lifecycle",
            "SGLang rollout",
            "full RL E2E",
            "model quality",
            "2D per-token TI2V timestep boundary",
        ],
        "world_size": 2,
        "global_batch_size": 2,
        "model_config": config,
        "global_input_shapes": {name: list(value.shape) for name, value in batches[0].items()},
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "tolerances": {
            "output_loss_relative_l2": args.output_rel_l2,
            "gradient_delta_relative_l2": args.update_rel_l2,
            "normalized_max": args.normalized_max,
            "per_tensor_l2_floor": "group L2 * 1e-6",
            "per_tensor_peak_floor": "group peak * 1e-4",
            "repeat": "bitwise exact",
        },
        "precision": (
            "FP32 master/reduction; FP32 compute"
            if args.cpu_sanity
            else "FP32 master/reduction and boundary inputs; BF16 autocast; Wan norm2 FP32 override"
        ),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": None if args.cpu_sanity else torch.cuda.get_device_name(),
        },
        "checks": [],
        "steps": [],
        "kernel_calls": [],
        "norm2_calls": [],
    }
    trace = KernelTrace(report)
    try:
        if not args.cpu_sanity:
            trace.install()
        run_matrix(args, report, trace, initial, batches, config, device)
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        rank_path = (
            args.output_json if dist.get_rank() == 0 else args.output_json.with_suffix(f".rank{dist.get_rank()}.json")
        )
        rank_path.parent.mkdir(parents=True, exist_ok=True)
        rank_path.write_text(json.dumps(report, indent=2) + "\n")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
