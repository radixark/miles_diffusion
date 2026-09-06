from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.backends.fsdp_utils import arguments as fsdp_args
from miles.backends.fsdp_utils import flash_attention_3


def _args(deterministic_mode, fsdp_attention_backend):
    return Namespace(deterministic_mode=deterministic_mode, fsdp_attention_backend=fsdp_attention_backend)


class TestValidateAttentionArgs:
    def test_disabled_is_noop(self):
        # deterministic_mode off -> no validation, any backend is accepted
        fsdp_args.validate_attention_args(_args(False, "sage"))

    @pytest.mark.parametrize("backend", [None, "native", "_native_efficient", "NATIVE"])
    def test_native_backends_ok(self, backend):
        # torch's global flag covers SDPA/native
        fsdp_args.validate_attention_args(_args(True, backend))

    @pytest.mark.parametrize("backend", ["sage", "xformers", "flex", "aiter"])
    def test_custom_kernels_rejected(self, backend):
        # opaque to torch's flag, no hook -> refuse rather than run nondeterministic
        with pytest.raises(ValueError):
            fsdp_args.validate_attention_args(_args(True, backend))

    def test_flash_rejected_when_no_capable_fn(self, monkeypatch):
        monkeypatch.setattr(fsdp_args, "deterministic_capable_flash_fns", lambda: [])
        with pytest.raises(RuntimeError):
            fsdp_args.validate_attention_args(_args(True, "flash"))

    def test_flash_ok_when_capable(self, monkeypatch):
        monkeypatch.setattr(fsdp_args, "deterministic_capable_flash_fns", lambda: ["flash_attn_func"])
        fsdp_args.validate_attention_args(_args(True, "flash"))  # no raise

    def test_flash3_rejected_when_not_installed(self, monkeypatch):
        # FA3 bypasses diffusers' entry points; only the trainer's own binding matters
        monkeypatch.setattr(flash_attention_3, "is_available", lambda: False)
        monkeypatch.setattr(fsdp_args, "deterministic_capable_flash_fns", lambda: ["flash_attn_func"])
        with pytest.raises(RuntimeError):
            fsdp_args.validate_attention_args(_args(True, "_flash_3"))

    def test_flash3_ok_when_installed(self, monkeypatch):
        monkeypatch.setattr(flash_attention_3, "is_available", lambda: True)
        monkeypatch.setattr(fsdp_args, "deterministic_capable_flash_fns", lambda: [])
        fsdp_args.validate_attention_args(_args(True, "_flash_3"))  # no raise


def test_is_flash3_backend_accepts_cli_string_and_diffusers_enum():
    from diffusers.models.attention_dispatch import AttentionBackendName

    assert fsdp_args.is_flash3_backend("_flash_3")
    assert fsdp_args.is_flash3_backend(AttentionBackendName._FLASH_3)
    assert not fsdp_args.is_flash3_backend(None)
    assert not fsdp_args.is_flash3_backend("_flash_varlen_3")
    assert not fsdp_args.is_flash3_backend(AttentionBackendName.NATIVE)


def _sp_args(fsdp_attention_backend, ulysses_degree, sequence_parallel_size=4):
    return Namespace(
        actor_num_gpus_per_node=4,
        actor_num_nodes=1,
        sequence_parallel_size=sequence_parallel_size,
        ulysses_degree=ulysses_degree,
        fsdp_attention_backend=fsdp_attention_backend,
    )


class TestValidateSpArgs:
    @pytest.mark.parametrize("backend", [None, "_native_flash", "_native_cudnn", "_flash_3"])
    def test_ring_capable_backends_ok(self, backend):
        fsdp_args.validate_sp_args(_sp_args(backend, ulysses_degree=2))

    @pytest.mark.parametrize("backend", ["flash", "sage", "_flash_varlen_3", "native"])
    def test_ring_rejects_backends_without_a_ring_kernel(self, backend):
        with pytest.raises(ValueError, match="cannot drive ring attention"):
            fsdp_args.validate_sp_args(_sp_args(backend, ulysses_degree=2))

    def test_ulysses_only_accepts_any_backend(self):
        fsdp_args.validate_sp_args(_sp_args("sage", ulysses_degree=4))


def test_fsdp_args_expose_new_flags():
    import dataclasses

    from miles.backends.fsdp_utils.arguments import FSDPArgs

    names = {f.name for f in dataclasses.fields(FSDPArgs)}
    assert "fsdp_attention_backend" in names
    assert "deterministic_mode" in names
