from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.backends.fsdp_utils import arguments as fsdp_args


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

    @pytest.mark.parametrize("backend", ["flash_attention_3", "flash_attention_4"])
    def test_native_model_spellings_defer_to_package_hook(self, monkeypatch, backend):
        # Native packages own these kernels; Diffusers availability is unrelated.
        monkeypatch.setattr(fsdp_args, "deterministic_capable_flash_fns", lambda: [])
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

    def test_fa2_does_not_validate_fa3(self, monkeypatch):
        monkeypatch.setattr(fsdp_args, "deterministic_capable_flash_fns", lambda: ["flash_attn_func"])
        with pytest.raises(RuntimeError, match="flash_attn_3_func"):
            fsdp_args.validate_attention_args(_args(True, "_flash_3"))

    @pytest.mark.parametrize(
        "backend,entrypoint",
        [("flash", "flash_attn_func"), ("flash_varlen", "flash_attn_varlen_func"), ("_flash_3", "flash_attn_3_func")],
    )
    def test_selected_flash_ok_when_capable(self, monkeypatch, backend, entrypoint):
        monkeypatch.setattr(fsdp_args, "deterministic_capable_flash_fns", lambda: [entrypoint])
        fsdp_args.validate_attention_args(_args(True, backend))

    @pytest.mark.parametrize("backend", ["_flash_3_hub", "_flash_3_varlen_hub", "_flash_varlen_3", "flash_hub"])
    def test_unpatched_flash_backends_rejected(self, monkeypatch, backend):
        monkeypatch.setattr(
            fsdp_args, "deterministic_capable_flash_fns", lambda: list(fsdp_args._FLASH_ATTN_DISPATCH_FNS)
        )
        with pytest.raises(ValueError):
            fsdp_args.validate_attention_args(_args(True, backend))


def test_fsdp_args_expose_new_flags():
    import dataclasses

    from miles.backends.fsdp_utils.arguments import FSDPArgs

    names = {f.name for f in dataclasses.fields(FSDPArgs)}
    assert "fsdp_attention_backend" in names
    assert "deterministic_mode" in names
