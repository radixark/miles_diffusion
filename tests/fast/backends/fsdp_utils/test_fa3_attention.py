"""CPU coverage of the FA3 dispatcher contract, using an autograd-capable spy.

These tests exercise Diffusers' real dispatch and registry. They do not establish
CUDA kernel correctness or determinism; the fast-GPU suite covers those claims.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import pytest
import torch
from diffusers.models import attention_dispatch as dispatch

from miles.backends.fsdp_utils.fa3_attention import install_diffusers_fa3_attention


@pytest.fixture
def kernel_spy(monkeypatch):
    registry = dispatch._AttentionBackendRegistry
    # Installation changes a process-wide registry. Give each test its own copy,
    # including the cached signatures that the actual dispatcher uses to filter.
    for name in ("_backends", "_constraints", "_supported_arg_names"):
        monkeypatch.setattr(registry, name, getattr(registry, name).copy())
    monkeypatch.setattr(registry, "_supports_context_parallel", registry._supports_context_parallel.copy())
    monkeypatch.setattr(registry, "_active_backend", registry._active_backend)
    monkeypatch.setattr(registry, "_checks_enabled", False)
    monkeypatch.setattr(dispatch, "_CAN_USE_FLASH_ATTN_3", True)

    calls = []

    def fake_fa3(
        q,
        k,
        v,
        softmax_scale=None,
        causal=False,
        num_splits=1,
        deterministic=False,
        return_attn_probs=False,
    ):
        calls.append(
            {
                "q": q,
                "k": k,
                "v": v,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "num_splits": num_splits,
                "deterministic": deterministic,
                "return_attn_probs": return_attn_probs,
            }
        )
        # Distinct coefficients make detached or swapped inputs visible in the
        # backward assertion without pretending to emulate a CUDA kernel.
        output = q + 2 * k + 3 * v
        if return_attn_probs:
            lse = q.sum(dim=-1).transpose(1, 2)
            return output, lse, None
        return output

    monkeypatch.setattr(dispatch, "flash_attn_3_func", fake_fa3)
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield calls, fake_fa3
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


def _qkv():
    return tuple(torch.randn(2, 3, 2, 4, requires_grad=True) for _ in range(3))


def _dispatch(qkv, **kwargs):
    return dispatch.dispatch_attention_fn(*qkv, backend="_flash_3", **kwargs)


@pytest.mark.parametrize("force_deterministic", [False, True])
def test_real_dispatch_retains_autograd_and_forwards_kernel_arguments(kernel_spy, force_deterministic):
    calls, _ = kernel_spy
    install_diffusers_fa3_attention(deterministic=force_deterministic)
    q, k, v = _qkv()

    output = _dispatch((q, k, v), scale=0.25, is_causal=True)
    output.sum().backward()

    torch.testing.assert_close(output, q + 2 * k + 3 * v, rtol=0, atol=0)
    for tensor, coefficient in zip((q, k, v), (1, 2, 3), strict=True):
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, coefficient), rtol=0, atol=0)
    assert len(calls) == 1
    assert calls[0]["q"] is q
    assert calls[0]["k"] is k
    assert calls[0]["v"] is v
    assert calls[0]["deterministic"] is force_deterministic
    assert calls[0]["softmax_scale"] == 0.25
    assert calls[0]["causal"] is True
    assert calls[0]["num_splits"] == 1
    assert calls[0]["return_attn_probs"] is False


@pytest.mark.parametrize(
    ("forced", "per_call", "global_flag", "expected"),
    [
        (False, False, False, False),
        (False, True, False, True),
        (False, False, True, True),
        (True, False, False, True),
        (True, True, False, True),
    ],
)
def test_effective_deterministic_mode(kernel_spy, forced, per_call, global_flag, expected):
    calls, _ = kernel_spy
    install_diffusers_fa3_attention(deterministic=forced)
    torch.use_deterministic_algorithms(global_flag)

    _dispatch(_qkv(), attention_kwargs={"deterministic": per_call})

    assert calls[-1]["deterministic"] is expected


def test_lse_uses_diffusers_layout_and_does_not_advertise_a_gradient(kernel_spy):
    calls, _ = kernel_spy
    install_diffusers_fa3_attention(deterministic=True)
    q, k, v = _qkv()

    output, lse = _dispatch((q, k, v), attention_kwargs={"return_lse": True})

    assert lse.shape == q.shape[:-1]
    torch.testing.assert_close(lse, q.sum(dim=-1), rtol=0, atol=0)
    assert not lse.requires_grad
    output.sum().backward()
    assert all(tensor.grad is not None for tensor in (q, k, v))
    assert calls[-1]["return_attn_probs"] is True


def test_install_updates_cached_signature_and_preserves_other_backends(kernel_spy):
    registry = dispatch._AttentionBackendRegistry
    fa3 = dispatch.AttentionBackendName._FLASH_3
    fa2 = dispatch.AttentionBackendName.FLASH
    original_fa2 = registry._backends[fa2]
    original_fa2_args = registry._supported_arg_names[fa2].copy()
    original_constraints = list(registry._constraints[fa3])
    original_active = registry._active_backend

    install_diffusers_fa3_attention(deterministic=True)

    assert {"deterministic", "num_splits", "dropout_p", "_parallel_config"} <= registry._supported_arg_names[fa3]
    assert registry._constraints[fa3] == original_constraints
    assert fa3.value not in registry._supports_context_parallel
    assert registry._backends[fa2] is original_fa2
    assert registry._supported_arg_names[fa2] == original_fa2_args
    assert registry._active_backend == original_active


def test_install_is_idempotent_and_explicit_false_resets_forced_mode(kernel_spy):
    calls, _ = kernel_spy
    registry = dispatch._AttentionBackendRegistry
    fa3 = dispatch.AttentionBackendName._FLASH_3
    install_diffusers_fa3_attention(deterministic=True)
    installed = registry._backends[fa3]
    install_diffusers_fa3_attention(deterministic=True)
    assert registry._backends[fa3] is installed

    install_diffusers_fa3_attention(deterministic=False)
    _dispatch(_qkv())

    assert calls[-1]["deterministic"] is False


@pytest.mark.parametrize("deterministic", [False, True])
def test_model_backend_selection_preserves_its_requested_mode(kernel_spy, deterministic):
    from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend

    class RecordingModel:
        selected = None

        def set_attention_backend(self, backend):
            self.selected = backend

    calls, _ = kernel_spy
    # Start with a previously forced process-wide installation, so a new actor's
    # default mode cannot accidentally inherit it.
    install_diffusers_fa3_attention(deterministic=True)
    model_backend = DiffusersModelBackend(None)
    model = RecordingModel()
    if deterministic:
        # This is the FSDP actor's real initialization order.
        model_backend.enable_deterministic_attention("_flash_3")
    model_backend.set_attention_backend(model, "_flash_3")
    output = _dispatch(_qkv(), attention_kwargs={"deterministic": False})
    output.sum().backward()

    assert model.selected == "_flash_3"
    assert calls[-1]["deterministic"] is deterministic


def test_unspecified_model_backend_honors_active_fa3(kernel_spy):
    from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend

    calls, _ = kernel_spy
    dispatch._AttentionBackendRegistry._active_backend = dispatch.AttentionBackendName._FLASH_3
    model_backend = DiffusersModelBackend(None)

    model_backend.enable_deterministic_attention(None)
    q, k, v = _qkv()
    # Omitting backend is how DIFFUSERS_ATTN_BACKEND selects the active entry.
    output = dispatch.dispatch_attention_fn(q, k, v, attention_kwargs={"deterministic": False})
    output.sum().backward()

    assert calls[-1]["deterministic"] is True
    for tensor, coefficient in zip((q, k, v), (1, 2, 3), strict=True):
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, coefficient), rtol=0, atol=0)


def test_unspecified_model_backend_patches_only_active_fa2(kernel_spy, monkeypatch):
    from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend

    _, fake_fa3 = kernel_spy
    fa2_calls = []

    def fake_fa2(q, k, v, deterministic=False):
        fa2_calls.append(deterministic)
        return q + 2 * k + 3 * v

    monkeypatch.setattr(dispatch, "flash_attn_func", fake_fa2)
    registry = dispatch._AttentionBackendRegistry
    registry._active_backend = dispatch.AttentionBackendName.FLASH
    original_fa3_backend = registry._backends[dispatch.AttentionBackendName._FLASH_3]
    original_varlen_kernel = dispatch.flash_attn_varlen_func

    DiffusersModelBackend(None).enable_deterministic_attention(None)
    dispatch.flash_attn_func(*_qkv())

    assert fa2_calls == [True]
    assert dispatch.flash_attn_3_func is fake_fa3
    assert dispatch.flash_attn_varlen_func is original_varlen_kernel
    assert registry._backends[dispatch.AttentionBackendName._FLASH_3] is original_fa3_backend


@pytest.mark.parametrize("active", [dispatch.AttentionBackendName.NATIVE, dispatch.AttentionBackendName._NATIVE_MATH])
def test_unspecified_native_backend_needs_no_kernel_patch(kernel_spy, active):
    from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend

    calls, _ = kernel_spy
    registry = dispatch._AttentionBackendRegistry
    registry._active_backend = active
    original_backends = registry._backends.copy()

    DiffusersModelBackend(None).enable_deterministic_attention(None)

    assert registry._backends == original_backends
    assert calls == []


def test_unspecified_unsupported_custom_backend_fails_closed(kernel_spy):
    from miles.backends.fsdp_utils.model_backend import DiffusersModelBackend

    calls, _ = kernel_spy
    dispatch._AttentionBackendRegistry._active_backend = dispatch.AttentionBackendName.SAGE

    with pytest.raises((ValueError, RuntimeError), match="(?i)(sage|deterministic|backend)"):
        DiffusersModelBackend(None).enable_deterministic_attention(None)

    assert calls == []


def test_kernel_is_resolved_at_call_time(kernel_spy, monkeypatch):
    calls, fake_fa3 = kernel_spy
    install_diffusers_fa3_attention(deterministic=True)
    late_calls = []

    def late_spy(*args, **kwargs):
        late_calls.append(kwargs)
        return fake_fa3(*args, **kwargs)

    monkeypatch.setattr(dispatch, "flash_attn_3_func", late_spy)
    _dispatch(_qkv())

    assert len(late_calls) == len(calls) == 1
    assert late_calls[0]["deterministic"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attn_mask": torch.ones(3, 3, dtype=torch.bool)},
        {"dropout_p": 0.1},
        {"parallel_config": object()},
        {"attention_kwargs": {"num_splits": 2}},
    ],
    ids=["attention-mask", "dropout", "diffusers-context-parallel", "split-kv"],
)
def test_unsupported_requests_fail_before_calling_kernel(kernel_spy, kwargs):
    calls, _ = kernel_spy
    install_diffusers_fa3_attention(deterministic=True)

    with pytest.raises((ValueError, NotImplementedError)):
        _dispatch(_qkv(), **kwargs)

    assert calls == []


@pytest.mark.parametrize("missing", [None, "no-deterministic-parameter"])
def test_missing_or_incompatible_kernel_fails_at_install(kernel_spy, monkeypatch, missing):
    def incompatible_kernel(q, k, v):
        return q

    monkeypatch.setattr(dispatch, "flash_attn_3_func", None if missing is None else incompatible_kernel)

    with pytest.raises(RuntimeError, match="(?i)(flash|fa3|deterministic)"):
        install_diffusers_fa3_attention(deterministic=True)
