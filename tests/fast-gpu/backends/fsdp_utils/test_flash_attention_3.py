from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="stage-b-3-gpu-h200",
    labels=["fsdp"],
)

import pytest
import torch
import torch.nn.functional as F

from miles.backends.fsdp_utils import flash_attention_3

pytestmark = pytest.mark.skipif(not flash_attention_3.is_available(), reason="flash_attn_interface not installed")

# Long enough for FA3's default (atomic-add) dQ accumulation to be visibly order-dependent.
SHAPE = (1, 8192, 8, 128)  # [batch, sequence, heads, head dim]


def _inputs(seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tensors = [torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(4)]
    query, key, value, grad_output = tensors
    return query.requires_grad_(True), key.requires_grad_(True), value.requires_grad_(True), grad_output


def _dispatch_flash3(query, key, value):
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    return dispatch_attention_fn(query, key, value, backend="_flash_3")


def _forward_backward(attention, query, key, value, grad_output):
    query.grad = key.grad = value.grad = None
    output = attention(query, key, value)
    output.backward(grad_output)
    return output.detach(), query.grad.clone(), key.grad.clone(), value.grad.clone()


def test_diffusers_flash3_backend_is_differentiable_and_matches_sdpa():
    flash_attention_3.install_diffusers_backend(deterministic=False)
    query, key, value, grad_output = _inputs()

    def sdpa(q, k, v):
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    got = _forward_backward(_dispatch_flash3, query, key, value, grad_output)
    want = _forward_backward(sdpa, query, key, value, grad_output)
    for name, g, w in zip(("out", "dQ", "dK", "dV"), got, want, strict=True):
        torch.testing.assert_close(g, w, rtol=2e-2, atol=2e-2, msg=lambda m, _name=name: f"{_name}: {m}")


def test_diffusers_flash3_backend_deterministic_backward_is_bitwise_repeatable():
    flash_attention_3.install_diffusers_backend(deterministic=True)
    query, key, value, grad_output = _inputs()
    first = _forward_backward(_dispatch_flash3, query, key, value, grad_output)
    second = _forward_backward(_dispatch_flash3, query, key, value, grad_output)
    for name, a, b in zip(("out", "dQ", "dK", "dV"), first, second, strict=True):
        assert torch.equal(a, b), f"{name} differs between two deterministic FA3 runs"


def test_install_is_idempotent_per_flag():
    from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry

    name = AttentionBackendName._FLASH_3
    flash_attention_3.install_diffusers_backend(deterministic=True)
    installed = _AttentionBackendRegistry._backends[name]
    flash_attention_3.install_diffusers_backend(deterministic=True)
    assert _AttentionBackendRegistry._backends[name] is installed
    flash_attention_3.install_diffusers_backend(deterministic=False)
    assert _AttentionBackendRegistry._backends[name] is not installed
    assert _AttentionBackendRegistry._backends[name]._miles_deterministic is False
