"""Guard: the sglang tree actually imported must contain the training patches.

Multiple sglang checkouts can coexist on a machine; training must resolve to the
one with the USP autograd wrappers and the dtype/shape-aware checksum, otherwise
Ulysses backward silently drops gradients. Run before any GPU parity script.

Usage: PYTHONPATH=. python -m pytest tests/sp/sglang_usp_import_guard.py
"""

import inspect


def test_usp_has_training_autograd():
    from sglang.multimodal_gen.runtime.layers import usp

    assert hasattr(usp, "_AllToAllSingle"), (
        f"imported sglang lacks _AllToAllSingle: {usp.__file__} — "
        "Ulysses training backward would silently drop q/k/v grads"
    )
    assert hasattr(usp, "_RingFlashAttention"), (
        f"imported sglang lacks _RingFlashAttention: {usp.__file__} — ring training dK/dV would be wrong"
    )


def test_checksum_covers_dtype_shape():
    from sglang.multimodal_gen.runtime.loader import weight_utils

    src = inspect.getsource(weight_utils.compute_weights_checksum)
    assert "dtype" in src and "shape" in src, (
        f"imported sglang checksum is bytes-only: {weight_utils.__file__} — "
        "transpose/reshape would not be detected"
    )


if __name__ == "__main__":
    test_usp_has_training_autograd()
    test_checksum_covers_dtype_shape()
    from sglang.multimodal_gen.runtime.layers import usp

    print(f"[GUARD OK] imported sglang = {usp.__file__}")
