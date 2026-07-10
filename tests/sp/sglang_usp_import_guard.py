"""Guard: the sglang tree actually imported must have the dtype/shape-aware checksum.

Training's only remaining sglang import is compute_weights_checksum (weight-sync
verify); a bytes-only checksum would miss transpose/reshape corruption. The USP
attention operators are miles-owned (sp_ops.py) and need no sglang patches.

Usage: PYTHONPATH=. python -m pytest tests/sp/sglang_usp_import_guard.py
"""

import inspect


def test_checksum_covers_dtype_shape():
    from sglang.multimodal_gen.runtime.loader import weight_utils

    src = inspect.getsource(weight_utils.compute_weights_checksum)
    assert "dtype" in src and "shape" in src, (
        f"imported sglang checksum is bytes-only: {weight_utils.__file__} — " "transpose/reshape would not be detected"
    )


if __name__ == "__main__":
    test_checksum_covers_dtype_shape()
    from sglang.multimodal_gen.runtime.loader import weight_utils

    print(f"[GUARD OK] imported sglang = {weight_utils.__file__}")
