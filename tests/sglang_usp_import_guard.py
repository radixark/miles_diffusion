"""守卫：断言**实际被 import 的** sglang 运行时确含训练所需的补丁。

环境里存在两棵 sglang：打了补丁的 editable 安装（miles-diffusion conda env 解析到的
/workspace/.../sglang，wan-strict-mode）与未打补丁的系统树（/sgl-workspace/sglang）。
训练必须解析到前者，否则 USP 反向静默断梯度、checksum 漏 dtype/shape。本测试在跑 GPU
parity 前先验证 import 路径正确，把"环境漂移到未打补丁树"变成显式失败而非沉默错误。

用法: conda activate miles-diffusion && PYTHONPATH=. python -m pytest tests/sglang_usp_import_guard.py
"""
import inspect


def test_usp_has_training_autograd():
    from sglang.multimodal_gen.runtime.layers import usp

    # 训练反向所需的可微 wrapper（推理路径不依赖，但训练必须有）
    assert hasattr(usp, "_AllToAllSingle"), (
        f"被 import 的 sglang 缺 _AllToAllSingle：{usp.__file__} —— "
        "可能漂移到未打补丁的 /sgl-workspace/sglang，Ulysses 训练反向会断梯度"
    )
    assert hasattr(usp, "_RingFlashAttention"), (
        f"被 import 的 sglang 缺 _RingFlashAttention：{usp.__file__} —— Ring 训练 dK/dV 会错"
    )


def test_checksum_covers_dtype_shape():
    from sglang.multimodal_gen.runtime.loader import weight_utils

    src = inspect.getsource(weight_utils.compute_weights_checksum)
    assert "str(t.dtype)" in src and "tuple(t.shape)" in src, (
        f"被 import 的 sglang checksum 未覆盖 dtype/shape：{weight_utils.__file__} —— "
        "与训练侧 _sha256_named_tensors 不对称，转置/reshape 不会被检出"
    )


if __name__ == "__main__":
    test_usp_has_training_autograd()
    test_checksum_covers_dtype_shape()
    from sglang.multimodal_gen.runtime.layers import usp
    print(f"[GUARD OK] imported sglang = {usp.__file__}")
    print("  _AllToAllSingle / _RingFlashAttention 在；checksum 覆盖 dtype/shape")
