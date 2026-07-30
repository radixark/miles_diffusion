import torch


def cast_cond_to_dtype(cond: dict, dtype: torch.dtype) -> dict:
    return {
        key: value.to(dtype=dtype) if isinstance(value, torch.Tensor) and value.dtype.is_floating_point else value
        for key, value in cond.items()
    }
