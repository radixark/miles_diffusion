"""Validate source-independent, unbatched π₀.₅ train pairs before device transfer.

Adapters own tokenization, state encoding, action semantics, normalization and
padding. See docs/models/pi05/pi05.md for the contract; no dataset package is needed.
"""

import torch

from .preprocessing import IMAGE_KEYS


def validate_train_pair(pair, *, action_shape, max_token_len, vocab_size):
    if not isinstance(pair, dict) or not {"observation", "actions"} <= pair.keys():
        raise ValueError("π₀.₅ train pairs require observation and actions")
    obs = pair["observation"]
    required = {"image", "image_mask", "tokenized_prompt", "tokenized_prompt_mask"}
    if not isinstance(obs, dict) or not required <= obs.keys() or obs.keys() - required - {"state"}:
        raise ValueError(f"π₀.₅ observation requires {sorted(required)}; state is optional")

    def tensor(value, name, *, shape=None, dtypes):
        value = torch.as_tensor(value)
        if value.dtype not in dtypes or (shape is not None and tuple(value.shape) != tuple(shape)):
            raise ValueError(f"{name}: expected shape {shape}, dtype {dtypes}; got {value.shape}, {value.dtype}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite values")
        return value

    tensor(pair["actions"], "actions", shape=action_shape, dtypes=(torch.float32,))
    if "state" in obs:
        tensor(obs["state"], "state", shape=(action_shape[1],), dtypes=(torch.float32, torch.float64))
    for name in ("image", "image_mask"):
        if not isinstance(obs[name], dict) or set(obs[name]) != set(IMAGE_KEYS):
            raise ValueError(f"{name} must contain exactly the camera slots {IMAGE_KEYS}")
    for name in IMAGE_KEYS:
        image = tensor(obs["image"][name], f"image.{name}", dtypes=(torch.uint8, torch.float32))
        if tuple(image.shape) not in ((224, 224, 3), (3, 224, 224)):
            raise ValueError(f"image.{name} must be HWC [224,224,3] or CHW [3,224,224]")
        if image.dtype == torch.float32 and ((image < -1).any() or (image > 1).any()):
            raise ValueError(f"image.{name}: floating RGB must be normalized to [-1,1]")
        tensor(obs["image_mask"][name], f"image_mask.{name}", shape=(), dtypes=(torch.bool,))
    tokens = tensor(obs["tokenized_prompt"], "tokenized_prompt", dtypes=(torch.int32, torch.int64))
    if tokens.ndim != 1 or not 0 < tokens.numel() <= max_token_len:
        raise ValueError(f"tokenized_prompt must have shape [T], 1 <= T <= {max_token_len}")
    if ((tokens < 0) | (tokens >= vocab_size)).any():
        raise ValueError(f"tokenized_prompt IDs must lie in [0,{vocab_size})")
    tensor(obs["tokenized_prompt_mask"], "tokenized_prompt_mask", shape=tokens.shape, dtypes=(torch.bool,))
