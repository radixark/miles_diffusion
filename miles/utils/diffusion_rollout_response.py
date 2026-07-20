"""Parse sglang-diffusion ``POST /rollout/generate`` JSON into :class:`~miles.utils.types.Sample`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import msgpack
import ray
import torch

from miles.utils.processing_utils import fhwc_to_cfhw
from miles.utils.types import CondKwargs, DenoisingEnv, DiTTrajectory, RolloutDebugTensors, Sample, decode_tensor

__all__ = [
    "apply_rollout_image_response",
    "RolloutImageResponseParserActor",
]

# Prefer these keys for mapping dict ``rollout_log_probs`` → ``Sample.rollout_log_probs``.
_ROLLOUT_LOG_PROB_PRIMARY_KEYS = ("log_prob", "log_probs", "total", "per_step")
_IMAGE_CHANNEL_COUNTS = (1, 3, 4)


def _default_deserialize_func(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, dict) and value.get("__tensor__"):
        return decode_tensor(value["data"]).detach().cpu()
    raise TypeError(f"Cannot deserialize {type(value)}")


def _normalize_generated_output(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Normalize a per-sample image/video to ``[C, F, H, W]``."""
    if tensor is None:
        return None
    if tensor.ndim not in (3, 4):
        raise ValueError("generated_output must be CHW/HWC or CFHW/FHWC, " f"got shape {tuple(tensor.shape)}")

    first_is_channel = tensor.shape[0] in _IMAGE_CHANNEL_COUNTS
    last_is_channel = tensor.shape[-1] in _IMAGE_CHANNEL_COUNTS
    if first_is_channel and last_is_channel:
        raise ValueError(
            "generated_output layout is ambiguous because both the first and last "
            f"dimensions look like channels: {tuple(tensor.shape)}"
        )

    # TODO: Move this canonicalization into sglang-diffusion's rollout response
    # builder once it exposes an explicit generated-output layout contract.
    if first_is_channel:
        canonical = tensor.unsqueeze(1) if tensor.ndim == 3 else tensor
    elif last_is_channel:
        fhwc = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
        canonical = fhwc_to_cfhw(fhwc)
    else:
        raise ValueError(
            "generated_output has no recognizable channel dimension; expected "
            f"1, 3, or 4 channels, got shape {tuple(tensor.shape)}"
        )

    return canonical.contiguous()


def _deserialize_rollout_log_probs(
    value: Any,
    *,
    deserialize_func: Callable[[Any], torch.Tensor | None],
) -> torch.Tensor | None:
    # Eval-mode rollout (rollout=False) sends no log_probs; train-mode always does.
    if value is None:
        return None
    return deserialize_func(value)


def _parse_tensor_or_list(
    value: Any,
    *,
    deserialize_func: Callable[[Any], torch.Tensor | None],
) -> list[torch.Tensor] | None:
    """Deserialize a field that may be a single serialized tensor or a list of them.

    sglang-diffusion models differ in what they put in cond_kwargs:
    - Qwen-Image etc.: ``encoder_hidden_states`` is a **list** of serialized tensors.
    - SD3: ``encoder_hidden_states`` / ``pooled_projections`` is a **single** serialized
      tensor (dict with ``__tensor__`` key), not a list.
    Both cases are normalised to ``list[Tensor]`` to match ``CondKwargs`` field types.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        # Single serialized tensor (SD3 and similar)
        return [deserialize_func(value)]
    # List of serialized tensors (Qwen-Image etc.)
    return [deserialize_func(x) for x in value]


def _parse_cond_kwargs(
    data: dict[str, Any] | None,
    *,
    deserialize_func: Callable[[Any], torch.Tensor | None],
) -> CondKwargs | None:
    if not data:
        return None
    return CondKwargs(
        txt_seq_lens=data.get("txt_seq_lens"),
        freqs_cis=[deserialize_func(x) for x in data.get("freqs_cis", [])],
        img_shapes=data.get("img_shapes"),
        encoder_hidden_states=_parse_tensor_or_list(
            data.get("encoder_hidden_states") or data.get("context"),
            deserialize_func=deserialize_func,
        ),
        audio_encoder_hidden_states=_parse_tensor_or_list(
            data.get("audio_encoder_hidden_states"),
            deserialize_func=deserialize_func,
        ),
        encoder_attention_mask=deserialize_func(data.get("encoder_attention_mask")),
        audio_encoder_attention_mask=deserialize_func(data.get("audio_encoder_attention_mask")),
        pooled_projections=_parse_tensor_or_list(data.get("pooled_projections"), deserialize_func=deserialize_func),
    )


def _parse_denoising_env(
    data: dict[str, Any] | None,
    *,
    deserialize_func: Callable[[Any], torch.Tensor | None],
) -> DenoisingEnv | None:
    if not data:
        return None
    return DenoisingEnv(
        image_kwargs=data.get("image_kwargs"),
        pos_cond_kwargs=_parse_cond_kwargs(data.get("pos_cond_kwargs"), deserialize_func=deserialize_func),
        neg_cond_kwargs=_parse_cond_kwargs(data.get("neg_cond_kwargs"), deserialize_func=deserialize_func),
        guidance=data.get("guidance"),
    )


def _parse_dit_trajectory(
    data: dict[str, Any] | None,
    *,
    deserialize_func: Callable[[Any], torch.Tensor | None],
) -> DiTTrajectory | None:
    if not data:
        return None
    return DiTTrajectory(
        latents=deserialize_func(data.get("latents")),
        timesteps=deserialize_func(data.get("timesteps")),
        sigmas=deserialize_func(data.get("sigmas")),
    )


def _parse_rollout_debug_tensors(
    data: dict[str, Any] | None,
    *,
    deserialize_func: Callable[[Any], torch.Tensor | None],
) -> RolloutDebugTensors | None:
    if not data:
        return None
    return RolloutDebugTensors(
        rollout_variance_noises=deserialize_func(data.get("rollout_variance_noises")),
        rollout_prev_sample_means=deserialize_func(data.get("rollout_prev_sample_means")),
        rollout_noise_std_devs=deserialize_func(data.get("rollout_noise_std_devs")),
        rollout_model_outputs=deserialize_func(data.get("rollout_model_outputs")),
    )


def apply_rollout_image_response(
    sample: Sample,
    body: dict[str, Any],
    *,
    deserialize_func: Callable[[Any], torch.Tensor | None] = _default_deserialize_func,
) -> Sample:
    """Fill ``sample`` fields from one ``RolloutImageResponse``-shaped dict (per-sample tensors, no batch dim)."""
    sample.request_id = body.get("request_id") or sample.request_id
    if "prompt" in body:
        sample.prompt = str(body["prompt"])
    if "seed" in body:
        sample.seed = int(body["seed"])

    sample.generated_output = _normalize_generated_output(deserialize_func(body.get("generated_output")))
    sample.rollout_log_probs = _deserialize_rollout_log_probs(
        body.get("rollout_log_probs"), deserialize_func=deserialize_func
    )
    sample.rollout_debug_tensors = _parse_rollout_debug_tensors(
        body.get("rollout_debug_tensors"),
        deserialize_func=deserialize_func,
    )
    sample.denoising_env = _parse_denoising_env(body.get("denoising_env"), deserialize_func=deserialize_func)
    sample.dit_trajectory = _parse_dit_trajectory(body.get("dit_trajectory"), deserialize_func=deserialize_func)

    if "inference_time_s" in body and body["inference_time_s"] is not None:
        sample.inference_time_s = float(body["inference_time_s"])
    if "peak_memory_mb" in body and body["peak_memory_mb"] is not None:
        sample.peak_memory_mb = float(body["peak_memory_mb"])
    return sample


@ray.remote(num_cpus=1)
class RolloutImageResponseParserActor:
    def apply_raw(self, samples: list[Sample], raw: bytes) -> list[Sample]:
        bodies = msgpack.unpackb(raw, raw=False)
        return [apply_rollout_image_response(s, b) for s, b in zip(samples, bodies, strict=True)]
