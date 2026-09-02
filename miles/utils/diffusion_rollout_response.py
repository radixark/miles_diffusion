"""Parse sglang-diffusion ``POST /rollout/generate`` JSON into :class:`~miles.utils.types.Sample`."""

from __future__ import annotations

import time
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
        encoder_attention_mask=_parse_tensor_or_list(
            data.get("encoder_attention_mask"), deserialize_func=deserialize_func
        ),
        audio_encoder_attention_mask=_parse_tensor_or_list(
            data.get("audio_encoder_attention_mask"), deserialize_func=deserialize_func
        ),
        pooled_projections=_parse_tensor_or_list(data.get("pooled_projections"), deserialize_func=deserialize_func),
        h3_packed_layout=_parse_h3_packed_layout(data.get("h3_packed_layout"), deserialize_func=deserialize_func),
        h3_token_tags=_deserialize_optional_tensor(data.get("h3_token_tags"), deserialize_func=deserialize_func),
        text_ids=deserialize_func(data.get("text_ids")),
        text_mask=deserialize_func(data.get("text_mask")),
        fps=data.get("fps"),
    )


def _deserialize_optional_tensor(value, *, deserialize_func):
    if value is None:
        return None
    if isinstance(value, dict) and value.get("__tensor__"):
        return deserialize_func(value)
    if isinstance(value, torch.Tensor):
        return value
    return value


def _parse_h3_packed_layout(value, *, deserialize_func):
    if value is None:
        return None
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        if isinstance(item, dict) and item.get("__tensor__"):
            out[key] = deserialize_func(item)
        else:
            out[key] = item
    return out


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
        latent_step_indices=deserialize_func(data.get("latent_step_indices")),
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
    # Eval-mode rollout (rollout=False) sends no log_probs; train-mode always does.
    sample.rollout_log_probs = deserialize_func(body.get("rollout_log_probs"))
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
    def __init__(self) -> None:
        import httpx

        # Keep-alive off: a pooled connection reused as the server's 5s idle timeout fires dies mid-read
        self._client = httpx.Client(limits=httpx.Limits(max_keepalive_connections=0), timeout=httpx.Timeout(None))

    def apply_raw(self, samples: list[Sample], raw: bytes) -> list[Sample]:
        bodies = msgpack.unpackb(raw, raw=False)
        return [apply_rollout_image_response(s, b) for s, b in zip(samples, bodies, strict=True)]

    def fetch_and_apply(
        self, samples: list[Sample], router_url: str, payload: dict, max_retries: int = 60
    ) -> list[Sample]:
        """Fetch the rollout response straight from an engine and parse it in this process.

        Args:
            samples: samples to fill from the response, positionally paired with its bodies.
            router_url: router base URL, used to pick a least-loaded engine and to ack completion.
            payload: JSON body of the generate request.
            max_retries: attempts before giving up; each retry re-picks, so failures re-balance.

        Returns:
            The filled samples.
        """
        for attempt in range(max_retries):
            try:
                picked = self._client.get(f"{router_url}/pick_worker_for_request")
                picked.raise_for_status()
                worker = picked.json()["url"]
                try:
                    response = self._client.post(f"{worker}/rollout/generate", json=payload)
                    response.raise_for_status()
                finally:
                    try:
                        self._client.post(f"{router_url}/worker_finish_request", params={"url": worker})
                    except Exception:
                        pass  # a lost ack costs one count; failing the request costs the sample
                break
            except Exception:
                if attempt + 1 >= max_retries:
                    raise
                time.sleep(1)
        bodies = msgpack.unpackb(response.content, raw=False)
        return [apply_rollout_image_response(s, b) for s, b in zip(samples, bodies, strict=True)]
