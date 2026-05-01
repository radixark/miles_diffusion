from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import torch
import base64
from safetensors.torch import load, save

def decode_tensor_base64(b64: str) -> torch.Tensor:
    """Deserialize base64 to CPU tensor (same wire format as inference: safetensors ``[\"t\"]``, else ``torch.load``)."""
    raw = base64.b64decode(b64.encode("ascii") if isinstance(b64, str) else b64)
    return load(raw)["t"]

def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Encode a CPU tensor as base64 safetensors (single key ``tensor_key``, default ``t``)."""
    tensor = tensor.detach().cpu()
    raw = save({"t": tensor})
    return base64.b64encode(raw).decode("ascii")


@dataclass
class RolloutDebugTensors:
    rollout_variance_noises: torch.Tensor | None = None
    rollout_prev_sample_means: torch.Tensor | None = None
    rollout_noise_std_devs: torch.Tensor | None = None
    rollout_model_outputs: torch.Tensor | None = None


@dataclass
class CondKwargs:
    txt_seq_lens: list[int] | None = None
    freqs_cis: list[torch.Tensor] | None = None
    img_shapes: list[list[tuple[int, int, int]]] | None = None
    encoder_hidden_states: list[torch.Tensor] | None = None
    pooled_projections: list[torch.Tensor] | None = None


@dataclass
class DenoisingEnv:
    image_kwargs: Any | None = None
    pos_cond_kwargs: CondKwargs | None = None
    neg_cond_kwargs: CondKwargs | None = None
    guidance: Any | None = None


@dataclass
class DiTTrajectory:
    latents: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    # Rollout's scheduler.sigmas snapshot [T+1] (post-shift, includes
    # terminal 0). Use this on the training side instead of recomputing
    # sigmas from `timesteps / num_train_timesteps` — that round-trips
    # σ * 1000 / 1000 in fp32 and drifts 1-2 ULPs, amplifying to ~3e-5
    # log_prob diff.
    sigmas: torch.Tensor | None = None


@dataclass
class Sample:
    """The sample generated.

    Diffusion image rollout: fill from sglang-diffusion ``POST /rollout/generate`` via
    `apply_rollout_image_response`
    """

    group_index: int | None = None
    index: int | None = None
    # correlation id from rollout engine (e.g. UUID string)
    request_id: str | None = None
    # prompt
    prompt: str = ""
    # reproducibility
    seed: int | None = None
    # Eager tensor on CPU. Image rollout shape: ``[C, T, H, W]`` (``T==1`` typical).
    generated_output: torch.Tensor | None = None
    rollout_log_probs: torch.Tensor | None = None
    rollout_debug_tensors: RolloutDebugTensors | None = None
    denoising_env: DenoisingEnv | None = None
    dit_trajectory: DiTTrajectory | None = None

    inference_time_s: float | None = None
    peak_memory_mb: float | None = None

    reward: dict[str, Any] | None = None
    weight_versions: list[str] = field(default_factory=list)

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        ABORTED = "aborted"
        # Indicates a recoverable or non-critical failure during generation (e.g., tool call failure,
        # external API error, parsing error). Unlike ABORTED, FAILED samples may still contain partial
        # valid output and can be retried or handled gracefully.
        FAILED = "failed"

    status: Status = Status.PENDING

    metadata: dict = field(default_factory=dict)
    # metadata used during training, e.g., what loss to use for this sample.
    train_metadata: dict | None = None

    non_generation_time: float = 0.0  # time spent in non-generation steps

    def to_dict(self):
        value = self.__dict__.copy()
        value["status"] = self.status.value
        return value

    @staticmethod
    def from_dict(data: dict):
        data = dict(data)
        data["status"] = Sample.Status(data["status"])
        field_names = set(Sample.__dataclass_fields__.keys())
        init_data = {k: v for k, v in data.items() if k in field_names}
        sample = Sample(**init_data)

        for key, value in data.items():
            if key not in field_names:
                setattr(sample, key, value)

        return sample

    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]
