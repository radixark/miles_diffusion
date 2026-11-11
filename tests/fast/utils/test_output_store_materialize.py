"""The trainer reads a shard's stored tensors back into the tensors the inline path would have carried."""

import msgspec
import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from radix_transfer_plugins import OutputStore, StorageBackend
from radix_transfer_plugins.stashes.tensor_stash import TensorStash

from miles.ray.data_conversion_hub.flow_grpo import expand_samples_to_train_pairs
from miles.utils.output_store import StoredTensor, materialize_stored_tensors
from miles.utils.types import CondKwargs, DenoisingEnv, DiTTrajectory, RolloutDebugTensors, Sample

B, T = 2, 3
K = T + 1
SDE = [1]


class MemoryBackend(StorageBackend):
    name = "memory"
    config_type = msgspec.Struct

    def __init__(self):
        self.objects = {}
        self.reads = []

    def put_many(self, objects):
        keys = []
        for parts in objects:
            key = f"mem/{len(self.objects)}"
            self.objects[key] = {name: bytes(data) for name, data in parts.items()}
            keys.append(key)
        return keys

    def get_many(self, keys, *, parts):
        self.reads.append([(key, [name for name, _ in named]) for key, named in zip(keys, parts, strict=True)])
        return [
            {name: memoryview(bytearray(self.objects[key][name])) for name, _ in named}
            for key, named in zip(keys, parts, strict=True)
        ]

    def register(self, memory):
        pass

    def unregister(self, memory):
        pass

    def get_ranges(self, reads, *, into):
        raise NotImplementedError

    def delete_many(self, keys, *, parts):
        for key in keys:
            del self.objects[key]

    def close(self):
        pass


def _batched():
    g = torch.Generator().manual_seed(3)
    return {
        "dit_trajectory.latents": torch.randn(B, K, 4, 6, generator=g).bfloat16(),
        "rollout_debug_tensors.rollout_model_outputs": torch.randn(B, T, 4, 6, generator=g),
        "denoising_env.pos_cond_kwargs.encoder_hidden_states.0": torch.randn(B, 5, 8, generator=g).bfloat16(),
        "denoising_env.pos_cond_kwargs.pooled_projections.0": torch.randn(8, generator=g),
    }


def _fields(batched):
    """The fields sglang stores: one per sample and step for the step-axis tensors, one per sample for cond."""
    fields = {
        "denoising_env.pos_cond_kwargs.pooled_projections.0": batched[
            "denoising_env.pos_cond_kwargs.pooled_projections.0"
        ]
    }
    for b in range(B):
        for name in ("dit_trajectory.latents", "rollout_debug_tensors.rollout_model_outputs"):
            for step, row in enumerate(batched[name][b]):
                fields[f"{name}.{b}.{step}"] = row
        hidden = "denoising_env.pos_cond_kwargs.encoder_hidden_states.0"
        fields[f"{hidden}.{b}"] = batched[hidden][b]
    return fields


def _sample(index, latents, model_outputs, hidden, pooled) -> Sample:
    return Sample(
        index=index,
        prompt="p",
        train_metadata={"sde_step_indices": SDE},
        rollout_log_probs=torch.linspace(-1.0, -2.0, T),
        rollout_debug_tensors=RolloutDebugTensors(rollout_model_outputs=model_outputs),
        denoising_env=DenoisingEnv(
            pos_cond_kwargs=CondKwargs(encoder_hidden_states=[hidden], pooled_projections=[pooled], txt_seq_lens=[5])
        ),
        dit_trajectory=DiTTrajectory(
            latents=latents,
            timesteps=torch.linspace(1000.0, 0.0, T + 1),
            sigmas=torch.linspace(1.0, 0.0, T + 1),
            latent_step_indices=torch.arange(K),
        ),
    )


def test_pairs_read_back_equal_the_inline_pairs():
    backend = MemoryBackend()
    store = OutputStore(backend=backend)
    batched = _batched()
    (ref,) = store.put_many([TensorStash(_fields(batched))])

    def steps_of(name, b, num_steps):
        return [StoredTensor(ref, f"{name}.{b}.{step}") for step in range(num_steps)]

    inline = [
        _sample(
            b,
            batched["dit_trajectory.latents"][b],
            batched["rollout_debug_tensors.rollout_model_outputs"][b],
            batched["denoising_env.pos_cond_kwargs.encoder_hidden_states.0"][b],
            batched["denoising_env.pos_cond_kwargs.pooled_projections.0"],
        )
        for b in range(B)
    ]
    stored = [
        _sample(
            b,
            steps_of("dit_trajectory.latents", b, K),
            steps_of("rollout_debug_tensors.rollout_model_outputs", b, T),
            StoredTensor(ref, f"denoising_env.pos_cond_kwargs.encoder_hidden_states.0.{b}"),
            StoredTensor(ref, "denoising_env.pos_cond_kwargs.pooled_projections.0"),
        )
        for b in range(B)
    ]
    rewards = [0.5, -0.5]
    expected = expand_samples_to_train_pairs(None, inline, rewards, rewards)["train_data"]
    pairs = expand_samples_to_train_pairs(None, stored, rewards, rewards)["train_data"]
    assert isinstance(pairs[0]["latent"], StoredTensor)

    materialize_stored_tensors(pairs, store=store)

    # Only the trained step, its next latent and the cond fields; not the whole trajectory
    read = sorted(
        [f"dit_trajectory.latents.{b}.{s}" for b in range(B) for s in (1, 2)]
        + [f"rollout_debug_tensors.rollout_model_outputs.{b}.1" for b in range(B)]
        + [f"denoising_env.pos_cond_kwargs.encoder_hidden_states.0.{b}" for b in range(B)]
        + ["denoising_env.pos_cond_kwargs.pooled_projections.0"]
    )
    assert len(backend.reads) == 1 and backend.reads[0] == [(ref.key, read)]
    assert len(pairs) == len(expected) == B * len(SDE)
    for pair, want in zip(pairs, expected, strict=True):
        for key in ("latent", "next_latent", "timestep", "sigma", "log_prob_old"):
            assert torch.equal(pair[key], want[key]), key
        assert pair["latent"].dtype == torch.bfloat16
        assert torch.equal(
            pair["rollout_debug_tensors"]["rollout_step_model_output"],
            want["rollout_debug_tensors"]["rollout_step_model_output"],
        )
        cond, want_cond = pair["denoising_env"].pos_cond_kwargs, want["denoising_env"].pos_cond_kwargs
        assert torch.equal(cond.encoder_hidden_states[0], want_cond.encoder_hidden_states[0])
        assert torch.equal(cond.pooled_projections[0], want_cond.pooled_projections[0])
        assert cond.txt_seq_lens == [5]
        assert pair["advantage"] == want["advantage"]


def test_pairs_without_stored_tensors_read_nothing():
    backend = MemoryBackend()
    pairs = [{"latent": torch.ones(2), "denoising_env": DenoisingEnv()}]
    materialize_stored_tensors(pairs, store=OutputStore(backend=backend))
    assert backend.reads == []
    assert torch.equal(pairs[0]["latent"], torch.ones(2))
