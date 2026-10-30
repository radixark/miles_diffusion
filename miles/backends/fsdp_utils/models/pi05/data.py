"""Use OpenPI's LeRobot chunking, normalization, padding and tokenization unchanged."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from .bundle import read_bundle


class _ColumnListDataset:
    """Present datasets 4 lazy columns as the tensor lists expected by LeRobot 2.x."""

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_dataset"), name)

    def __getitem__(self, key):
        value = self._dataset[key]
        return value[:] if isinstance(key, str) else value

    def select(self, indices, **kwargs):
        return type(self)(self._dataset.select(indices, **kwargs))


class _LeRobotDataset(LeRobotDataset):
    def load_hf_dataset(self):
        return _ColumnListDataset(super().load_hf_dataset())


def load_data_config(bundle):
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import normalize
    from openpi.training import config

    model_config = Pi0Config(**read_bundle(bundle)["model"])
    train_config = config.get_config("pi05_libero")
    data_config = train_config.data.create(Path(bundle) / "assets", model_config)
    # Bind both train and eval to the stats shipped alongside the weights, never
    # to a machine-local openpi assets directory that might contain different stats.
    stats = normalize.load(Path(bundle) / "assets" / "physical-intelligence" / "libero")
    return model_config, dataclasses.replace(data_config, norm_stats=stats)


def create_raw_dataset(model, data, episodes=None, revision=None):
    """Retain official LeRobot chunking while allowing a small episode subset."""
    if episodes is not None and (
        not episodes or any(index < 0 for index in episodes) or len(set(episodes)) != len(episodes)
    ):
        raise ValueError("Dataset episodes must be a nonempty list of unique non-negative indices")
    # OpenPI's factory does not expose LeRobot's episode filter. Use the same
    # dataset and prompt wrapper, adapting datasets 4 column access locally.
    from openpi import transforms
    from openpi.training import data_loader

    metadata = data_loader.lerobot_dataset.LeRobotDatasetMetadata(data.repo_id, revision=revision)
    dataset = _LeRobotDataset(
        data.repo_id,
        episodes=episodes,
        revision=revision,
        delta_timestamps={
            key: [t / metadata.fps for t in range(model.action_horizon)] for key in data.action_sequence_keys
        },
    )
    if data.prompt_from_task:
        dataset = data_loader.TransformedDataset(dataset, [transforms.PromptFromLeRobotTask(metadata.tasks)])
    return dataset


def load_dataset(bundle):
    from openpi.training import data_loader

    model, data = load_data_config(bundle)
    metadata = read_bundle(bundle)
    raw = create_raw_dataset(
        model, data, metadata.get("dataset_episodes"), (metadata.get("dataset_source") or {}).get("revision")
    )
    return data_loader.transform_dataset(raw, data)


def to_train_pair(item):
    import numpy as np
    import torch

    def convert(value):
        if isinstance(value, dict):
            return {key: convert(child) for key, child in value.items()}
        return torch.from_numpy(np.array(value, copy=True))

    required = ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask")
    return {
        "observation": {key: convert(item[key]) for key in required},
        "actions": convert(item["actions"]).float(),
    }
