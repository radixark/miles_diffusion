import json
import logging
import os
import random

from miles.utils.types import Sample

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


def read_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if not path.endswith(".jsonl"):
        raise ValueError(f"Unsupported file format: {path}. Supported format is .jsonl.")

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decode error at line {line_num}: {e}")
                continue


class Dataset:
    """T2I RL: same loading pattern as :class:`miles.utils.data.Dataset` — ``read_file`` yields row dicts; we build :class:`~miles.utils.types.Sample`."""

    def __init__(
        self,
        path,
        *,
        prompt_key="text",
        metadata_key="metadata",
        seed=42,
    ):
        origin_samples = []
        for data in read_file(path):
            prompt = data.get(prompt_key)
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            metadata = data.get(metadata_key) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            origin_samples.append(Sample(prompt=prompt.strip(), metadata=metadata))

        self.origin_samples = origin_samples
        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id: int) -> None:
        if self.epoch_id == new_epoch_id:
            return
        random.seed(self.seed + new_epoch_id)
        order = list(range(len(self.samples)))
        random.shuffle(order)
        self.samples = [self.origin_samples[i] for i in order]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def __len__(self) -> int:
        return len(self.samples)
