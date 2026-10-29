"""CPU OpenPI data producer for train_sft.py, using miles' resumable JSONL cursor."""

from functools import lru_cache

from miles.backends.fsdp_utils.models.pi05.data import load_dataset, to_train_pair
from miles.rollout.base_types import RolloutFnTrainOutput
from miles.utils.types import Sample


@lru_cache(maxsize=1)
def _dataset(bundle):
    return load_dataset(bundle)


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        raise ValueError("Use scripts/eval_pi05_libero.py for simulation evaluation")
    dataset = _dataset(args.hf_checkpoint)
    groups = data_source.get_samples(args.rollout_batch_size)
    for group in groups:
        for sample in group:
            index = sample.metadata["openpi_index"]
            if type(index) is not int or not 0 <= index < len(dataset):
                raise ValueError(f"Invalid OpenPI dataset index: {index}")
            sample.train_metadata = {"sft_pair": to_train_pair(dataset[index])}
            sample.status = Sample.Status.COMPLETED
    return RolloutFnTrainOutput(samples=groups, metrics={"sft_epoch": data_source.epoch_id})
