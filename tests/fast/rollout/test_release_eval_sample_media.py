from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import torch

from miles.rollout.sglang_diffusion_rollout import release_eval_sample_media
from miles.utils.types import Sample


def make_sample(index: int) -> Sample:
    return Sample(index=index, generated_output=torch.zeros(3, 1, 4, 4), generated_audio=torch.zeros(2, 8))


def make_args(save_debug_rollout_data=None, wandb_log_num_images=2, save_eval_media_dir=None) -> Namespace:
    return Namespace(
        save_debug_rollout_data=save_debug_rollout_data,
        wandb_log_num_images=wandb_log_num_images,
        save_eval_media_dir=save_eval_media_dir,
    )


def test_media_released_when_not_logged():
    sample = make_sample(index=3)

    release_eval_sample_media(make_args(), sample)

    assert sample.generated_output is None
    assert sample.generated_audio is None


def test_media_kept_for_logged_samples():
    sample = make_sample(index=1)

    release_eval_sample_media(make_args(), sample)

    assert sample.generated_output is not None


def test_media_kept_when_debug_dump_enabled():
    sample = make_sample(index=3)

    release_eval_sample_media(make_args(save_debug_rollout_data="/tmp/dump", wandb_log_num_images=0), sample)

    assert sample.generated_output is not None


def test_media_kept_when_eval_media_is_saved():
    sample = make_sample(index=3)

    release_eval_sample_media(make_args(wandb_log_num_images=0, save_eval_media_dir="/tmp/media"), sample)

    assert sample.generated_output is not None
    assert sample.generated_audio is not None
