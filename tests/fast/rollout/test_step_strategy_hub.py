"""Step strategies own both halves of the rollout request: the SDE step subset
and the return latents its trainer consumes."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

from miles.rollout.step_strategy_hub import (
    epoch_global_random_choice,
    ode_and_return_last,
    sde_window,
    stop_at_random_step,
)
from miles.utils.types import Sample


def test_sde_window_returns_window_plus_boundary():
    args = Namespace(diffusion_num_sde_steps=3, diffusion_sde_window_range="0,10")
    sde, ret, stop = sde_window(args, Sample(), num_steps=10, seed=7)

    assert sde == list(range(sde[0], sde[0] + 3))
    assert ret == sorted(set(sde) | {i + 1 for i in sde})
    assert stop is None


def test_epoch_global_random_choice_returns_subset_plus_boundary():
    args = Namespace(
        diffusion_sde_candidate_steps="1,3,5,7",
        diffusion_num_sde_steps=2,
        rollout_batch_size=4,
        rollout_seed=0,
    )
    sde, ret, stop = epoch_global_random_choice(args, Sample(group_index=0), num_steps=10, seed=7)

    assert len(sde) == 2 and set(sde) <= {1, 3, 5, 7}
    assert ret == sorted(set(sde) | {i + 1 for i in sde})
    assert stop is None


def test_ode_and_return_last_requests_only_x0():
    assert ode_and_return_last(Namespace(), Sample(), num_steps=10, seed=7) == (None, [10], None)


def test_stop_at_random_step_ships_x_i_and_its_x0_prediction():
    stops = set()
    for seed in range(32):
        sde, ret, stop = stop_at_random_step(Namespace(), Sample(), num_steps=4, seed=seed)
        assert sde is None
        assert ret == [stop, stop + 1]
        stops.add(stop)
    assert stops == set(range(4))
