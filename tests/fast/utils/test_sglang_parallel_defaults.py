"""The engine-parallelism default: multi-GPU engines prefer TP unless SP/CFG is opted into.

    user config                                  effective engine layout
    (nothing set), 4 GPUs/engine        ->       tp_size=4          (this default)
    --sglang-sp-degree 4                ->       untouched: SP is an explicit numerics opt-in
    --sglang-enable-cfg-parallel        ->       untouched: same
    --sglang-tp-size 2                  ->       untouched: explicit wins

This default once lived on a renamed argparse dest and silently stopped applying, which
flipped multi-GPU engines to SP+CFG and degraded sampling; these tests pin the semantics.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.utils.arguments import set_default_diffusion_args


def _args(**overrides):
    base = dict(
        sglang_tp_size=None,
        sglang_sp_degree=None,
        sglang_enable_cfg_parallel=False,
        rollout_num_gpus_per_engine=4,
        loss_type="grpo",
        custom_expand_samples_to_train_pairs_path=None,
        custom_prepare_train_batch_path=None,
        custom_loss_function_path=None,
        ref_mode="none",
        diffusion_kl_beta=0.0,
    )
    base.update(overrides)
    return Namespace(**base)


def test_multi_gpu_engine_defaults_to_full_tp():
    args = _args()
    set_default_diffusion_args(args)
    assert args.sglang_tp_size == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"sglang_sp_degree": 4},
        {"sglang_enable_cfg_parallel": True},
    ],
)
def test_explicit_sp_or_cfg_disables_the_tp_default(overrides):
    args = _args(**overrides)
    set_default_diffusion_args(args)
    assert args.sglang_tp_size is None


def test_explicit_tp_size_wins():
    args = _args(sglang_tp_size=2, sglang_sp_degree=2)
    set_default_diffusion_args(args)
    assert args.sglang_tp_size == 2
