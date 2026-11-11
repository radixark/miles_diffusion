"""--loss-type and the trainer role select a group's hooks from the loss hub; only custom_loss reads --custom-*-path.

    (loss_type, role) ──► LossHooks(prepare, formula, train_input, train_output)
      policy_loss / actor   Flow-GRPO prepare + formula, no input / output hook
      nft / actor           NFT prepare + formula
      sft_loss / actor      SFT prepare + formula
      dmd2 / actor          generator prepare + DMD2 formula, train_input = select_scored_pairs
      dmd2 / critic         critic prepare + critic formula, train_output = score_last_step_samples
    custom_loss / actor     each --custom-*-path loaded; unset prepare / formula fall back to Flow-GRPO's
    no such (loss_type, role) ──► KeyError

Each built-in entry is checked against the dotted path the argument defaults, the critic overrides or the SFT
recipes used to pass for it, so a hook moving to another function in the table is caught.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.backends.fsdp_utils.loss_hub import LOSS_HOOKS, LossHooks, get_loss_hooks
from miles.backends.fsdp_utils.loss_hub.flow_grpo import flow_grpo_loss_formula, prepare_flow_grpo_batch
from miles.utils.misc import function_registry, load_function

HOOK_PATHS_BY_LOSS_TYPE_AND_ROLE = {
    ("policy_loss", "actor"): dict(
        prepare="miles.backends.fsdp_utils.loss_hub.flow_grpo.prepare_flow_grpo_batch",
        formula="miles.backends.fsdp_utils.loss_hub.flow_grpo.flow_grpo_loss_formula",
    ),
    ("nft", "actor"): dict(
        prepare="miles.backends.fsdp_utils.loss_hub.nft.prepare_nft_batch",
        formula="miles.backends.fsdp_utils.loss_hub.nft.nft_loss_formula",
    ),
    ("sft_loss", "actor"): dict(
        prepare="miles.backends.fsdp_utils.loss_hub.sft.prepare_sft_batch",
        formula="miles.backends.fsdp_utils.loss_hub.sft.sft_loss_formula",
    ),
    ("dmd2", "actor"): dict(
        prepare="miles.backends.fsdp_utils.loss_hub.dmd2.prepare_generator_batch",
        formula="miles.backends.fsdp_utils.loss_hub.dmd2.dmd2_loss_formula",
        train_input="miles.backends.fsdp_utils.loss_hub.dmd2.select_scored_pairs",
    ),
    ("dmd2", "critic"): dict(
        prepare="miles.backends.fsdp_utils.loss_hub.dmd2.prepare_critic_batch",
        formula="miles.backends.fsdp_utils.loss_hub.dmd2.critic_loss_formula",
        train_output="miles.backends.fsdp_utils.loss_hub.dmd2.score_last_step_samples",
    ),
}


def _custom_args(**paths):
    unset = dict(
        custom_prepare_train_batch_path=None,
        custom_loss_function_path=None,
        custom_train_input_postprocess_path=None,
        custom_train_output_function_path=None,
    )
    return Namespace(loss_type="custom_loss", **{**unset, **paths})


@pytest.mark.parametrize(("loss_type", "role"), list(HOOK_PATHS_BY_LOSS_TYPE_AND_ROLE))
def test_builtin_hooks_are_the_functions_the_paths_named(loss_type, role):
    hooks = get_loss_hooks(Namespace(loss_type=loss_type), role)

    assert hooks is LOSS_HOOKS[(loss_type, role)]
    paths = HOOK_PATHS_BY_LOSS_TYPE_AND_ROLE[(loss_type, role)]
    assert hooks == LossHooks(
        prepare=load_function(paths["prepare"]),
        formula=load_function(paths["formula"]),
        train_input=load_function(paths.get("train_input")),
        train_output=load_function(paths.get("train_output")),
    )


def test_the_table_has_no_other_entries():
    assert set(LOSS_HOOKS) == set(HOOK_PATHS_BY_LOSS_TYPE_AND_ROLE)


def test_custom_loss_loads_every_path():
    def prepare(ctx, batch, *, pad_to_len=None):
        pass

    def formula(ctx, batch, prepared, **kwargs):
        pass

    def train_input(ctx, train_pairs):
        pass

    def train_output(ctx, train_pairs):
        pass

    args = _custom_args(
        custom_prepare_train_batch_path="my.prepare",
        custom_loss_function_path="my.formula",
        custom_train_input_postprocess_path="my.train_input",
        custom_train_output_function_path="my.train_output",
    )
    with (
        function_registry.temporary("my.prepare", prepare),
        function_registry.temporary("my.formula", formula),
        function_registry.temporary("my.train_input", train_input),
        function_registry.temporary("my.train_output", train_output),
    ):
        assert get_loss_hooks(args, "actor") == LossHooks(prepare, formula, train_input, train_output)


def test_custom_loss_falls_back_to_flow_grpo_for_unset_prepare_and_formula():
    assert get_loss_hooks(_custom_args(), "actor") == LossHooks(prepare_flow_grpo_batch, flow_grpo_loss_formula)


def test_a_role_without_hooks_for_the_loss_type_is_rejected():
    with pytest.raises(KeyError):
        get_loss_hooks(Namespace(loss_type="nft"), "critic")
