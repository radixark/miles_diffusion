"""Rollout-to-train-pair conversion hooks for diffusion algorithms."""

from typing import Any, Callable, NamedTuple

_LABELS_RESOLUTION = "_labels_rollout_resolution"


class Conversion(NamedTuple):
    """Which function a run builds its train pairs with, and how it is called.

    ``kind`` is "full" for --custom-convert-samples-to-train-data-path, which replaces reward
    post-processing too and takes ``(args, samples)``, or "pairs" for the expand hook and the
    Flow-GRPO default, which take ``(args, samples, rewards, raw_rewards)``.
    """

    kind: str
    where: str
    fn: Callable[..., Any]


def resolve_conversion(args: Any) -> Conversion:
    """The conversion a run will actually build its train pairs with.

    A full override wins and the expand hook beside it never runs, so resolving it anywhere
    else -- arg validation, say -- has to agree, and loading a hook this says is unused would
    fail a run over a function it never calls.
    """
    from miles.ray.data_conversion_hub.flow_grpo import expand_samples_to_train_pairs
    from miles.utils.misc import load_function

    if args.custom_convert_samples_to_train_data_path:
        path = args.custom_convert_samples_to_train_data_path
        return Conversion("full", path, load_function(path))
    if args.custom_expand_samples_to_train_pairs_path:
        path = args.custom_expand_samples_to_train_pairs_path
        return Conversion("pairs", path, load_function(path))
    return Conversion("pairs", "the Flow-GRPO conversion", expand_samples_to_train_pairs)


def labels_rollout_resolution(convert: Callable[..., Any]) -> Callable[..., Any]:
    """Declare that a conversion labels its train pairs for mixed-resolution training.

    Two fields on every pair whose sample carries a drawn resolution: ``resolution``, the
    (height, width) it was rendered at, and ``log_prob_rollout``, the log-prob the rollout
    reported -- a key apart from ``log_prob_old``, which ``--diffusion-recompute-old-log-prob``
    rebinds to a train-side value. Undeclared conversions are unsupported, so one that has not
    thought about resolution cannot leave the train-side checks reading nothing.
    """
    setattr(convert, _LABELS_RESOLUTION, True)
    return convert


def conversion_labels_rollout_resolution(convert: Callable[..., Any]) -> bool:
    """Whether ``convert`` declared itself with :func:`labels_rollout_resolution`."""
    return bool(getattr(convert, _LABELS_RESOLUTION, False))
