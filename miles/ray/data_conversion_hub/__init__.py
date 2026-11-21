"""Rollout-to-train-pair conversion hooks for diffusion algorithms."""

from typing import Any, Callable

_LABELS_RESOLUTION = "_labels_rollout_resolution"


def resolve_conversion(args: Any) -> tuple[str, Callable[..., Any]]:
    """The conversion a run will actually build its train pairs with, and how to name it.

    RolloutManager tries --custom-convert-samples-to-train-data-path first and returns from it
    if set, so an --custom-expand-samples-to-train-pairs-path given alongside never runs.
    Anything that needs to reason about the conversion ahead of time -- arg validation -- has
    to resolve it the same way, or it checks a function the run will not call.
    """
    from miles.ray.data_conversion_hub.flow_grpo import expand_samples_to_train_pairs
    from miles.utils.misc import load_function

    for path in (args.custom_convert_samples_to_train_data_path, args.custom_expand_samples_to_train_pairs_path):
        if path:
            return path, load_function(path)
    return "the Flow-GRPO conversion", expand_samples_to_train_pairs


def labels_rollout_resolution(convert: Callable[..., Any]) -> Callable[..., Any]:
    """Declare that a conversion labels its train pairs for mixed-resolution training.

    Two fields: ``resolution``, the (height, width) its sample was rendered at, and
    ``log_prob_rollout``, the log-prob the rollout reported -- kept apart from ``log_prob_old``,
    which ``--diffusion-recompute-old-log-prob`` overwrites with a train-side value.

    The train side keeps a micro-batch to one resolution by reading that label
    (``miles.utils.train_data_utils.check_single_resolution``), so a conversion that does not
    write one would make the check read ``None`` everywhere and pass -- vacuous exactly where
    it matters, since two resolutions of equal token count stack without complaint. Declaring
    it here rather than listing conversions elsewhere means a new one is unsupported until it
    says otherwise, which is the safe default.
    """
    setattr(convert, _LABELS_RESOLUTION, True)
    return convert


def conversion_labels_rollout_resolution(convert: Callable[..., Any]) -> bool:
    """Whether ``convert`` declared itself with :func:`labels_rollout_resolution`."""
    return bool(getattr(convert, _LABELS_RESOLUTION, False))
