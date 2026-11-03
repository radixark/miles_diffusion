"""Inference transforms for the local SGLang/LIBERO client."""

from .libero_data import load_data_config


def policy_transforms(bundle):
    from openpi import transforms

    _, data = load_data_config(bundle)
    return (
        transforms.compose(
            [
                *data.data_transforms.inputs,
                transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                *data.model_transforms.inputs,
            ]
        ),
        transforms.compose(
            [
                *data.model_transforms.outputs,
                transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                *data.data_transforms.outputs,
            ]
        ),
    )
