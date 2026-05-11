import asyncio

from miles.utils.misc import load_function
from miles.utils.types import Sample


def _resolve_rm_type(args, sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return (metadata.get("rm_type") or args.rm_type or "").strip()


async def async_rm(args, sample: Sample, **kwargs):
    rm_type = _resolve_rm_type(args, sample)

    if rm_type == "ocr":
        from .ocr import ocr_rm

        return await ocr_rm(args, sample)
    elif rm_type == "pickscore":
        from .pickscore import pickscore_rm

        return (await pickscore_rm(args, [sample]))[0]
    else:
        raise NotImplementedError(f"Rule-based RM for {rm_type!r} is not implemented.")


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)

    if samples:
        rm_types = [_resolve_rm_type(args, sample) for sample in samples]
        if all(rm_type == "pickscore" for rm_type in rm_types):
            from .pickscore import pickscore_rm

            return await pickscore_rm(args, samples)

    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
