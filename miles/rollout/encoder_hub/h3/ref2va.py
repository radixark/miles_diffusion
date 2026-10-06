"""Offline H3 Ref2VA encode: one SFT sample -> one joint video/audio train pair built from SGLang's native H3 helpers.

Pairs hold only CPU tensors, so they load with ``weights_only=True``.
"""

from __future__ import annotations

from argparse import Namespace
from functools import partial
from types import SimpleNamespace

import torch

from miles.rollout.encoder_hub.h3 import common
from miles.rollout.encoder_hub.h3.common import H3_FPS, H3_TEXT_HIDDEN_DIM
from miles.utils.types import Sample

# ---------------------------------------------------------------------------
# Text: the HF Qwen3-VL behind the native presentation stage
# ---------------------------------------------------------------------------


@torch.no_grad()
def _encode_ids(encoder: dict, input_ids: torch.Tensor, **vision_inputs) -> torch.Tensor:
    """HF Qwen adapter for the native H3 presentation stage's encode_ids API."""
    model = encoder["text_encoder"]
    device = encoder["device"]
    host_ids = input_ids.to(device="cpu", dtype=torch.long)[None]
    attention_mask = torch.ones_like(host_ids)
    grids = {}
    for name in ("image_grid_thw", "video_grid_thw"):
        value = vision_inputs.get(name)
        grids[name] = value.to(device="cpu", dtype=torch.long) if value is not None else None
    kwargs = {"input_ids": host_ids.to(device), "attention_mask": attention_mask.to(device)}
    if any(value is not None for value in grids.values()):
        # transformers 5.12 requires modality ids separately from token ids.
        # Native H3 supplies explicit vision-pad tokens in the presentation.
        modality_ids = torch.zeros_like(host_ids)
        modality_ids[host_ids == model.config.image_token_id] = 1
        modality_ids[host_ids == model.config.video_token_id] = 2
        positions, _ = model.get_rope_index(
            input_ids=host_ids,
            mm_token_type_ids=modality_ids,
            image_grid_thw=grids["image_grid_thw"],
            video_grid_thw=grids["video_grid_thw"],
            attention_mask=attention_mask,
        )
        kwargs["position_ids"] = positions.to(device)
        kwargs["mm_token_type_ids"] = modality_ids.to(device)
    for pixel_key, grid_key in (("pixel_values", "image_grid_thw"), ("pixel_values_videos", "video_grid_thw")):
        pixels = vision_inputs.get(pixel_key)
        if pixels is not None:
            kwargs[pixel_key] = pixels.to(device=device, dtype=torch.bfloat16)
            # The vision tower builds its position indices on the grid's device.
            kwargs[grid_key] = grids[grid_key].to(device)
    hidden = model(**kwargs, use_cache=False).last_hidden_state[0].to(torch.bfloat16)
    if list(hidden.shape) != [input_ids.numel(), H3_TEXT_HIDDEN_DIM]:
        raise ValueError(f"unexpected H3 text hidden shape {list(hidden.shape)}")
    return hidden


# ---------------------------------------------------------------------------
# References: native request -> rows in condition order -> native noise augmentation
# ---------------------------------------------------------------------------


def _target_canvas_spec(height: int, width: int) -> dict:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
        MINIMAX_H3_BASE_SHORT_EDGE,
        minimax_h3_resolve_spatial_shape,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
        MINIMAX_H3_FINITE_ASPECT_RATIOS,
    )

    # Grid rounding turns 16:9 into 1344:768, so match the canvas back to the preset the validator accepts.
    for short_edge in (MINIMAX_H3_BASE_SHORT_EDGE, min(height, width)):
        for aspect_ratio in MINIMAX_H3_FINITE_ASPECT_RATIOS:
            ratio_width, ratio_height = (int(value) for value in aspect_ratio.split(":"))
            shape = minimax_h3_resolve_spatial_shape(
                width=ratio_width,
                height=ratio_height,
                base_short_edge=short_edge,
            )
            if (int(shape["height"]), int(shape["width"])) == (height, width):
                return {"short_edge": short_edge, "aspect_ratio": aspect_ratio}
    raise ValueError(f"H3 Ref2VA canvas {height}x{width} does not match an admitted aspect-ratio preset")


def _reference_request(sample: Sample, media_clip: dict, seed: int):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
        minimax_h3_prepare_for_queue,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
        minimax_h3_validate_canonical_request,
    )

    height, width = media_clip["video"].shape[-2:]
    canonical = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt=sample.prompt,
        conditions=sample.conditions,
        target={**_target_canvas_spec(height, width), "duration_seconds": common.target_duration_seconds(media_clip)},
        seed=seed,
    )
    batch = SimpleNamespace(extra={"minimax_h3_canonical_request": canonical})
    # Sample media is already local, so SGLang probes it in place and never stages a copy. The returned plan is
    # frozen, and batch.extra now holds the probe facts and material shapes the native encode stages read.
    return batch, minimax_h3_prepare_for_queue(batch)


def _encode_text_and_fill_reference_rows(encoder: dict, batch, plan) -> dict:
    """Return the text payload; the visual and audio reference rows land in ``batch.extra``."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.audio_encoding import (
        MiniMaxH3AudioEncodingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.text_encoding import (
        MiniMaxH3TextEncodingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.visual_encoding import (
        MiniMaxH3VisualEncodingStage,
    )

    # Encode text first: the visual stage frees the decoded reference frames that both stages share.
    text_payload = MiniMaxH3TextEncodingStage._encode_ref2va(
        SimpleNamespace(tokenizer=encoder["tokenizer"], processor=encoder["processor"]),
        batch,
        plan,
        partial(_encode_ids, encoder),
    )["positive"]
    if plan.encoders["visual"]:
        stage = MiniMaxH3VisualEncodingStage(encoder["vae"], encoder["vae_arch"])
        stage._encode_keyframes_from_plan(batch, plan, plan.encoders["visual"])
    audio_indices = set(plan.encoders["audio"])
    audio_materials = [material for material in plan.materials if material.condition_index in audio_indices]
    if audio_materials:
        batch.extra["minimax_h3_reference_audio_rows"] = MiniMaxH3AudioEncodingStage._encode_reference_payload(
            SimpleNamespace(audio_vae=encoder["audio_vae"], vae_arch_config=encoder["audio_vae_arch"]),
            batch,
            plan,
            audio_materials,
        )
    return text_payload


def _rows_by_condition(extra: dict, key: str, field: str) -> dict:
    return {entry["condition_index"]: entry for entry in extra.get(key, {}).get(field, [])}


def _ordered_reference_rows(plan, extra: dict) -> dict:
    """Concatenate the stages' reference rows in condition order, keeping each block's kind and latent shape.

    The packed layout places the blocks by ``blocks``; the noise augmentation draws per block by the shapes.
    """
    images = _rows_by_condition(extra, "minimax_h3_reference_image_rows", "images")
    videos = _rows_by_condition(extra, "minimax_h3_reference_video_rows", "videos")
    audios = _rows_by_condition(extra, "minimax_h3_reference_audio_rows", "audios")
    keyframes = extra.get("minimax_h3_keyframe_cond_rows")
    visual_parts, audio_parts, blocks, visual_shapes, audio_latent_ts = [], [], [], [], []
    if keyframes is not None:
        # Hybrid Ref2VA keyframes precede the independent reference blocks.
        visual_parts.append(keyframes["rows"])
        visual_shapes.extend((1, entry["latent_h"], entry["latent_w"]) for entry in keyframes["keyframes"])
    for material in plan.materials:
        kind, index = material.condition_type, material.condition_index
        if material.material_chain == "image.target_canvas":
            continue
        if kind == "image":
            image = images[index]
            blocks.append({"kind": kind, "latent_h": image["latent_h"], "latent_w": image["latent_w"]})
            visual_parts.append(image["rows"])
            visual_shapes.append((1, image["latent_h"], image["latent_w"]))
            continue
        # audio, video or video_audio, the remaining types SGLang's request validator admits
        audio = audios[index]
        audio_t = int(audio["ref_audio_t"])
        block = {"kind": kind, "ref_audio_t": audio_t}
        if audio_t:
            audio_parts.append(audio["rows"])
            audio_latent_ts.append(audio_t)
        if kind != "audio":
            video = videos[index]
            block.update({key: int(video[key]) for key in ("latent_t", "latent_h", "latent_w")})
            visual_parts.append(video["rows"])
            visual_shapes.append((video["latent_t"], video["latent_h"], video["latent_w"]))
        blocks.append(block)
    return {
        "blocks": blocks,
        "visual": torch.cat(visual_parts) if visual_parts else torch.empty(0, 96),
        "audio": torch.cat(audio_parts) if audio_parts else torch.empty(0, 32),
        "visual_shapes": visual_shapes,
        "audio_latent_ts": audio_latent_ts,
        "keyframes": keyframes,
    }


def _augment_references(references: dict, *, latent_t: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.condition_noise import (
        minimax_h3_audio_cond_noise_aug_rows,
        minimax_h3_imgvid_cond_noise_aug_rows,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
        MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
        MINIMAX_H3_IMGVID_COND_TIMESTEP,
    )

    reference_visual_rows, reference_audio_rows = references["visual"], references["audio"]
    if reference_visual_rows.numel():
        reference_visual_rows = minimax_h3_imgvid_cond_noise_aug_rows(
            reference_visual_rows,
            condition_shapes=references["visual_shapes"],
            target_latent_t=latent_t,
            imgvid_cond_num_frames=len(references["visual_shapes"]),
            seed=seed,
            noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
        )
    if reference_audio_rows.numel():
        reference_audio_rows = minimax_h3_audio_cond_noise_aug_rows(
            reference_audio_rows,
            condition_audio_t=references["audio_latent_ts"],
            seed=seed,
            noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
        )
    return reference_visual_rows.cpu(), reference_audio_rows.cpu()


# ---------------------------------------------------------------------------
# Encoder hub interface: the entry points sft_rollout calls
# ---------------------------------------------------------------------------


def validate_args(args: Namespace) -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_MAX_DURATION_SECONDS,
        MINIMAX_H3_MIN_DURATION_SECONDS,
    )

    common.validate_frame_grid_and_stride(args)
    # raises unless the canvas is an H3 aspect-ratio preset
    _target_canvas_spec(int(args.diffusion_height), int(args.diffusion_width))
    num_frames = int(args.diffusion_output_num_frames)
    duration_seconds = num_frames / H3_FPS
    if not MINIMAX_H3_MIN_DURATION_SECONDS <= duration_seconds <= MINIMAX_H3_MAX_DURATION_SECONDS:
        raise ValueError(
            f"H3 Ref2VA targets must last {MINIMAX_H3_MIN_DURATION_SECONDS:g}-{MINIMAX_H3_MAX_DURATION_SECONDS:g} s, "
            f"got {num_frames} frames = {duration_seconds:g} s"
        )


def load_encoder(args: Namespace, device: torch.device) -> dict:
    from transformers import AutoProcessor

    # Every H3 partition ships byte-identical encoders, so FL2VA's serve Ref2VA; only the DiT differs.
    encoder = common.load_video_and_text_encoders(args, device)
    ckpt_dir = common.checkpoint_dir(args.sft_encoder_checkpoint, ["FL2VA/audio_vae/*", "processor/*"])
    encoder["audio_vae"], encoder["audio_vae_arch"] = common.load_audio_vae(ckpt_dir, device)
    encoder["processor"] = AutoProcessor.from_pretrained(f"{ckpt_dir}/processor")
    return encoder


@torch.no_grad()
def encode_sample(
    encoder: dict, sample: Sample, media_clip: dict, generator: torch.Generator, args: Namespace
) -> dict:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
        MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
        MINIMAX_H3_IMGVID_COND_TIMESTEP,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    # The target audio spans num_frames / 24 s from the first frame, so each frame must sit at its constant 24 fps
    # time; an average of 24 fps still lets variable-rate frames drift off the audio.
    frame_times = media_clip["frame_times_seconds"]
    if frame_times is None or any(
        abs(frame_time - frame_times[0] - index / H3_FPS) > 2e-3 for index, frame_time in enumerate(frame_times)
    ):
        raise ValueError("H3 Ref2VA target requires constant 24 fps video")
    seed = generator.initial_seed()
    target_audio_rows = common.encode_target_audio(encoder, sample, media_clip)
    target_visual_rows, latent_t, latent_h, latent_w = common.encode_target_video(encoder, media_clip)
    batch, plan = _reference_request(sample, media_clip, seed)
    _, frame_count, height, width = media_clip["video"].shape
    resolved = (int(plan.shape["frame_count"]), int(plan.shape["height"]), int(plan.shape["width"]))
    if resolved != (frame_count, height, width):
        raise ValueError(
            f"H3 target {frame_count} frames at {height}x{width} resolves to "
            f"{plan.shape['frame_count']} frames at {plan.shape['height']}x{plan.shape['width']}"
        )
    text_payload = _encode_text_and_fill_reference_rows(encoder, batch, plan)
    references = _ordered_reference_rows(plan, batch.extra)
    keyframes = references["keyframes"]
    keyframe_kwargs = {}
    if keyframes is not None:
        keyframe_kwargs = {
            "keyframe_frame_indices": keyframes["semantic_frame_indices"],
            "frame_count": keyframes["frame_count"],
        }
    packed_layout = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=int(text_payload["text_len"]),
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=target_audio_rows.shape[0] // 2,
        ref_blocks=references["blocks"],
        **keyframe_kwargs,
    )
    token_tags = packed_layout["token_tags"].clone()
    token_tags[packed_layout["text_pos"]] = text_payload["text_token_tags"].to(dtype=torch.long, device="cpu")
    reference_visual_rows, reference_audio_rows = _augment_references(references, latent_t=latent_t, seed=seed)
    # The trainer scatters these rows into the packed layout; check the counts before they are cached.
    layout_rows = [int(packed_layout[mask].sum()) for mask in ("update_mask", "audio_update_mask")]
    layout_rows += [int((~packed_layout[mask]).sum()) for mask in ("update_mask", "audio_update_mask")]
    encoded_rows = [
        rows.shape[0] for rows in (target_visual_rows, target_audio_rows, reference_visual_rows, reference_audio_rows)
    ]
    if layout_rows != encoded_rows:
        raise ValueError(f"H3 packed layout expects {layout_rows} target/reference rows, encoded {encoded_rows}")
    return common.train_pair(
        sample,
        {"visual": target_visual_rows, "audio": target_audio_rows},
        text_payload["hidden_states"][None],
        packed_layout,
        token_tags,
        h3_reference_visual=reference_visual_rows,
        h3_reference_audio=reference_audio_rows,
        h3_reference_visual_timestep=MINIMAX_H3_IMGVID_COND_TIMESTEP,
        h3_reference_audio_timestep=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
        h3_target_has_soundtrack=media_clip["audio_sample_rate"] is not None,
    )
