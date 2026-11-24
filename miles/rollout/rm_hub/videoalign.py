"""VideoAlign / VideoReward (KwaiVGI/VideoReward): Qwen2-VL-2B-Instruct video-text reward.

The model is a Qwen2-VL-2B-Instruct with a linear 'rm_head' (hidden_size -> 1, no bias) on top of
the hidden state of three added special tokens <|VQ_reward|> / <|MQ_reward|> / <|TA_reward|>,
fine-tuned with a LoRA adapter.  Each token ends a section of a fixed rubric, so one forward
returns three raw scores (Visual Quality, Motion Quality, Text Alignment).  videoalign_rm returns
a dict per sample {"vq", "mq", "ta", "overall"}; overall = vq + mq + ta, and --reward-key picks the
entry GRPO trains on (every key also gets its own rollout/reward/<name>_mean panel).

The official implementation normalises each dimension with the means/stds shipped in the
checkpoint's model_config.json (inference_config) before summing the overall score; the port matches
it.  tests/fast-gpu/test_videoalign_alignment.py checks the port against the official package.
"""

from __future__ import annotations

import math
import types
from collections.abc import Sequence

import ray
import torch
from PIL import Image

from miles.utils.misc import SingletonMeta
from miles.utils.output_store import StoredTensor
from miles.utils.processing_utils import generated_output_to_rgb_hwc_uint8_frames, sample_frame_indices
from miles.utils.types import Sample

from .core import AsyncRewardActorPool, record_reward_queue_depth

# checkpoint repo and base model (model_config.json: model_name_or_path)
_VIDEOALIGN_REPO = "KwaiVGI/VideoReward"
_BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

# model_config.json: use_special_tokens + the rubric's three reward tokens
_SPECIAL_TOKENS = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
_DIMENSIONS = ("vq", "mq", "ta")

# model_config.json inference_config: per-dimension (mean, std), applied when --videoalign-normalize
_NORM = {
    "vq": (3.6757, 2.2476),
    "mq": (1.1646, 1.3811),
    "ta": (2.8105, 2.5121),
}

# model_config.json data_config: max_frame_pixels
_DEFAULT_MAX_PIXELS = 200704

# Verbatim DETAILED_PROMPT_WITH_SPECIAL_TOKEN from the official prompt_template.py.  Exact
# whitespace matters for bit-exact scores; the alignment test re-checks this against the official
# string loaded from the overlay, so keep it byte-identical.
_RUBRIC = """
You are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 being the worst and 10 being the best. Each evaluation should be independent of the others.

**Visual Quality:**  
Evaluate the overall visual quality of the video, with a focus on static factors. The following sub-dimensions should be considered:
- **Reasonableness:** The video should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. 

Please provide the ratings of Visual Quality: <|VQ_reward|>
END

**Motion Quality:**  
Assess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following sub-dimensions:
- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing body parts).
- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, mouth movements).
- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.
- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally with the background, without obvious artifacts or the feeling of cut-and-paste effects.
- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the video might have blurry or unsteady sections that hinder visual continuity.
- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.

Please provide the ratings of Motion Quality: <|MQ_reward|>
END

**Text Alignment:**  
Assess how well the video matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, scale, and direction.
- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking if real-world locations or scenes are accurately represented, though some stylistic adaptation is acceptable.  
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the video adheres to this style.
- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) are consistent with the expected behavior from the prompt.

Textual prompt - {text_prompt}
Please provide the ratings of Text Alignment: <|TA_reward|>
END
"""


_IMAGE_FACTOR = 28
_MIN_PIXELS = 4 * 28 * 28
_MAX_RATIO = 200


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(height: int, width: int, max_pixels: int) -> tuple[int, int]:
    """Port of the official smart_resize (vision_process.py) used to size each video frame."""
    if max(height, width) / min(height, width) > _MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {_MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(_IMAGE_FACTOR, _round_by_factor(height, _IMAGE_FACTOR))
    w_bar = max(_IMAGE_FACTOR, _round_by_factor(width, _IMAGE_FACTOR))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, _IMAGE_FACTOR)
        w_bar = _floor_by_factor(width / beta, _IMAGE_FACTOR)
    elif h_bar * w_bar < _MIN_PIXELS:
        beta = math.sqrt(_MIN_PIXELS / (height * width))
        h_bar = _ceil_by_factor(height * beta, _IMAGE_FACTOR)
        w_bar = _ceil_by_factor(width * beta, _IMAGE_FACTOR)
    return h_bar, w_bar


def _preprocess_frame(frame: Image.Image, max_pixels: int) -> Image.Image:
    """Port of the official fetch_image for a video frame: RGB + smart_resize + PIL resize."""
    image = frame.convert("RGB")
    width, height = image.size
    resized_height, resized_width = _smart_resize(height, width, max_pixels)
    return image.resize((resized_width, resized_height))


def _remap_checkpoint_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map the checkpoint (saved with transformers 4.45 + peft 0.10) onto the current naming.

    Three drifts happened since the checkpoint was written:
      * the text model moved from ``Qwen2VLModel.model`` to ``Qwen2VLModel.language_model``;
      * the vision tower moved from the top level to ``Qwen2VLModel.visual``;
      * peft 0.18 stores the original weight of a LoRA-wrapped linear under ``module.base_layer.*``.
    """
    prefix = "base_model.model."
    lora_modules = {key.split(".lora_A.default.weight")[0] for key in state_dict if ".lora_A." in key}

    def remap(key: str) -> str:
        rest = key[len(prefix) :]
        if rest.startswith("visual."):
            rest = "model." + rest
        elif rest.startswith("model."):
            rest = "model.language_model." + rest[len("model.") :]
        return prefix + rest

    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = remap(key)
        for module in lora_modules:
            if key == module + ".weight":
                new_key = remap(module) + ".base_layer.weight"
                break
            if key == module + ".bias":
                new_key = remap(module) + ".base_layer.bias"
                break
        remapped[new_key] = value
    return remapped


def _find_target_linear_names(model, lora_namespan_exclude: Sequence[str]) -> list[str]:
    """Linear/embedding modules that get LoRA adapters; the official excludes lm_head/rm_head/embed_tokens/visual."""
    linear_cls = torch.nn.Linear
    embedding_cls = torch.nn.Embedding
    names = []
    for name, module in model.named_modules():
        if any(excluded in name for excluded in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            names.append(name)
    return names


def _official_vision_attention(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
    """Qwen2-VL vision attention as the official implementation computes it: one sdpa over every
    frame's patches with a block-diagonal mask. transformers 5 instead calls sdpa per frame without a
    mask, which picks a different kernel and moves bf16 scores; this keeps the port bit-exact with the
    checkpoint's reference scores (same patch as hpsv3.py)."""
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision

    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
    mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), mask, dropout_p=0.0
    )
    return self.proj(out.transpose(0, 1).reshape(seq_length, -1))


# subclassing needs the parent at class-definition time; videoalign is only imported lazily by rm_hub
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration


class _Qwen2VLRewardModel(Qwen2VLForConditionalGeneration):
    """Ported Qwen2VLRewardModelBT: Qwen2-VL-2B plus a scalar head read at the reward tokens.

    Subclasses Qwen2VLForConditionalGeneration so the checkpoint's PeftModel state dict keys match;
    rm_head is Linear(hidden_size, 1, bias=False) and forward returns {"logits": [B, 3]} (VQ, MQ, TA
    in order), read at the three special tokens.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.output_dim = 1
        self.rm_head = torch.nn.Linear(config.text_config.hidden_size, self.output_dim, bias=False)
        self.reward_token = "special"
        self.special_token_ids: list[int] = []

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor = None,
        labels=None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_grid_thw: torch.LongTensor = None,
        video_grid_thw: torch.LongTensor = None,
        **kwargs,
    ) -> dict:
        # The reference calls the text model directly: all three rotary axes use sequential positions.
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1)
            position_ids = position_ids.expand(3, input_ids.shape[0], -1)
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state  # [B, L, D]
        logits = self.rm_head(hidden_states)  # [B, L, 1]

        batch_size = input_ids.shape[0]
        special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.special_token_ids:
            special_token_mask = special_token_mask | (input_ids == token_id)
        pooled = logits[special_token_mask, ...]  # [B * 3, 1]
        pooled = pooled.view(batch_size, 3, self.output_dim)  # [B, 3, 1]
        pooled = pooled.view(batch_size, -1)  # [B, 3] (VQ, MQ, TA)
        return {"logits": pooled}


class VideoAlignScorer(torch.nn.Module):
    """VideoAlign scorer for aligned (prompt, video-frames) batches; returns one dict per sample."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        model_path: str = _BASE_MODEL,
        checkpoint_path: str | None = None,
        max_pixels: int = _DEFAULT_MAX_PIXELS,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        from pathlib import Path

        import huggingface_hub
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

        self.device = torch.device(device)
        self.max_pixels = max_pixels
        self.normalize = normalize
        repo = checkpoint_path or _VIDEOALIGN_REPO

        # tokenizer/processor from the base model + the three added reward tokens, as the official does
        self.processor = AutoProcessor.from_pretrained(model_path, padding_side="right")
        self.processor.image_processor = Qwen2VLImageProcessorPil.from_pretrained(model_path)
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": _SPECIAL_TOKENS})
        special_token_ids = self.processor.tokenizer.convert_tokens_to_ids(_SPECIAL_TOKENS)

        # The reward checkpoint contains every weight, so only the base configuration is needed.
        config = AutoConfig.from_pretrained(model_path)
        config._attn_implementation = "sdpa"
        model = _Qwen2VLRewardModel(config)
        model.special_token_ids = list(special_token_ids)
        model.resize_token_embeddings(len(self.processor.tokenizer))
        # match the checkpoint's reference scores: use the official block-diagonal vision attention
        for block in model.model.visual.blocks:
            block.attn.forward = types.MethodType(_official_vision_attention, block.attn)

        # the checkpoint is a full PeftModel state dict (base + LoRA deltas), so rebuild the same
        # LoRA structure before loading it; r/alpha/dropout and the excluded namespaces come from
        # model_config.json peft_lora_config / lora_namespan_exclude.
        target_modules = _find_target_linear_names(model, ["lm_head", "rm_head", "embed_tokens", "visual"])
        peft_config = LoraConfig(
            target_modules=target_modules,
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            use_rslora=False,
            bias="none",
        )
        self.model = get_peft_model(model, peft_config)

        if Path(repo).is_dir():
            model_pth = Path(repo) / "checkpoint-11352" / "model.pth"
            if not model_pth.is_file():
                raise FileNotFoundError(f"VideoAlign checkpoint not found at {model_pth}")
        else:
            model_pth = huggingface_hub.hf_hub_download(repo, "checkpoint-11352/model.pth")
        state_dict = _remap_checkpoint_state_dict(torch.load(model_pth, map_location="cpu", weights_only=True))
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected VideoAlign checkpoint keys: {unexpected[:5]} ...")
        if missing:
            raise RuntimeError(f"missing VideoAlign checkpoint keys: {missing[:5]} ...")

        self.model.to(self.device, dtype=torch.bfloat16)
        self.model.eval()

    def build_inputs(self, prompts: Sequence[str], videos: Sequence[list[Image.Image]]):
        resized = [[_preprocess_frame(frame, self.max_pixels) for frame in frames] for frames in videos]
        conversations = []
        for prompt, frames in zip(prompts, videos, strict=True):
            text = _RUBRIC.format(text_prompt=prompt)
            conversations.append(
                [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": text}]}]
            )
        texts = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        pixels, grids = [], []
        image_processor = self.processor.image_processor
        patch = image_processor.patch_size
        temporal = image_processor.temporal_patch_size
        for i, frames in enumerate(resized):
            # The PIL processor duplicates each image across the temporal patch. Repack the
            # first copy from consecutive frames to reproduce the reference video processor.
            frames = frames + [frames[-1]] * (-len(frames) % temporal)
            images = image_processor(images=frames, return_tensors="pt")
            _, grid_h, grid_w = images["image_grid_thw"][0].tolist()
            frame_pixels = images["pixel_values"].reshape(len(frames), grid_h * grid_w, 3, temporal, patch, patch)
            video_pixels = frame_pixels[:, :, :, 0].reshape(-1, temporal, grid_h * grid_w, 3, patch, patch)
            video_pixels = video_pixels.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3 * temporal * patch * patch)
            grid = [len(frames) // temporal, grid_h, grid_w]
            pixels.append(video_pixels)
            grids.append(grid)
            token_count = math.prod(grid) // image_processor.merge_size**2
            texts[i] = texts[i].replace("<|video_pad|>", "<|video_pad|>" * token_count, 1)
        inputs = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
        for token_id in self.processor.tokenizer.convert_tokens_to_ids(_SPECIAL_TOKENS):
            if not torch.all((inputs["input_ids"] == token_id).sum(dim=1) == 1):
                raise ValueError("Each VideoAlign input must contain exactly one of each reward token")
        inputs["pixel_values_videos"] = torch.cat(pixels)
        inputs["video_grid_thw"] = torch.tensor(grids)
        return inputs

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], videos: Sequence[list[Image.Image]]) -> list[dict[str, float]]:
        inputs = self.build_inputs(prompts, videos)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        logits = self.model(**inputs)["logits"]  # [B, 3]
        rewards = []
        for row in logits.detach().float().cpu().tolist():
            dims = dict(zip(_DIMENSIONS, row, strict=True))
            if self.normalize:
                for dim in _DIMENSIONS:
                    mean, std = _NORM[dim]
                    dims[dim] = (dims[dim] - mean) / std
            dims["overall"] = dims["vq"] + dims["mq"] + dims["ta"]
            rewards.append(dims)
        return rewards


class VideoAlignRewardActor:
    def __init__(self, *, num_frames: int | None, **scorer_kwargs) -> None:
        self.num_frames = num_frames
        use_cuda = bool(ray.get_gpu_ids()) and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        self.scorer = VideoAlignScorer(device="cuda" if use_cuda else "cpu", **scorer_kwargs)

    def score_batch(self, outputs: list[torch.Tensor], prompts: list[str]) -> list[dict[str, float]]:
        videos = []
        for output in outputs:
            frames = generated_output_to_rgb_hwc_uint8_frames(output, self.num_frames)
            videos.append([Image.fromarray(frame) for frame in frames])
        return self.scorer(prompts, videos)


class AsyncVideoAlignPool(AsyncRewardActorPool, metaclass=SingletonMeta):
    """Ray actor pool for VideoAlign reward inference."""

    def __init__(self, args, placement_group=None, slots=None) -> None:
        super().__init__(
            args=args,
            actor_cls=VideoAlignRewardActor,
            actor_kwargs={
                "num_frames": args.videoalign_num_frames,
                "model_path": args.videoalign_model_path or _BASE_MODEL,
                "checkpoint_path": args.videoalign_checkpoint_path,
                "max_pixels": args.videoalign_max_pixels,
                "normalize": not args.videoalign_no_normalize,
            },
            num_workers=args.videoalign_num_workers,
            batch_size=args.videoalign_batch_size,
            num_gpus_per_worker=args.videoalign_num_gpus_per_worker,
            colocate=args.videoalign_reward_colocate,
            name="videoalign",
            placement_group=placement_group,
            slots=slots,
        )


async def videoalign_rm(args, samples: Sequence[Sample]) -> list[dict[str, float]]:
    pool = AsyncVideoAlignPool(args)
    # pick --videoalign-num-frames here so only the scored frames cross the object store; a stored
    # output is not here to pick from, so its actor picks them
    outputs = [
        (
            output
            if isinstance(output, StoredTensor)
            else output[:, sample_frame_indices(output.shape[1], args.videoalign_num_frames)]
        )
        for output in (s.generated_output for s in samples)
    ]
    scores, max_queue_depth = await pool.score(outputs, [s.prompt for s in samples])
    record_reward_queue_depth(samples, "videoalign", max_queue_depth)
    return scores
