"""HPSv3 reward: Qwen2-VL-7B with a preference head, ported from MizzenAI/HPSv3 to this transformers.

The official package pins transformers 4.45 and copies that release's Qwen2-VL forward, so this
module rebuilds the same computation on the pinned transformers instead of importing it;
tests/fast-gpu/test_hpsv3_alignment.py checks it against scores recorded with the official code.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import ray
import torch
from PIL import Image

from miles.utils.misc import SingletonMeta
from miles.utils.processing_utils import generated_output_to_rgb_hwc_uint8_frames
from miles.utils.types import Sample

from .core import AsyncRewardActorPool, record_reward_queue_depth

_HPSV3_REPO = "MizzenAI/HPSv3"
_HPSV3_FILENAME = "HPSv3.safetensors"
# the checkpoint's own config.json says Qwen2.5-VL, but the weights are Qwen2-VL; the official code builds from this
DEFAULT_BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
_REWARD_TOKEN = "<|Reward|>"
_PIXELS = 256 * 28 * 28
_FACTOR = 28
_UNUSED_CHECKPOINT_KEYS = frozenset({"lm_head.weight"})

# Verbatim from hpsv3/dataset/data_collator_qwen.py (INSTRUCTION, prompt_with_special_token), trailing spaces included.
_INSTRUCTION = """
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. 

**Visual Quality:**  
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. 

**Text Alignment:**  
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
Textual prompt - {text_prompt}


"""
_REWARD_SUFFIX = """
Please provide the overall ratings of this image: <|Reward|>

END
"""


def _smart_resize(height: int, width: int) -> tuple[int, int]:
    """HPSv3's own smart_resize with min_pixels == max_pixels: fit ~256 visual tokens, keep the aspect ratio."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(_FACTOR, round(height / _FACTOR) * _FACTOR)
    w_bar = max(_FACTOR, round(width / _FACTOR) * _FACTOR)
    if h_bar * w_bar > _PIXELS:
        beta = math.sqrt((height * width) / _PIXELS)
        h_bar = math.floor(height / beta / _FACTOR) * _FACTOR
        w_bar = math.floor(width / beta / _FACTOR) * _FACTOR
    elif h_bar * w_bar < _PIXELS:
        beta = math.sqrt(_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / _FACTOR) * _FACTOR
        w_bar = math.ceil(width * beta / _FACTOR) * _FACTOR
    return h_bar, w_bar


def split_official_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    """Map the official (transformers 4.45 Qwen2VLForConditionalGeneration) keys onto Qwen2VLModel and the head."""
    model_state, head_state = {}, {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state["language_model." + key.removeprefix("model.")] = value
        elif key.startswith("visual."):
            model_state[key] = value
        elif key.startswith("rm_head."):
            head_state[key.removeprefix("rm_head.")] = value
        elif key not in _UNUSED_CHECKPOINT_KEYS:
            raise KeyError(f"unexpected key in the HPSv3 checkpoint: {key}")
    return model_state, head_state


def _build_rm_head(hidden_size: int) -> torch.nn.Sequential:
    # the official ranknet head without rm_head_kwargs; indices 0/3/5 match the checkpoint
    return torch.nn.Sequential(
        torch.nn.Linear(hidden_size, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.05),
        torch.nn.Linear(1024, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
    )


class HPSv3Scorer(torch.nn.Module):
    """HPSv3 scorer for aligned prompt/image batches; returns the preference mean (mu) per pair."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        checkpoint_path: str | None = None,
        base_model_path: str = DEFAULT_BASE_MODEL,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        import huggingface_hub
        import safetensors.torch
        from tokenizers import AddedToken
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(base_model_path, padding_side="right")
        # transformers 4.45 (the official reference) only had the PIL-based image processor
        self.processor.image_processor = Qwen2VLImageProcessorPil.from_pretrained(base_model_path)
        tokenizer = self.processor.tokenizer
        tokenizer.add_tokens([AddedToken(_REWARD_TOKEN, special=True, normalized=False)], special_tokens=True)
        self.reward_token_id = tokenizer.convert_tokens_to_ids(_REWARD_TOKEN)

        config = AutoConfig.from_pretrained(base_model_path)
        config.text_config.vocab_size = len(tokenizer)
        with torch.device(self.device):
            self.model = Qwen2VLModel._from_config(
                config, attn_implementation=attn_implementation, dtype=torch.bfloat16
            )
            # dtype= leaves the rotary inv_freq buffers fp32, but HPSv3 was trained and is scored after
            # model.to(bfloat16), which rounds them to bf16; keeping them fp32 shifts mu by up to ~0.2
            self.model.to(torch.bfloat16)
            # the official head stays fp32 and its fp32 autocast upcasts the bf16 hidden state
            self.rm_head = _build_rm_head(config.text_config.hidden_size)

        if checkpoint_path is None:
            checkpoint_path = huggingface_hub.hf_hub_download(_HPSV3_REPO, _HPSV3_FILENAME)
        model_state, head_state = split_official_state_dict(safetensors.torch.load_file(checkpoint_path))
        self.model.load_state_dict(model_state, strict=True)
        self.rm_head.load_state_dict(head_state, strict=True)
        self.eval()

    def build_inputs(self, prompts: Sequence[str], images: Sequence[Image.Image]):
        conversations, resized = [], []
        for prompt, image in zip(prompts, images, strict=True):
            image = image.convert("RGB")
            height, width = _smart_resize(image.height, image.width)
            resized.append(image.resize((width, height), Image.BICUBIC))
            text = _INSTRUCTION.format(text_prompt=prompt) + _REWARD_SUFFIX
            conversations.append([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}])
        texts = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        return self.processor(text=texts, images=resized, padding=True, return_tensors="pt")

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> list[float]:
        inputs = self.build_inputs(prompts, images).to(self.device)
        input_ids = inputs["input_ids"]
        # HPSv3 was trained without M-RoPE: every token uses its sequence index on all three rotary axes
        position_ids = torch.arange(input_ids.shape[1], device=self.device).view(1, 1, -1)
        position_ids = position_ids.expand(3, input_ids.shape[0], -1)
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            position_ids=position_ids,
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            use_cache=False,
        ).last_hidden_state
        reward_mask = input_ids == self.reward_token_id
        if not torch.all(reward_mask.sum(dim=-1) == 1):
            raise RuntimeError(f"expected exactly one {_REWARD_TOKEN} per sample")
        logits = self.rm_head(hidden_states[reward_mask].float())
        return [float(mu) for mu in logits[:, 0].cpu()]


class HPSv3RewardActor:
    def __init__(self, **scorer_kwargs) -> None:
        use_cuda = bool(ray.get_gpu_ids()) and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        self.scorer = HPSv3Scorer(device="cuda" if use_cuda else "cpu", **scorer_kwargs)

    def score_batch(self, outputs: list[torch.Tensor], prompts: list[str]) -> list[float]:
        # round like hps / openai_api, which matches the PNG files the official scorer reads
        images = []
        for output in outputs:
            if output.shape[1] != 1:
                raise ValueError(f"HPSv3 scores images only, got {output.shape[1]} frames")
            (image,) = generated_output_to_rgb_hwc_uint8_frames(output, None, round_normalized=True)
            images.append(Image.fromarray(image))
        return self.scorer(prompts, images)


class AsyncHPSv3Pool(AsyncRewardActorPool, metaclass=SingletonMeta):
    """Ray actor pool for HPSv3 reward inference."""

    def __init__(self, args, placement_group=None, slots=None) -> None:
        super().__init__(
            actor_cls=HPSv3RewardActor,
            actor_kwargs={
                "checkpoint_path": args.hpsv3_checkpoint_path,
                "base_model_path": args.hpsv3_base_model_path,
                "attn_implementation": args.hpsv3_attn_implementation,
            },
            num_workers=args.hpsv3_num_workers,
            batch_size=args.hpsv3_batch_size,
            num_gpus_per_worker=args.hpsv3_num_gpus_per_worker,
            colocate=args.hpsv3_reward_colocate,
            name="hpsv3",
            placement_group=placement_group,
            slots=slots,
        )


async def hpsv3_rm(args, samples: Sequence[Sample]) -> list[float]:
    pool = AsyncHPSv3Pool(args)
    scores, max_queue_depth = await pool.score([s.generated_output for s in samples], [s.prompt for s in samples])
    record_reward_queue_depth(samples, "hpsv3", max_queue_depth)
    return scores
