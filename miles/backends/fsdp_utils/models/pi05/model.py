"""Native differentiable π₀.₅ training on unmodified Transformers.

Architecture and tensor names follow Physical-Intelligence/openpi at 215abfb
(Apache-2.0), models_pytorch/{pi0_pytorch,gemma_pytorch,transformers_replace}.
Only π₀.₅'s conditional norms and joint prefix/action attention are specialized;
SigLIP, Gemma projections and MLPs use the installed Transformers implementation.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import GemmaConfig, GemmaForCausalLM, PaliGemmaConfig, PaliGemmaForConditionalGeneration

from .preprocessing import preprocess_images

# width, layers, MLP width, query heads, KV heads, head width (OpenPI Gemma variants).
GEMMA_VARIANTS = {
    "gemma_2b": (2048, 18, 16384, 8, 1, 256),
    "gemma_300m": (1024, 18, 4096, 8, 1, 256),
}


def gemma_config(variant):
    if variant not in GEMMA_VARIANTS:
        raise ValueError(f"Unsupported π₀.₅ Gemma variant: {variant}")
    width, depth, mlp, heads, kv_heads, head_dim = GEMMA_VARIANTS[variant]
    return GemmaConfig(
        hidden_size=width,
        num_hidden_layers=depth,
        intermediate_size=mlp,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=257152,
        hidden_activation="gelu_pytorch_tanh",
        dtype="float32",
    )


def paligemma_config(variant):
    config = PaliGemmaConfig(text_config=gemma_config(variant).to_dict(), image_token_index=257152)
    config.vision_config.intermediate_size = 4304
    config.vision_config.projection_dim = 2048
    config.vision_config.projector_hidden_act = "gelu_fast"
    config._attn_implementation = "eager"
    return config


class Pi05VisionTower(nn.Module):
    """Preserve OpenPI's vision_model key prefix around the Transformers 5 SigLIP tower."""

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, image):
        return self.vision_model(image)


class Pi05RMSNorm(nn.Module):
    """OpenPI RMS/AdaRMS: return normalized activations and the residual gate."""

    def __init__(self, width, cond_dim=None):
        super().__init__()
        if cond_dim is None:
            self.weight = nn.Parameter(torch.zeros(width))
            self.dense = None
        else:
            self.dense = nn.Linear(cond_dim, 3 * width)
            nn.init.zeros_(self.dense.weight)

    def forward(self, x, cond=None):
        normed = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        if self.dense is None:
            return (normed * (1 + self.weight.float())).to(x.dtype), None
        if cond is None:
            raise ValueError("π₀.₅ action expert requires the timestep condition")
        modulation = self.dense(cond)
        if x.ndim == 3:
            modulation = modulation[:, None, :]
        scale, shift, gate = modulation.chunk(3, dim=-1)
        return (normed * (1 + scale.float()) + shift.float()).to(x.dtype), gate.to(x.dtype)


def install_norms(model, width, *, adaptive):
    # Replace modules on this model instance only; never patch the global HF classes.
    for layer in model.layers:
        layer.input_layernorm = Pi05RMSNorm(width, width if adaptive else None)
        layer.post_attention_layernorm = Pi05RMSNorm(width, width if adaptive else None)
    model.norm = Pi05RMSNorm(width, width if adaptive else None)


def gated_residual(x, y, gate):
    return x + y if gate is None else x + y * gate


def rotary(query, key, positions):
    # OpenPI/Gemma uses float32 rotary angles and the split-half rotation convention.
    dim = query.shape[-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=query.device).float() / dim))
    angles = (inv_freq[None, :, None].expand(positions.shape[0], -1, 1) @ positions[:, None, :].float()).transpose(
        1, 2
    )
    angles = torch.cat([angles, angles], dim=-1)
    cos, sin = angles.cos()[:, None].to(query.dtype), angles.sin()[:, None].to(query.dtype)

    def apply(x):
        first, second = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-second, first], dim=-1) * sin

    return apply(query), apply(key)


def joint_layer(prefix_layer, action_layer, prefix, action, mask, positions, condition):
    layers, values, conditions = (prefix_layer, action_layer), (prefix, action), (None, condition)
    queries, keys, vals, gates = [], [], [], []
    for layer, value, cond in zip(layers, values, conditions, strict=True):
        hidden, gate = layer.input_layernorm(value, cond)
        shape = (*hidden.shape[:-1], -1, layer.self_attn.head_dim)
        queries.append(layer.self_attn.q_proj(hidden).view(shape).transpose(1, 2))
        keys.append(layer.self_attn.k_proj(hidden).view(shape).transpose(1, 2))
        vals.append(layer.self_attn.v_proj(hidden).view(shape).transpose(1, 2))
        gates.append(gate)
    query, key = rotary(torch.cat(queries, dim=2), torch.cat(keys, dim=2), positions)
    value = torch.cat(vals, dim=2)
    groups = query.shape[1] // key.shape[1]
    key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    scores = (query @ key.transpose(-2, -1)) * prefix_layer.self_attn.scaling
    probs = F.softmax(scores + mask, dim=-1, dtype=torch.float32).to(query.dtype)
    attended = (probs @ value).transpose(1, 2).contiguous().flatten(2)
    outputs = []
    offset = 0
    for layer, hidden, cond, gate in zip(layers, values, conditions, gates, strict=True):
        size = hidden.shape[1]
        hidden = gated_residual(hidden, layer.self_attn.o_proj(attended[:, offset : offset + size]), gate)
        normed, gate = layer.post_attention_layernorm(hidden, cond)
        outputs.append(gated_residual(hidden, layer.mlp(normed), gate))
        offset += size
    return outputs


class Pi05Policy(nn.Module):
    """Joint training forward with OpenPI-compatible state dicts and velocity output."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        if not config.pi05:
            raise ValueError("This backend implements π₀.₅ only")
        prefix_config = paligemma_config(config.paligemma_variant)
        expert_config = gemma_config(config.action_expert_variant)
        prefix_shape = prefix_config.text_config
        if (
            prefix_shape.num_hidden_layers,
            prefix_shape.num_attention_heads,
            prefix_shape.num_key_value_heads,
            prefix_shape.head_dim,
        ) != (
            expert_config.num_hidden_layers,
            expert_config.num_attention_heads,
            expert_config.num_key_value_heads,
            expert_config.head_dim,
        ):
            raise ValueError("Prefix and action expert must have matching layer and attention geometry")
        # Retain the official hierarchy, including unused LM heads, for strict weight loading.
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = PaliGemmaForConditionalGeneration(prefix_config)
        pali = self.paligemma_with_expert.paligemma.model
        pali.vision_tower = Pi05VisionTower(pali.vision_tower)
        self.paligemma_with_expert.gemma_expert = GemmaForCausalLM(expert_config)
        self.paligemma_with_expert.gemma_expert.model.embed_tokens = None
        install_norms(self.prefix_model, prefix_shape.hidden_size, adaptive=False)
        install_norms(self.action_model, expert_config.hidden_size, adaptive=True)
        width = expert_config.hidden_size
        self.action_in_proj = nn.Linear(config.action_dim, width)
        self.action_out_proj = nn.Linear(width, config.action_dim)
        self.time_mlp_in = nn.Linear(width, width)
        self.time_mlp_out = nn.Linear(width, width)
        self.gradient_checkpointing_enabled = False

    @property
    def prefix_model(self):
        return self.paligemma_with_expert.paligemma.model.language_model

    @property
    def action_model(self):
        return self.paligemma_with_expert.gemma_expert.model

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def _checkpoint(self, fn, *args):
        if self.gradient_checkpointing_enabled and self.training:
            return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
        return fn(*args)

    @staticmethod
    def sample_noise(shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    @staticmethod
    def sample_time(batch_size, device):
        alpha, beta = torch.tensor(1.5, device=device), torch.tensor(1.0, device=device)
        return (torch.distributions.Beta(alpha, beta).sample((batch_size,)) * 0.999 + 0.001).float()

    def forward(self, observation, noisy_actions, time):
        images = preprocess_images(observation["image"], train=self.training)
        embs, masks = [], []
        pali = self.paligemma_with_expert.paligemma.model

        def embed_image(image):
            features = pali.vision_tower(image).last_hidden_state
            return pali.multi_modal_projector(features)

        for name, image in images.items():
            features = self._checkpoint(embed_image, image)
            embs.append(features)
            masks.append(observation["image_mask"][name][:, None].expand(-1, features.shape[1]))
        language = self.prefix_model.embed_tokens(observation["tokenized_prompt"])
        # Transformers 5 GemmaTextScaledWordEmbedding already applies sqrt(hidden_size).
        embs.append(language)
        masks.append(observation["tokenized_prompt_mask"])
        prefix = torch.cat(embs, dim=1)
        action = self.action_in_proj(noisy_actions)
        # Match OpenPI's float64 time sinusoid before casting to the model input dtype.
        fraction = torch.linspace(0, 1, self.action_in_proj.out_features // 2, device=time.device, dtype=torch.float64)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        angles = time[:, None] * (1.0 / period * 2 * math.pi)[None]
        time_emb = torch.cat([angles.sin(), angles.cos()], dim=1).to(time.dtype)
        condition = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))
        prefix_pad = torch.cat(masks, dim=1)
        pad = torch.cat([prefix_pad, torch.ones(action.shape[:2], dtype=torch.bool, device=action.device)], dim=1)
        # Prefix attends only to prefix; all action tokens attend to prefix and actions.
        blocks = torch.cat([torch.zeros_like(prefix_pad), torch.ones_like(pad[:, prefix.shape[1] :])], dim=1)
        allowed = (blocks[:, None, :] <= blocks[:, :, None]) & pad[:, None, :] & pad[:, :, None]
        mask = torch.where(allowed[:, None], 0.0, -2.3819763e38)
        positions = torch.cumsum(pad, dim=1) - 1
        for prefix_layer, action_layer in zip(self.prefix_model.layers, self.action_model.layers, strict=True):
            prefix, action = self._checkpoint(
                joint_layer, prefix_layer, action_layer, prefix, action, mask, positions, condition
            )
        action, _ = self.action_model.norm(action, condition)
        return self.action_out_proj(action.float())
