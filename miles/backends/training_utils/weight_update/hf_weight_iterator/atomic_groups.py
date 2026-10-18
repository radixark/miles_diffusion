"""HF-namespace atomic update groups: tensors an sglang-d loader fuses into one engine
parameter or layer, which therefore must arrive in the same bucket."""

from miles.backends.training_utils.weight_update.hf_weight_iterator.bucketing import AtomicUpdateGroup

# One list serves weight and adapter units; the suffix decides which a group matches, so every group is
# optional. lora_merge composes a fused layer from the sections one request carries and replaces the layer's
# LoRA with them; an adapter unit is one module's lora_A + lora_B, matched by its lora_A name.
_H3_LORA_GROUPS = [
    AtomicUpdateGroup("qkv_proj", (".attn.to_q.lora_A", ".attn.to_k.lora_A", ".attn.to_v.lora_A"), optional=True),
]

_ADDED_QKV_LORA_GROUPS = [
    AtomicUpdateGroup(
        "to_added_qkv", (".add_q_proj.lora_A", ".add_k_proj.lora_A", ".add_v_proj.lora_A"), optional=True
    ),
]


def get_hf_atomic_update_groups(diffusion_model_family: str | None) -> list[AtomicUpdateGroup]:
    """Atomic groups for a diffusion model family. H3 fuses to_q/k/v into qkv_proj; Qwen-Image and Cosmos3
    fuse add_q/k/v_proj (into to_added_qkv / cross_attention.to_qkv)."""
    if diffusion_model_family == "h3":
        return list(_H3_LORA_GROUPS)
    if diffusion_model_family in ("qwen_image", "cosmos3"):
        return list(_ADDED_QKV_LORA_GROUPS)
    return []
