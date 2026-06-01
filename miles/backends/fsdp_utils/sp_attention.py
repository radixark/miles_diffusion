"""序列并行注意力：把 diffusers Wan DiT 的 self-attention 导向 sglang-d 的 USPAttention。

Option B 形态下每个 sp_rank 只持有 1/sp 的 latent token（S_local），整模型在 S_local 上跑（省显存）；
self-attention 内部由 USPAttention 做 Ulysses all-to-all（+Ring）临时聚到全序列、出来再切回 S_local。
切分契约（patchify 后切、norm_out 前 gather）与 RoPE 全局 offset 在 apply_sequence_parallel 里接线。

USPAttention/通信组件复用本地 sglang-d（= rollout 同一份代码），保证算子精度与 sglang-diffusion 对齐。
"""
import functools

import torch
import torch.distributed as dist


class _GatherSequence(torch.autograd.Function):
    """沿序列维 all-gather 各 sp_rank 的 S_local 片，拼回全序列 S。

    forward：按 rank 序拼接（与 USPAttention all-to-all 的重建序一致）。
    backward：各 rank 只取回自己那一段 grad（token 不相交，无需求和）。
    """

    @staticmethod
    def forward(ctx, x, group, sp_rank, sp_size):
        ctx.sp_rank = sp_rank
        ctx.s_local = x.shape[1]
        parts = [torch.empty_like(x) for _ in range(sp_size)]
        dist.all_gather(parts, x.contiguous(), group=group)
        return torch.cat(parts, dim=1)

    @staticmethod
    def backward(ctx, grad):
        start = ctx.sp_rank * ctx.s_local
        return grad[:, start : start + ctx.s_local], None, None, None


def shard_sequence(x, parallel_state):
    """切出本 rank 的连续 S_local 段（dim=1）。原生可微：backward 自动散回全序列。"""
    sp = parallel_state.sp_size
    s = x.shape[1]
    if s % sp:
        raise ValueError(f"序列长度 {s} 不能被 sp_size {sp} 整除（如需可在 patchify 前 padding）")
    s_local = s // sp
    start = parallel_state.sp_rank * s_local
    return x[:, start : start + s_local]


def gather_sequence(x, parallel_state):
    return _GatherSequence.apply(x, parallel_state.sp_group, parallel_state.sp_rank, parallel_state.sp_size)


class WanUSPAttnProcessor:
    """Wan attention processor：self-attn 走 USPAttention，cross-attn 走本地 SDPA（text KV 各 rank 复制）。

    复用原 WanAttnProcessor 的 QKV/RMSNorm/RoPE 计算；rotary_emb 由 apply_sequence_parallel 预切到 S_local，
    故本处理器对 q/k/v 的 [B,S_local,H,D] 直接送入 USPAttention。
    """

    def __init__(self, num_heads, head_dim, compute_dtype=torch.bfloat16):
        from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention

        init_sp_backend(compute_dtype)
        self.usp_attn = USPAttention(num_heads=num_heads, head_size=head_dim, causal=False)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        is_self_attn = encoder_hidden_states is None

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        if is_self_attn:
            hidden_states = self.usp_attn(query, key, value)
        else:
            hidden_states = torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if encoder_hidden_states_img is not None:
            key_img = attn.norm_added_k(attn.add_k_proj(encoder_hidden_states_img)).unflatten(2, (attn.heads, -1))
            value_img = attn.add_v_proj(encoder_hidden_states_img).unflatten(2, (attn.heads, -1))
            hidden_states_img = torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2), key_img.transpose(1, 2), value_img.transpose(1, 2),
                attn_mask=None, dropout_p=0.0, is_causal=False,
            ).transpose(1, 2).flatten(2, 3).type_as(query)
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def init_sp_backend(compute_dtype):
    """训练进程不经 sglang server 启动，没有 global ServerArgs / mixed-precision state。

    - 强制 FA 后端（绕开 server-args 选择；Ring 仅支持 FA/SAGE）。
    - 设 compute dtype：USPAttention 构造时按 get_compute_dtype() 选后端，默认 fp32 会把 FA 降级成 SDPA。
    - 持久化 forward_context：USPAttention.forward 依赖；用 module-global 而非 with-scope，
      以便 gradient checkpointing 反向 recompute（在 backward 阶段、原 with 已退出）仍可读到。
    """
    from sglang.multimodal_gen.runtime.layers.attention.selector import global_force_attn_backend
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang.multimodal_gen.utils import set_mixed_precision_policy

    global_force_attn_backend(AttentionBackendEnum.FA)
    set_mixed_precision_policy(param_dtype=compute_dtype, reduce_dtype=torch.float32)

    from sglang.multimodal_gen.runtime.managers import forward_context as fc

    if fc._forward_context is None:
        fc._forward_context = fc.ForwardContext(current_timestep=0, attn_metadata=None)


def _apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def apply_sequence_parallel(transformer, parallel_state, compute_dtype=None):
    """把序列并行接到 Wan transformer：替换 self-attn processor + 装序列切分/聚合 hook。

    切分契约：
    - rope 输出按全局 offset 切到 S_local（每 block 复用同一份，故包在 rope 产出处）。
    - hidden_states 在第一个 block 前切到 S_local、最后一个 block 后 all-gather 回全序列。
    - forward_context 由 init_sp_backend 持久化（USPAttention 依赖，含 checkpointing recompute）。
    """
    heads = transformer.config.num_attention_heads
    head_dim = transformer.config.attention_head_dim
    # compute dtype 决定 USPAttention 选 FA 还是被降级成 SDPA：须传 forward dtype（bf16），
    # 而非 FSDP master 参数 dtype（可能是 fp32）。
    if compute_dtype is None:
        compute_dtype = next(transformer.parameters()).dtype
    transformer.set_attn_processor(WanUSPAttnProcessor(heads, head_dim, compute_dtype))

    # rope 在 forward 开头调用一次、其输出被所有 block 复用 —— 在产出处切到本 rank 的 S_local。
    rope = transformer.rope
    orig_rope_forward = rope.forward

    @functools.wraps(orig_rope_forward)
    def sp_rope_forward(hidden_states):
        cos, sin = orig_rope_forward(hidden_states)
        return shard_sequence(cos, parallel_state), shard_sequence(sin, parallel_state)

    rope.forward = sp_rope_forward

    blocks = transformer.blocks

    def slice_block_input(module, args):
        hs, *rest = args
        return (shard_sequence(hs, parallel_state), *rest)

    def gather_block_output(module, args, output):
        return gather_sequence(output, parallel_state)

    blocks[0].register_forward_pre_hook(slice_block_input)
    blocks[-1].register_forward_hook(gather_block_output)
