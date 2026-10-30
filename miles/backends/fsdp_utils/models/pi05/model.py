"""Expose OpenPI's FP32 training velocity through the miles model-forward contract."""

import torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks


class Pi05Policy(PI0Pytorch):
    """Keep OpenPI's modules/state dict; move corruption and MSE to prepare/loss hooks.

    OpenPI 215abfb's forward returns MSE only. This boundary follows its joint
    training forward, reusing the official embeddings, masks, transformer and
    checkpoint helpers. It deliberately does not use the inference KV-cache path.
    """

    def forward(self, observation, noisy_actions, time):
        images, img_masks, tokens, token_masks, state = self._preprocess_observation(observation, train=True)
        prefix, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, tokens, token_masks)
        suffix, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(state, noisy_actions, time)
        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)
        attention_mask = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def joint_forward(prefix, suffix, attention_mask, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix, suffix],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(joint_forward, prefix, suffix, attention_mask, position_ids, adarms_cond)
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
        return self._apply_checkpoint(self.action_out_proj, suffix_out)
