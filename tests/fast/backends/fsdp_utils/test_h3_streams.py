"""H3 packed forward on the SFT training path, without pretrained weights or encoders.

    cached t2va pair {"latent": {"visual": [rows, 96]}, "cond_kwargs": packed layout + text}
         │ prepare_sft_batch
         ▼
    latents_input {"visual": [1, rows, 96]}, timesteps_input {"visual": [sigma * 1000]}
         │ scatter into the packed video buffer at update_mask
         ▼
    H3 DiT(hidden_states, audio_hidden_states, timestep = 1 - sigma, ...) over packed rows [0, cu_seqlens[1]);
         the padding tail the engine isolates as its own document never reaches it
         ▼
    {"visual": -pred[:, update_mask]}   same [1, rows, 96] as the SFT target ──► sft_loss_formula

What each test pins: the t2va prediction keeps the batch dim, so the SFT loss accepts it; the DiT sees no
padding rows.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import torch
from torch import nn

from miles.backends.fsdp_utils.configs.h3 import H3TrainPipelineConfig
from miles.backends.fsdp_utils.loss_hub.sft import prepare_sft_batch, sft_loss_formula
from miles.backends.fsdp_utils.loss_hub.types import DiffusionLossContext
from miles.backends.fsdp_utils.metrics import new_metric_buffer


class _VisualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, **kwargs):
        self.inputs = kwargs
        return Namespace(sample=kwargs["hidden_states"] * self.scale)


def test_t2va_prediction_matches_the_sft_target_and_skips_padding_rows():
    # Packed rows: [text 0-1 | video 2-4 | audio 5-6 | pad 7]; every t2va video row is a target row.
    layout = {
        "seq_len": 8,
        "img_pos": torch.tensor([2, 3, 4]),
        "audio_pos": torch.tensor([5, 6]),
        "text_pos": torch.tensor([0, 1]),
        "update_mask": torch.tensor([True, True, True]),
        "img_position_ids": torch.arange(24.0).view(8, 3),
        "cu_seqlens": torch.tensor([0, 7, 8]),
    }
    cond = {
        "encoder_hidden_states": torch.ones(1, 2, 8),
        "h3_packed_layout": layout,
        "h3_token_tags": torch.tensor([1, 1, 0, 0, 0, 2, 2, -1]),
    }
    config = H3TrainPipelineConfig()
    context = DiffusionLossContext(
        models={"transformer": _VisualHead()},
        train_pipeline_config=config,
        sde_backend=None,
        args=Namespace(seed=42, fsdp_flow_shift={"visual": 3.0}, log_loss_sigma_bucket=5),
        forward_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    batch = [{"latent": {"visual": torch.ones(3, 96)}, "cond_kwargs": cond}]
    prepared = prepare_sft_batch(context, batch)
    prediction = config.compute_noise_pred(
        model=context.models["transformer"],
        latents_input=prepared.latents,
        timesteps_input=prepared.timesteps_for_model,
        pos_cond=prepared.pos_cond,
        neg_cond=None,
        joint_cond=None,
        use_cfg=False,
        cfg_batching=False,
        guidance_scale=0.0,
        true_cfg_scale=None,
    )
    assert prediction["visual"].shape == prepared.extras["target"]["visual"].shape == (1, 3, 96)
    # cu_seqlens = [0, 7, 8]: the DiT sees rows 0-6 only; pad row 7 (tag -1) is dropped, not clamped.
    inputs = context.models["transformer"].inputs
    assert inputs["token_tags"].tolist() == [1, 1, 0, 0, 0, 2, 2]
    assert torch.equal(inputs["position_ids"], layout["img_position_ids"][:7])
    assert inputs["timestep_indices"].shape == (7,)
    metrics = new_metric_buffer(None, torch.device("cpu"), ["transformer"], sigma_buckets=5)
    loss = sft_loss_formula(
        context, batch, prepared, new_pred=prediction, old_pred=None, ref_pred=None, metrics=metrics
    )
    assert torch.isfinite(loss) and loss.item() > 0
