from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import torch

from miles.backends.fsdp_utils.configs.wan2_2_ti2v import Wan2_2_TI2VTrainPipelineConfig
from miles.utils.types import CondKwargs


class _EchoWan(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.timestep_shape = None

    def forward(self, hidden_states, timestep, **_kwargs):
        self.timestep_shape = tuple(timestep.shape)
        return (hidden_states,)


def test_wan_ti2v_channel_mask_does_not_become_batch_dimension():
    config = Wan2_2_TI2VTrainPipelineConfig()
    channel_mask = torch.ones(48, 2, 16, 28)
    channel_mask[:, 0] = 0
    cond = CondKwargs(
        encoder_hidden_states=[torch.zeros(1, 512, 4)],
        wan_ti2v_reserved_frames_mask=channel_mask,
        wan_ti2v_patch_size=(1, 2, 2),
    )

    per_sample = config.prepare_cond_kwargs(cond, torch.device("cpu"))
    assert per_sample["wan_ti2v_reserved_frames_mask"].shape == (1, 2, 16, 28)

    collated = config.collate_cond_for_sample_batch([per_sample, per_sample], torch.device("cpu"))
    model = _EchoWan()
    latents = torch.zeros(2, 48, 2, 16, 28)
    output = config.compute_noise_pred(
        model=model,
        latents_input=latents,
        timesteps_input=torch.tensor([500.0, 250.0]),
        pos_cond=collated,
        neg_cond=None,
        joint_cond=None,
        use_cfg=False,
        cfg_batching=False,
        guidance_scale=1.0,
        true_cfg_scale=None,
    )

    assert model.timestep_shape == (2, 224)
    assert output.shape == latents.shape
