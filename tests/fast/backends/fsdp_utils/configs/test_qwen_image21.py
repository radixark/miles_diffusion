from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import torch

from miles.backends.fsdp_utils.configs.qwen_image21 import QwenImage21TrainPipelineConfig
from miles.backends.fsdp_utils.configs.train_pipeline_config import TrainPipelineConfig
from miles.utils.types import CondKwargs


class TestQwenImage21Cond:
    def setup_method(self):
        self.cfg = QwenImage21TrainPipelineConfig()
        self.cfg.height = 64
        self.cfg.width = 64

    def test_prepare_appends_target_slots(self):
        enc = torch.randn(8, 16)
        cond = CondKwargs(encoder_hidden_states=[enc], txt_seq_lens=[8])
        out = self.cfg.prepare_cond_kwargs(cond, torch.device("cpu"))
        assert out["encoder_hidden_states"].shape == (1, 8, 16)
        assert out["img_shapes"] == [[(1, 4, 4)]]
        assert out["img_mask"].shape == (1, 8 + 4)  # 4x4 latents / 4 slots
        assert not out["img_mask"][0, :8].any()
        assert out["img_mask"][0, 8:].all()
        assert out["encoder_hidden_states_mask"].shape == (1, 8)
        assert out["encoder_hidden_states_mask"].all()

    def test_collate_pads_text_and_keeps_target_slots(self):
        short = self.cfg.prepare_cond_kwargs(
            CondKwargs(encoder_hidden_states=[torch.randn(3, 8)], txt_seq_lens=[3]),
            torch.device("cpu"),
        )
        long = self.cfg.prepare_cond_kwargs(
            CondKwargs(encoder_hidden_states=[torch.randn(7, 8)], txt_seq_lens=[7]),
            torch.device("cpu"),
        )
        out = self.cfg.collate_cond_for_sample_batch([short, long], torch.device("cpu"), pad_to_len=10)
        assert out["encoder_hidden_states"].shape == (2, 10, 8)
        assert out["encoder_hidden_states_mask"].shape == (2, 10)
        assert out["img_mask"].shape == (2, 10 + 4)
        assert out["img_mask"][:, :10].sum() == 0
        assert out["img_mask"][:, 10:].all()

    def test_cfg_combine_has_no_rescale(self):
        pos = torch.ones(1, 4, 2)
        neg = torch.zeros(1, 4, 2)
        out = self.cfg.cfg_combine(pos, neg, guidance_scale=1.0, true_cfg_scale=2.0)
        torch.testing.assert_close(out, torch.full((1, 4, 2), 2.0))

    def test_timestep_divides_by_1000(self):
        t = torch.tensor([1000.0, 250.0])
        torch.testing.assert_close(self.cfg.process_timestep_as_input(t), t / 1000.0)

    def test_no_legacy_window_pad(self):
        # The window-wide pad exists only so Qwen-Image 1.0 stays bitwise with the old collate.
        assert (
            QwenImage21TrainPipelineConfig.maybe_legacy_window_pad_len
            is TrainPipelineConfig.maybe_legacy_window_pad_len
        )


class _TailDiT(torch.nn.Module):
    def forward(self, hidden_states, timestep, return_dict=False, **kwargs):
        extra = hidden_states.new_ones(hidden_states.shape[0], 3, hidden_states.shape[-1])
        return (torch.cat([extra, hidden_states * 2], dim=1),)


def test_compute_noise_pred_keeps_target_tail():
    cfg = QwenImage21TrainPipelineConfig()
    latents = torch.arange(8.0).reshape(1, 4, 2)
    pred = cfg.compute_noise_pred(
        model=_TailDiT(),
        latents_input=latents,
        timesteps_input=torch.tensor([0.5]),
        pos_cond={},
        neg_cond=None,
        joint_cond=None,
        use_cfg=False,
        cfg_batching=False,
        guidance_scale=1.0,
        true_cfg_scale=1.0,
    )
    torch.testing.assert_close(pred, latents * 2)
