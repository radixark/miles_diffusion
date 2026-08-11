from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-3-gpu-h200", labels=[])

from argparse import Namespace

from miles.backends.sglang_diffusion_utils.sglang_diffusion_engine import _compute_server_args


def _server_args(**overrides):
    base = dict(
        hf_checkpoint="Qwen/Qwen-Image",
        diffusion_flow_shift=None,
        rollout_num_gpus_per_engine=1,
        sglang_tp_size=1,
        sglang_sp_degree=None,
        sglang_enable_cfg_parallel=False,
        use_lora=True,
        lora_ipc_weight_sync=True,
        lora_target_modules=["to_q", "to_k"],
    )
    base.update(overrides)
    return Namespace(**base)


class TestLoRATargetModulesServerArgs:
    def test_lora_ipc_uses_resolved_args(self):
        args = _server_args()
        kwargs = _compute_server_args(args, "127.0.0.1", 15000, 15001)
        assert kwargs["lora_target_modules"] == ["to_q", "to_k"]

    def test_lora_ipc_omitted_when_disabled(self):
        args = _server_args(lora_ipc_weight_sync=False)
        kwargs = _compute_server_args(args, "127.0.0.1", 15000, 15001)
        assert "lora_target_modules" not in kwargs
