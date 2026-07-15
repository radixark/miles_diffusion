from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="stage-b-cpu-h200", labels=[])

import miles.backends.sglang_diffusion_utils.monkey_patches as mp


class TestPatchGroupsApplyOnRealSglang:
    # Every registered group must apply against the real sgl-d in the CI image;
    # ubuntu-latest CI can't run this (its sgl_kernel/triton are MagicMock stubs).
    # Catches patch-target drift when the pinned sglang fork is bumped.
    def test_all_registered_groups_apply(self, monkeypatch):
        assert mp._ROLLOUT_PATCH_APPLIERS, "no patch groups registered"
        monkeypatch.setenv(mp.ROLLOUT_PATCH_GROUPS_ENV, ",".join(mp._ROLLOUT_PATCH_APPLIERS))
        mp.apply_env_selected_rollout_patches()
