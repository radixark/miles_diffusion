from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest

import miles.backends.sglang_diffusion_utils.monkey_patches as mp


@pytest.fixture
def isolated_registry(monkeypatch):
    # Swap the registry for a copy (auto-restored) so the real public decorator
    # can be exercised without leaking the dummy group into other tests.
    monkeypatch.setattr(mp, "_ROLLOUT_PATCH_APPLIERS", dict(mp._ROLLOUT_PATCH_APPLIERS))


class TestRolloutPatchGroups:
    # End-to-end group mechanics: a dummy group registered through the public
    # decorator is applied iff its name is listed in MILES_ROLLOUT_PATCH_GROUPS.
    def test_dummy_group_applied_when_selected(self, isolated_registry, monkeypatch):
        calls = []

        @mp.register_rollout_patch_group("dummy")
        def apply_dummy() -> None:
            calls.append("dummy")

        monkeypatch.setenv(mp.ROLLOUT_PATCH_GROUPS_ENV, "dummy")
        mp.apply_env_selected_rollout_patches()
        # Only the selected group ran (built-in appliers would import sglang and fail here).
        assert calls == ["dummy"]

    def test_no_selection_applies_nothing(self, isolated_registry, monkeypatch):
        calls = []
        mp.register_rollout_patch_group("dummy")(lambda: calls.append("dummy"))
        monkeypatch.delenv(mp.ROLLOUT_PATCH_GROUPS_ENV, raising=False)
        mp.apply_env_selected_rollout_patches()
        assert calls == []

    def test_unknown_group_fails_loud(self, monkeypatch):
        monkeypatch.setenv(mp.ROLLOUT_PATCH_GROUPS_ENV, "bogus")
        with pytest.raises(ValueError, match="Unknown rollout patch group"):
            mp.apply_env_selected_rollout_patches()

    def test_builtin_group_registered(self):
        # The decorator ran at import time for the in-repo groups.
        assert "qwen_image" in mp._ROLLOUT_PATCH_APPLIERS
        assert "wan" in mp._ROLLOUT_PATCH_APPLIERS
        assert "ltx" in mp._ROLLOUT_PATCH_APPLIERS
        assert "ltx_av" in mp._ROLLOUT_PATCH_APPLIERS


class TestLtxPatchGroups:
    # The two LTX groups share the cond-kwargs patch; only the video-only group turns AV cross-attention off:
    #   ltx     --> cond kwargs + AV cross-attention off   (video-only training forward)
    #   ltx_av  --> cond kwargs                             (joint audio-video training forward)
    @pytest.fixture
    def applied(self, monkeypatch):
        from miles.backends.sglang_diffusion_utils.monkey_patches import (
            patch_ltx2_disable_av_cross,
            patch_ltx2_rollout_cond_kwargs,
        )

        calls = []
        monkeypatch.setattr(patch_ltx2_rollout_cond_kwargs, "apply", lambda: calls.append("cond_kwargs"))
        monkeypatch.setattr(patch_ltx2_disable_av_cross, "apply", lambda: calls.append("disable_av_cross"))
        return calls

    @pytest.mark.parametrize(
        ("group", "expected"),
        [("ltx", ["cond_kwargs", "disable_av_cross"]), ("ltx_av", ["cond_kwargs"])],
    )
    def test_group_applies_its_patches(self, applied, monkeypatch, group, expected):
        monkeypatch.setenv(mp.ROLLOUT_PATCH_GROUPS_ENV, group)
        mp.apply_env_selected_rollout_patches()
        assert applied == expected


class TestValidateRolloutPatchGroups:
    # The arg-validation entry point behind --rollout-patch-group:
    #   --rollout-patch-group "qwen_image,ltx"  ──► registered appliers ──► pass
    #   --rollout-patch-group "qwen_image,bogus" ─► "bogus" unregistered ─► ValueError
    def test_known_pass_unknown_raises(self):
        mp.validate_rollout_patch_groups(["qwen_image", "ltx", "ltx_av", "wan"])
        with pytest.raises(ValueError, match="Unknown rollout patch group"):
            mp.validate_rollout_patch_groups(["qwen_image", "bogus"])
