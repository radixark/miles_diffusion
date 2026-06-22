from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import os

import pytest

from miles.utils.misc import FunctionRegistry, function_registry, load_function, should_run_periodic_action


def _fn_a():
    return "a"


def _fn_b():
    return "b"


class TestFunctionRegistry:
    def test_register_and_get(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            assert registry.get("my_fn") is _fn_a

    def test_register_duplicate_raises(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            with pytest.raises(AssertionError):
                with registry.temporary("my_fn", _fn_b):
                    pass

    def test_unregister(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            assert registry.get("my_fn") is _fn_a
        assert registry.get("my_fn") is None

    def test_temporary_cleanup_on_exception(self):
        registry = FunctionRegistry()
        with pytest.raises(RuntimeError):
            with registry.temporary("temp_fn", _fn_a):
                raise RuntimeError("test")
        assert registry.get("temp_fn") is None


class TestLoadFunction:
    def test_load_from_module(self):
        import os.path

        assert load_function("os.path.join") is os.path.join

    def test_load_none_returns_none(self):
        assert load_function(None) is None

    def test_load_from_registry(self):
        with function_registry.temporary("test:my_fn", _fn_a):
            assert load_function("test:my_fn") is _fn_a

    def test_registry_takes_precedence(self):
        with function_registry.temporary("os.path.join", _fn_b):
            assert load_function("os.path.join") is _fn_b
        assert load_function("os.path.join") is os.path.join


class TestShouldRunPeriodicAction:
    def test_interval_none_never_runs(self):
        for rid in (0, 1, 5, 99):
            assert should_run_periodic_action(rid, interval=None) is False

    def test_last_rollout_always_runs(self):
        assert should_run_periodic_action(rollout_id=9, interval=4, num_rollout=10) is True

    def test_interval_boundary(self):
        assert should_run_periodic_action(rollout_id=3, interval=4) is True
        assert should_run_periodic_action(rollout_id=2, interval=4) is False
        assert should_run_periodic_action(rollout_id=7, interval=4) is True

    def test_epoch_boundary_triggers(self):
        assert should_run_periodic_action(rollout_id=4, interval=10, num_rollout_per_epoch=5) is True

    @pytest.mark.parametrize("rid", [0, 1, 2])
    def test_no_trigger_at_small_steps(self, rid):
        assert should_run_periodic_action(rollout_id=rid, interval=100, num_rollout=1000) is False
