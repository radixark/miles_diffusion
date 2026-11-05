from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import pytest

from tests.ci import ci_register

_CUDA_TEST = """\
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-5-gpu-h200")


def test_on_gpu():
    pass
"""

_MAIN_GUARD = """

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
"""

_E2E_TEST = """\
from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(est_time=10, suite="stage-c-5-gpu-h200", script="scripts/example.py", metrics=["m"])
"""


def test_cuda_registration_requires_a_main_guard(tmp_path):
    """run_suite executes CUDA files as `python3 <file>`, so one without the entry passes running no test."""
    silent = tmp_path / "test_silent.py"
    silent.write_text(_CUDA_TEST)
    with pytest.raises(ValueError, match="register_cuda_ci files need"):
        ci_register.ut_parse_one_file(str(silent))

    guarded = tmp_path / "test_guarded.py"
    guarded.write_text(_CUDA_TEST + _MAIN_GUARD)
    assert len(ci_register.ut_parse_one_file(str(guarded))) == 1

    e2e = tmp_path / "test_e2e.py"
    e2e.write_text(_E2E_TEST)
    assert len(ci_register.ut_parse_one_file(str(e2e))) == 1
