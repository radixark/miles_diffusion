from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

import base64
import binascii

import pytest

from miles.utils.file_arg_utils import resolve_file_arg


def test_file_and_inline_payload_preserve_multiline_utf8(tmp_path):
    text = "prompt: |\n  Check the image.\n  检查图像。\n"
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    encoded = "base64:" + base64.b64encode(text.encode()).decode()

    assert resolve_file_arg(str(path)) == resolve_file_arg(encoded) == text


@pytest.mark.parametrize("payload", ["!!!!", "a: 1", "eval:"])
def test_corrupt_inline_payload_is_rejected(payload):
    with pytest.raises(binascii.Error):
        resolve_file_arg(f"base64:{payload}")


def test_missing_file_is_not_treated_as_inline_content(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_file_arg(str(tmp_path / "missing.yaml"))
