from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

import base64
import pickle
from argparse import Namespace

import pytest
import yaml

from miles.utils.api_rm_config import ApiRewardConfig, load_api_rm_config, resolve_api_rm_configs


@pytest.mark.parametrize("custom_inline", [False, True], ids=["custom-file", "custom-base64"])
@pytest.mark.parametrize("openai_inline", [False, True], ids=["openai-file", "openai-base64"])
def test_both_api_configs_resolve_independently_and_serialize(tmp_path, custom_inline, openai_inline):
    custom_data = {
        "actor_class": "my_rewards.CustomActor",
        "actor_kwargs": {"endpoint": "http://custom-reward.test"},
        "max_concurrency": 3,
    }
    openai_data = {
        "model": "judge",
        "api_key_env": "TEST_RM_KEY",
        "prompt": "Check the image.\n检查图像。",
        "max_concurrency": 7,
    }

    def config_arg(name, data, inline):
        text = yaml.safe_dump(data)
        if inline:
            return "base64:" + base64.b64encode(text.encode()).decode()
        path = tmp_path / f"{name}.yaml"
        path.write_text(text, encoding="utf-8")
        return str(path)

    args = Namespace(
        custom_api_rm_config=config_arg("custom", custom_data, custom_inline),
        openai_api_rm_config=config_arg("openai", openai_data, openai_inline),
    )
    resolve_api_rm_configs(args)
    args = pickle.loads(pickle.dumps(args))

    assert args._custom_api_rm_config == ApiRewardConfig(**custom_data)
    assert args._openai_api_rm_config.actor_kwargs["model"] == "judge"
    assert args._openai_api_rm_config.actor_kwargs["prompt"] == openai_data["prompt"]
    assert args._openai_api_rm_config.max_concurrency == 7


def test_unconfigured_api_rewards_do_not_load_files():
    args = Namespace(custom_api_rm_config=None, openai_api_rm_config=None)
    resolve_api_rm_configs(args)
    assert args._custom_api_rm_config is None
    assert args._openai_api_rm_config is None


@pytest.mark.parametrize("rm_type", ["custom_api", "openai_api"])
def test_invalid_config_reports_its_own_flag(rm_type):
    args = Namespace(custom_api_rm_config=None, openai_api_rm_config=None)
    value = "base64:" + base64.b64encode(b"[]").decode()
    setattr(args, f"{rm_type}_rm_config", value)
    with pytest.raises(ValueError, match=f"--{rm_type.replace('_', '-')}-rm-config must contain a mapping"):
        resolve_api_rm_configs(args)


def test_inline_config_requires_embedded_prompt_instead_of_prompt_path():
    text = yaml.safe_dump({"model": "judge", "api_key_env": "TEST_RM_KEY", "prompt_path": "rubric.txt"})
    value = "base64:" + base64.b64encode(text.encode()).decode()
    with pytest.raises(ValueError, match="inline configs must embed prompt"):
        load_api_rm_config(value)
