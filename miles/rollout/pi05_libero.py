"""SGLang action client with the exact OpenPI transforms used in LIBERO training."""

from __future__ import annotations

import os
import sys

import numpy as np

from miles.backends.fsdp_utils.models.pi05.checkpoint import read_bundle
from miles.backends.fsdp_utils.models.pi05.data import policy_transforms


def configure_libero(checkout, config_dir):
    """Use an isolated config instead of LIBERO's interactive ~/.libero setup."""
    import yaml

    root = checkout.resolve() / "libero" / "libero"
    if not (root / "bddl_files").is_dir():
        raise FileNotFoundError(f"Not a LIBERO checkout: {checkout}")
    # Upstream uses a namespace package that its legacy setup.py does not expose
    # with recent setuptools editable installs. Use the explicit checkout path.
    sys.path.insert(0, str(checkout.resolve()))
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark_root": root,
        "bddl_files": root / "bddl_files",
        "init_states": root / "init_files",
        "assets": root / "assets",
        "datasets": config_dir / "datasets",
    }
    paths["datasets"].mkdir(exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump({key: str(value) for key, value in paths.items()}))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir.resolve())


def reset_libero_env(env, max_attempts=20):
    """Bound placement retries and propagate other errors swallowed by LIBERO's wrapper."""
    from robosuite.utils.errors import RandomizationError

    for attempt in range(max_attempts):
        try:
            return env.env.reset()
        except RandomizationError:
            if attempt == max_attempts - 1:
                raise
    raise ValueError("max_attempts must be positive")


def load_libero_init_states(suite, task_id):
    """Read LIBERO's NumPy states with Torch >= 2.6's restricted unpickler."""
    import torch

    # The upstream files contain NumPy float64 arrays. Keep weights_only enabled
    # instead of allowing arbitrary pickle execution to accommodate older files.
    with torch.serialization.safe_globals(
        [
            np.core.multiarray._reconstruct,
            (np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
            np.ndarray,
            np.dtype,
            type(np.dtype("float64")),
        ]
    ):
        return suite.get_task_init_states(task_id)


class SGLangLiberoPolicy:
    def __init__(self, bundle, endpoint, *, seed=7, num_steps=10, timeout=120):
        import requests

        metadata = read_bundle(bundle)
        if metadata.get("validation_profile"):
            raise ValueError("Local smoke-test bundles cannot be evaluated as pretrained π₀.₅ policies")
        self.model = metadata["model"]
        self.input_transform, self.output_transform = policy_transforms(bundle)
        self.endpoint = endpoint.rstrip("/") + "/v1/actions/generations"
        self.session = requests.Session()
        # The endpoint is stateless and the payload carries explicit noise. Retrying
        # transport failures repeats inference, never advances the simulator twice.
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry = Retry(total=3, backoff_factor=1, allowed_methods={"POST"}, status_forcelist=[502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.rng = np.random.default_rng(seed)
        self.num_steps = num_steps
        self.timeout = timeout

    def infer(self, observation):
        processed = self.input_transform(observation)
        noise = self.rng.standard_normal((self.model["action_horizon"], self.model["action_dim"])).astype(np.float32)
        payload = build_request(processed, noise, self.num_steps)
        response = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        actions = np.asarray(response.json()["actions"], dtype=np.float32)
        if actions.shape != noise.shape or not np.isfinite(actions).all():
            raise ValueError(f"SGLang must return finite padded actions of shape {noise.shape}, got {actions.shape}")
        result = self.output_transform({"state": processed["state"], "actions": actions})["actions"]
        if result.shape != (self.model["action_horizon"], 7) or not np.isfinite(result).all():
            raise ValueError("LIBERO output transform must produce finite [horizon, 7] actions")
        return result

    def close(self):
        self.session.close()


def build_request(processed, noise, num_steps):
    # Explicit tokens preserve pi05_libero's discrete_state_input=False. Sending a
    # raw task would let SGLang tokenize with its different default state convention.
    # Keep images uint8: JSON float [0,1]/[-1,1] inference is ambiguous for constant frames.
    images = processed["image"]
    if any(np.asarray(value).dtype != np.uint8 for value in images.values()):
        raise ValueError("OpenPI policy transforms must produce uint8 images")
    return {
        "input": {
            "observation": {
                "images": {key: np.asarray(value).tolist() for key, value in images.items()},
                "image_masks": {key: bool(value) for key, value in processed["image_mask"].items()},
                "camera_order": list(images),
                "state": np.asarray(processed["state"]).tolist(),
                "tokenized_prompt": np.asarray(processed["tokenized_prompt"]).tolist(),
                "tokenized_prompt_mask": np.asarray(processed["tokenized_prompt_mask"]).tolist(),
                "noise": noise.tolist(),
            }
        },
        "parameters": {"num_inference_steps": num_steps},
        "runtime": {"output_format": "list", "response_format": "raw", "prefix_cache": False},
    }


def libero_observation(obs, prompt):
    from openpi_client import image_tools

    def image(key):
        # LIBERO's renderer is rotated 180 degrees relative to its training dataset.
        value = np.ascontiguousarray(obs[key][::-1, ::-1])
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(value, 224, 224))

    quat = np.asarray(obs["robot0_eef_quat"])
    w = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - w * w)
    rotation = np.zeros(3) if np.isclose(denominator, 0.0) else quat[:3] * (2 * np.arccos(w) / denominator)
    return {
        "observation/image": image("agentview_image"),
        "observation/wrist_image": image("robot0_eye_in_hand_image"),
        "observation/state": np.concatenate([obs["robot0_eef_pos"], rotation, obs["robot0_gripper_qpos"]]),
        "prompt": str(prompt),
    }
