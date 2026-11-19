"""Prepare the public PaliGemma tokenizer and an FP32 SGLang pi05 configuration."""

import argparse
import json
from pathlib import Path


def prepare(output):
    from openpi.shared import download
    from transformers import GemmaTokenizer

    if output.exists():
        raise FileExistsError(output)
    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    tokenizer = output.resolve() / "tokenizer"
    GemmaTokenizer(vocab_file=str(path)).save_pretrained(tokenizer)
    config = {
        "tokenizer_name": str(tokenizer),
        "materialize_dtype": "fp32",
        "enable_autocast": False,
        "enable_prefix_cuda_graph": False,
        "enable_action_cuda_graph": False,
    }
    (output / "serving.json").write_text(json.dumps(config, indent=2) + "\n")
    print(output / "serving.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
