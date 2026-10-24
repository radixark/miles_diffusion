---
title: Rewards
description: Built-in reward models (PickScore, HPS, HPSv3, OCR, API rewards), rm_hub dispatch, and prompt data format.
---
Miles-diffusion scores generated images (or video frames) after each rollout
microgroup. Reward computation lives in `miles/rollout/rm_hub/` and is invoked
from `sglang_diffusion_rollout.generate_and_rm_microgroup`.

For `--custom-rm-path`, `--custom-reward-post-process-path`, and other
`--*-path` hooks, see [Customization](customization.md).

## 1. At a glance

| Stage | Flag | Role |
|---|---|---|
| Reward type | `--rm-type` | Selects built-in scorer (`pickscore`, `hps`, `hpsv3`, `ocr`, `openai_api`, `custom_api`); ignored when `--custom-rm-path` is set |
| API reward config | `--openai-api-rm-config`, `--custom-api-rm-config` | YAML file or inline `base64:<payload>` for the two API rewards |
| Per-sample override | `metadata.rm_type` in JSONL | Overrides global `--rm-type` |
| Custom reward / norm | see [Customization](customization.md) | `--custom-rm-path`, `--custom-reward-post-process-path` |

## 2. Built-in reward models

### PickScore (`--rm-type pickscore`)

Implementation: `miles/rollout/rm_hub/pickscore.py`.

PickScore scores text–image alignment using a CLIP model pair:

- Processor: `--pickscore-processor-path` (e.g.
  `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`)
- Model: `--pickscore-model-path` (e.g. `yuvalkirstain/PickScore_v1`)

Scoring formula:

```
score = exp(logit_scale) * dot(text_emb, image_emb) / 26.0
```

The `/ 26.0` scaling maps raw PickScore logits (~0–26) to roughly 0–1.

PickScore runs as a **Ray actor pool** (`PickScoreRewardActor`) with round-robin
batching. For video outputs, frames are uniformly sampled
(`--pickscore-num-frames`) and scores are averaged.

| Flag | Default | Description |
|---|---|---|
| `--pickscore-num-workers` | 1 | Ray actor count |
| `--pickscore-num-gpus-per-worker` | 1.0 | GPU per worker (non-colocate) |
| `--pickscore-batch-size` | 8 | Batch size per actor |
| `--pickscore-processor-path` | — | Required for pickscore |
| `--pickscore-model-path` | — | Required for pickscore |
| `--pickscore-num-frames` | None | Evenly spaced video frames scored per sample (None = every frame) |
| `--pickscore-reward-colocate` | False | Seat the `--pickscore-num-workers` actors on rollout GPUs, one per placement-group bundle (requires `--colocate`) |

Example from `scripts/run_diffusion_nft_sd3_pickscore.py`:

```bash
--rm-type pickscore \
--pickscore-num-workers 1 \
--pickscore-num-gpus-per-worker 1.0 \
--pickscore-batch-size 8 \
--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
--pickscore-model-path yuvalkirstain/PickScore_v1
```

### HPS (`--rm-type hps`)

Implementation: `miles/rollout/rm_hub/hps.py`.

HPSv2 scores text–image alignment using a preference-tuned CLIP model:

- Model: `ViT-H-14`
- Version: `--hps-version` (`v2.0` or `v2.1`)
- Checkpoint: `--hps-checkpoint-path` (optional; defaults to `xswu/HPSv2`)

Scoring formula:

```
score = diagonal(image_features @ text_features.T)
```

The diagonal pairs each image with its corresponding prompt in the batch.

HPS runs as a **Ray actor pool** (`HPSRewardActor`) with round-robin batching.
It currently accepts image outputs only (`generated_output` with one frame).
Use a custom RM when defining video frame aggregation semantics.

| Flag | Default | Description |
|---|---|---|
| `--hps-num-workers` | 1 | Ray actor count |
| `--hps-num-gpus-per-worker` | 1.0 | GPU per worker (non-colocate) |
| `--hps-batch-size` | 8 | Batch size per actor |
| `--hps-version` | `v2.1` | `v2.0` or `v2.1` checkpoint |
| `--hps-checkpoint-path` | None | Local checkpoint; unset downloads from `xswu/HPSv2` |
| `--hps-reward-colocate` | False | Seat the `--hps-num-workers` actors on rollout GPUs, one per placement-group bundle (requires `--colocate`) |

Example from `scripts/run_diffusion_grpo_sd3_hps_sglang.py`:

```bash
--rm-type hps \
--hps-num-workers 1 \
--hps-batch-size 8 \
--hps-version v2.1 \
--hps-reward-colocate
```

### HPSv3 (`--rm-type hpsv3`)

Implementation: `miles/rollout/rm_hub/hpsv3.py`.

[HPSv3](https://github.com/MizzenAI/HPSv3) is Qwen2-VL-7B-Instruct fine-tuned as a preference model:
the image and a fixed rubric containing the prompt end in a `<|Reward|>` token, and a small MLP
head on that token's hidden state outputs a preference mean μ and an uncertainty σ. The reward is μ.

- Weights: `MizzenAI/HPSv3` (`HPSv3.safetensors`, full bf16 model), or `--hpsv3-checkpoint-path`
- Config / tokenizer / processor: `--hpsv3-base-model-path` (default `Qwen/Qwen2-VL-7B-Instruct`;
  its weights are not loaded). The checkpoint repo's own `config.json` says Qwen2.5-VL but the
  weights are Qwen2-VL, so it is not used.
- Images are resized to ~256 visual tokens (`256·28·28` pixels, aspect ratio kept).

The official package pins `transformers==4.45.2`, so this is a port onto miles' transformers
rather than a dependency. It reproduces the official scorer's quirks that the checkpoint was
trained with: 1D rotary positions instead of Qwen2-VL's M-RoPE, rotary frequencies rounded to
bf16, and an fp32 head. `tests/fast-gpu/test_hpsv3_alignment.py` checks it against scores recorded
with the official package (`tests/fixtures/gen_hpsv3_reference.py`): identical model inputs, μ within
1e-3 with the backbone in fp32, and within the cross-release bf16 kernel noise (≈0.1) as deployed.

μ is an unbounded logit: on SD3.5-medium GRPO rollouts (512², 10 steps) it spanned about −10 to
+16 with a per-rollout mean of 8–10 and std 5–6. It accepts image outputs only (one frame).

| Flag | Default | Description |
|---|---|---|
| `--hpsv3-num-workers` | 1 | Ray actor count |
| `--hpsv3-num-gpus-per-worker` | 1.0 | GPU per worker (non-colocate) |
| `--hpsv3-batch-size` | 8 | Batch size per actor call |
| `--hpsv3-checkpoint-path` | None | Local `HPSv3.safetensors`; unset downloads from `MizzenAI/HPSv3` |
| `--hpsv3-base-model-path` | `Qwen/Qwen2-VL-7B-Instruct` | Config, tokenizer and processor source |
| `--hpsv3-attn-implementation` | `sdpa` | `sdpa` / `flash_attention_2` / `eager` |
| `--hpsv3-reward-colocate` | False | Seat the `--hpsv3-num-workers` actors on rollout GPUs, one per placement-group bundle (requires `--colocate`) |

Each worker keeps ~14.5 GiB of weights resident; on an H100 with sdpa a call takes ~65 ms at batch
1 and ~47 ms per image from batch 8 (peak 15.4 GiB). When colocated that memory sits on a rollout
GPU next to the train actor and the engine, and Ray does not account for it.

Example from `scripts/run_diffusion_grpo_sd3_hpsv3_sglang.py`:

```bash
--rm-type hpsv3 \
--hpsv3-num-workers 1 \
--hpsv3-batch-size 8 \
--hpsv3-reward-colocate
```

### Reward placement

Every GPU reward pool is placed one of two ways:

- `--<rm>-reward-colocate`: every worker takes one **slot** on a rollout placement-group bundle,
  sharing that GPU with the train actor and the rollout engine. Colocated pools share one slot ledger, so `hps` and `pickscore` can both colocate without
  overlapping; more workers than bundles is rejected at parse time. Requires `--colocate`.
- Otherwise the pool is **standalone**: default-scheduled at `--<rm>-num-gpus-per-worker`,
  which only lands on GPUs outside every placement group (placement groups reserve their
  GPUs, so Ray never packs these onto rollout GPUs).

`RolloutManager` seats the colocated pools before the first rollout; standalone pools are
built on first use.

### Combining rewards

`--custom-rm-path` receives `(args, samples)` and can call the built-in scorers
directly; `--custom-rm-args` is an opaque string the framework hands to that function
through `args`, so the function owns its own config grammar. The shipped example
`miles/rollout/rm_hub/weighted_mixture_rm.py` reads `name=weight,name=weight`, with names
from `hps`, `hpsv3`, `pickscore`, `ocr`, `custom_api`, `openai_api`:

```bash
--custom-rm-path miles.rollout.rm_hub.weighted_mixture_rm.weighted_mixture_rm \
--custom-rm-args "hps=0.7,pickscore=0.3" --reward-key weighted \
--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
--pickscore-model-path yuvalkirstain/PickScore_v1 \
--hps-reward-colocate --pickscore-reward-colocate   # each reward keeps its own placement flags
```

The example returns a dict per sample (`{"hps": ..., "pickscore": ..., "weighted": ...}`).
`--reward-key` picks the entry GRPO trains on, and every entry of a dict reward gets its own
`rollout/reward/<name>_mean` panel, so the components stay visible while the sum is optimized.

Weights apply to raw scores (HPSv2.1 ≈ 0.25–0.35, HPSv3 μ unbounded and roughly −10 to +16,
PickScore/26 ≈ 0.8–0.9, OCR ∈ [0, 1], default OpenAI API rubric ∈ [0, 4]), so pick them with the
scales in mind. A zero weight keeps a reward as a monitor only, e.g.
`--custom-rm-args "hpsv3=1.0,hps=0,pickscore=0" --reward-key hpsv3` trains on HPSv3 and plots the
other two to catch reward hacking. Colocated pools share one slot ledger, so several rewards
can colocate without overlapping. Rewards receive `generated_output` itself, and every reward actor
quantises it to uint8 on its own terms.

A shipped recipe uses it: `scripts/run_diffusion_grpo_sd3_ocr_pickscore_sglang.py` trains SD3.5 on
0.8 OCR + 0.2 PickScore with `--rollout-shuffle`; see [SD3](../models/sd3/sd3.md) § 5.4 for the
recipe and numbers and § 9 for the curves. Shuffling matters more than usual there: with 8 prompts per rollout one hard
batch moves the per-rollout mean visibly.

### OCR (`--rm-type ocr`)

Implementation: `miles/rollout/rm_hub/ocr.py`.

OCR reward compares PaddleOCR output against target text embedded in the
prompt. The target is the string between the first pair of double quotes.
Simplified scoring:

```python
target = prompt.split('"')[1]
reward = 1 - levenshtein_distance(recognized, target) / len(target)
```

The implementation also lowercases both sides, strips spaces, treats a
substring hit as perfect (`dist=0`), and caps `dist` at `len(target)` — see
`miles/rollout/rm_hub/ocr.py`.

OCR runs on **CPU** Ray actors (`--ocr-num-workers`, default 4). Used by the
SD3 Flow-GRPO OCR and OCR & PickScore recipes, the three Krea-2 DiffusionNFT recipes, and the
smoke mode of the SD3 DiffusionNFT recipe.

### API rewards (`--rm-type openai_api` / `custom_api`)

Implementation: `miles/rollout/rm_hub/openai_api.py` and `miles/rollout/rm_hub/api.py`.

Each sample is one HTTP request, sent by a single zero-GPU Ray actor that runs up to
`max_concurrency` (default 8) requests at once. The config is a YAML file or inline
`base64:<payload>` with `actor_class`, `actor_kwargs`, and `max_concurrency`; a flat mapping is
taken as the `actor_kwargs` of the default actor.

| `--rm-type` | Config flag | Actor |
|---|---|---|
| `openai_api` | `--openai-api-rm-config` | `OpenAIImageRewardActor` (or a subclass) |
| `custom_api` | `--custom-api-rm-config` | Any `ApiRewardActor` subclass implementing `build_request` / `send_request` / `parse_response`; defaults to `OpenAIImageRewardActor` |

`OpenAIImageRewardActor` sends the generation prompt and the image (PNG data URL) to an
OpenAI-compatible Chat Completions endpoint, with the config `prompt` as the system rubric, and asks
for a JSON `score`. Its kwargs are `model` and `api_key_env` (required), `base_url`, `prompt` or
`prompt_path` (resolved relative to the config file, not allowed in an inline `base64:` config; with neither, a
built-in 0–4 rubric is used), `score_min` / `score_max` (default 0 / 4), and `timeout_s`; scores outside that range
raise. It accepts image outputs only (one frame).

### Remote RM (`--rm-type remote_rm`)

The CLI exposes `--rm-url` for a remote reward service, but **`rm_hub` does not
implement `remote_rm` today** — selecting it raises `NotImplementedError`.
To call an external service, use the API rewards above or a `--custom-rm-path`
function that reads `args.rm_url`.

## 3. Call chain

```
generate_and_rm_microgroup()
  → batched_async_rm(args, microgroup)          # rm_hub/__init__.py
    → custom_rm_path?  user batched function
    → all pickscore?   pickscore_rm (batched)
    → all hps?         hps_rm (batched)
    → all hpsv3?       hpsv3_rm (batched)
    → all ocr?         ocr_rm (batched, one image per actor call)
    → all custom_api?  custom_api_rm (batched, one request per actor call)
    → all openai_api?  openai_api_rm (batched, one request per actor call)
    → else             per-sample async_rm → ocr / pickscore / hps / hpsv3 / custom_api / openai_api / NotImplementedError
  → sample.reward = score

# after the rollout, in the RolloutManager Ray actor:
RolloutManager._convert_samples_to_train_data()
  → _post_process_rewards()                     # GRPO advantage normalization
```

After rollout, `RolloutManager._post_process_rewards` subtracts the mean and
optionally divides by std to produce normalized advantages for training.
Override that path with `--custom-reward-post-process-path` — see
[Customization](customization.md) § Reward.

## 4. Prompt data

### JSONL format

Training prompts are loaded from `.jsonl` files via `miles/utils/diffusion_data.py` and read in
file order; `--rollout-shuffle` permutes them once per epoch (seeded by `--rollout-seed`):

```json
{"input": "A photo of a cat wearing sunglasses"}
{"input": "A logo saying \"Miles\"", "metadata": {"rm_type": "ocr"}}
```

| Field | CLI mapping | Notes |
|---|---|---|
| Prompt text | `--input-key input` | Non-empty string; rows without one are skipped |
| Per-sample metadata | `--metadata-key metadata` | Optional dict; supports `rm_type` override |

### Dataset subsets

Dataset repo: [`rockdu/miles-diffusion-datasets`](https://huggingface.co/datasets/rockdu/miles-diffusion-datasets)

| Subset | Used by |
|---|---|
| `hpdv2/` | SD3 HPS Flow-GRPO |
| `flowgrpo_pickscore/` | PickScore recipes (SD3 NFT, Qwen-Image, Wan2.2, LTX, Cosmos3, Krea-2, MiniMax H3) |
| `flowgrpo_ocr/` | SD3 OCR and OCR & PickScore Flow-GRPO, Krea-2 OCR NFT, NFT smoke test |

### Per-sample rm_type override

JSONL `metadata.rm_type` overrides the global `--rm-type` for that sample:

```python
# rm_hub/__init__.py
metadata.get("rm_type") or args.rm_type
```

Mixed rm_types within one microgroup fall back to per-sample dispatch (no
batched fast path).
