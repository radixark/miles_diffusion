---
title: Rewards
description: Local and API reward models, weighted mixtures, rm_hub dispatch, and prompt data format.
---
Miles-diffusion scores generated images (or video frames) after each rollout
microgroup. Reward computation lives in `miles/rollout/rm_hub/` and is invoked
from `sglang_diffusion_rollout.generate_and_rm_microgroup`.

For `--custom-rm-path`, `--custom-reward-post-process-path`, and other
`--*-path` hooks, see [Customization](customization.md).

## 1. At a glance

| Stage | Flag | Role |
|---|---|---|
| Reward type | `--rm-type` | Selects `pickscore`, `hps`, `ocr`, or `api`; ignored when `--custom-rm-path` is set |
| API configuration | `--api-rm-config` | YAML configuration for one API reward: model, endpoint, and API key environment variable |
| Per-sample override | `metadata.rm_type` in JSONL | Overrides global `--rm-type` |
| Custom reward / norm | see [Customization](customization.md) | `--custom-rm-path`, `--custom-reward-post-process-path` |

## 2. Reward models

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
| `--pickscore-num-frames` | None | Video frame sampling count |
| `--pickscore-reward-colocate` | False | One worker per rollout GPU (requires `--colocate`) |

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
| `--hps-reward-colocate` | False | One worker per rollout GPU (requires `--colocate`) |

Example from `scripts/run_diffusion_grpo_sd3_hps_sglang.py`:

```bash
--rm-type hps \
--hps-num-workers 1 \
--hps-batch-size 8 \
--hps-version v2.1 \
--hps-reward-colocate
```

### API rewards

Implementation: `miles/rollout/rm_hub/api.py`. OpenAI/Gemini API rewards use the OpenAI-compatible
Chat Completions API. Each request includes the generation prompt and one RGB
image from `sample.generated_output`, encoded as a PNG data URL. This integration
currently supports images only; video and audio outputs are rejected.

Set your provider's key in the shell that launches training, then save one of
the following configurations as `rewards.yaml`. Replace the model placeholder
with an image-capable model ID/version available to your account.

**OpenAI:**

```bash
export OPENAI_API_KEY="your-key"
```

```yaml
model: YOUR_OPENAI_VISION_MODEL
base_url: https://api.openai.com/v1
api_key_env: OPENAI_API_KEY
```

**Gemini:**

```bash
export GEMINI_API_KEY="your-key"
```

```yaml
model: YOUR_GEMINI_VISION_MODEL
base_url: https://generativelanguage.googleapis.com/v1beta/openai/
api_key_env: GEMINI_API_KEY
```

Add these reward arguments to your image training recipe, replacing its existing
reward selection:

```bash
--api-rm-config rewards.yaml \
--rm-type api
```

`model` is the provider's model ID/version, `base_url` is the API endpoint, and
`api_key_env` names the environment variable containing the key. The configuration
stores the environment variable's name, not the key itself.

Each training job uses one fixed API reward configuration and one pool, including
for evaluation. Multiple API models or scoring rubrics in the same job are not
supported. To use a different API metric, change the configuration for a new job.

Pass the key explicitly through your launch script's `execute_train(...,
extra_env_vars={"GEMINI_API_KEY": os.environ["GEMINI_API_KEY"]})` (use the variable
named by `api_key_env`). The launcher does not read the reward YAML or automatically
forward API keys. The shipped Gemini recipe passes its configured key this way.
If submitting a Ray job yourself, include the key in that job's
`runtime_env.env_vars` so the reward worker can read it.
The YAML must be readable by the training driver; the resolved configuration and
rubric text are carried with the training arguments.

#### Scoring and configuration

The default rubric evaluates prompt adherence: requested subjects, attributes,
counts, actions, and spatial relationships. It asks for an integer score from
0 (does not depict the requested content) to 4 (satisfies all observable
requirements). The response must be a JSON object containing only a numeric
`score`, for example `{"score": 3}`. Miles validates that the score is finite
and within the configured range, then returns it as a float.

| YAML field | Default | Meaning |
|---|---|---|
| `model` | Required | Provider model ID/version |
| `base_url` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `api_key_env` | Required | Environment variable containing the API key |
| `prompt` / `prompt_path` | Built-in prompt-adherence rubric | Inline rubric or a text file relative to the YAML; set at most one |
| `score_min` / `score_max` | `0` / `4` | Accepted score range |
| `timeout_s` | `60` | Request deadline in seconds |
| `max_concurrency` | `8` | Maximum concurrent requests in one zero-GPU Ray actor, shared across microgroups |

For a custom rubric, add `prompt_path: rubric.txt` to the configuration.
The rubric should request the same JSON `score` field and describe the score
range. Scores are returned without rescaling.

API rewards reuse `AsyncRewardActorPool` from `rm_hub/core.py`, like OCR and the
GPU rewards. The singleton pool owns one zero-GPU Ray actor, using Ray's
`max_concurrency` to run requests in threads. Other reward actors remain serial
by default. `ApiRewardActor`
converts rollout tensors to images, and `OpenAIImageScorer` handles the HTTP
request and score parsing. The shared pool handles batching, worker selection,
result ordering, and queue-depth metrics. The API actor does not consume colocated
GPU reward slots; API credentials must be available in its Ray runtime environment.

HTTP errors, timeouts, refusals, malformed responses, and invalid scores propagate
to fail the training job after applicable SDK retries (up to two retries for transient errors).
Failed scores are not replaced with zero or dropped. The API client is reused for the actor's lifetime;
failures do not explicitly cancel other queued or in-flight requests.

To support another API protocol, implement a scorer and an actor exposing
`score_batch(outputs, prompts)`, then configure `AsyncRewardActorPool` with that
actor and zero GPUs, and provide an async RM function. The existing pool's
dispatch logic can be reused without changing the OpenAI-compatible scorer.

A standalone API reward returns one float per sample, so leave `--reward-key`
unset. To combine it with local rewards, use the example below.

### Reward placement

Every GPU reward pool is placed one of two ways:

- `--<rm>-reward-colocate`: every worker takes one **slot** on a rollout placement-group bundle,
  sharing that GPU with the train actor and the rollout engine. Colocated pools share one slot ledger, so `hps` and `pickscore` can both colocate without
  overlapping; more workers than bundles is rejected at parse time. Requires `--colocate`.
- Otherwise the pool is **standalone**: default-scheduled at `--<rm>-num-gpus-per-worker`,
  which only lands on GPUs outside every placement group (placement groups reserve their
  GPUs, so Ray never packs these onto rollout GPUs).

`RolloutManager` seats the colocated pools before the first rollout; standalone pools are
built on first use. API rewards use no local GPU slots.

### Combining rewards

`--custom-rm-path` receives `(args, samples)` and can call local scorers or configured API rewards
directly; `--custom-rm-args` is an opaque string the framework hands to that function
through `args`, so the function owns its own config grammar. The shipped example
`miles/rollout/rm_hub/weighted_mixture_rm.py` reads `name=weight,name=weight`:

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

Weights apply to raw scores (HPSv2.1 ≈ 0.25–0.35, PickScore/26 ≈ 0.8–0.9, OCR ∈ [0, 1]), so
pick them with the scales in mind. Colocated pools share one slot ledger, so several rewards
can colocate without overlapping. Rewards receive `generated_output` itself, and every reward actor
quantises it to uint8 on its own terms.

A shipped recipe uses it: `scripts/run_diffusion_grpo_sd3_ocr_pickscore_sglang.py` trains SD3.5 on
0.8 OCR + 0.2 PickScore with `--rollout-shuffle`; see [SD3](../models/sd3/sd3.md) § 5.4 for the
curve and numbers. Shuffling matters more than usual there: with 8 prompts per rollout one hard
batch moves the per-rollout mean visibly.

Using the configuration from [API rewards](#api-rewards),
add these reward arguments to a colocated image training recipe:

```bash
--api-rm-config rewards.yaml \
--custom-rm-path miles.rollout.rm_hub.weighted_mixture_rm.weighted_mixture_rm \
--custom-rm-args "api=0.7,hps=0.3" \
--reward-key weighted \
--hps-version v2.1 \
--hps-reward-colocate
```

This example requires the recipe's `--colocate` flag for HPS placement.
The weights illustrate the syntax; they are not tuned defaults. The API reward can
also be combined with PickScore or OCR. Each local reward retains its model and
placement settings; the API reward uses its YAML settings.

For each sample, this function returns a dictionary such as:

```python
{
    "hps": 0.3,
    "api": 3.0,
    "weighted": 2.19,  # 0.7 * 3.0 + 0.3 * 0.3
}
```

The three custom-reward flags have separate roles:

- `--custom-rm-path` selects the Python function implementing the calculation.
- `--custom-rm-args` supplies the weights interpreted by this example function.
- `--reward-key weighted` selects `sample.reward["weighted"]` for advantage
  computation and training. `weighted` is a dictionary key defined by the
  example, not an instruction to the framework to perform weighting.

All components remain available in reward logs. Selecting `--reward-key hps`
would still compute all components but train on the HPS score alone. A custom RM
that returns a scalar per sample does not need `--reward-key`.

The mixture applies the same raw weighted sum to API scores (default range
[0, 4]) and local scores. Existing advantage normalization is unchanged.
Results are matched to input samples in input order, regardless of request
completion order. A failure in any required component fails the job.

For a complete 0.7 Gemini API + 0.3 HPS example, use
`scripts/run_diffusion_grpo_sd3_hps_gemini_sglang.py` with its inline API
configuration. See [SD3](../models/sd3/sd3.md) § 5.6 for launch instructions and
verification status.

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
SD3 Flow-GRPO recipe (`scripts/run_diffusion_grpo_sd3_ocr_sglang.py`).

### Remote RM (`--rm-type remote_rm`)

The CLI exposes `--rm-url` for a remote reward service, but `rm_hub` has no
built-in `remote_rm` implementation. Selecting it raises `NotImplementedError`.
For OpenAI-compatible image scoring, use [API rewards](#api-rewards).
For other service protocols, use `--custom-rm-path` (see [Customization](customization.md)).

## 3. Call chain

```
generate_and_rm_microgroup()
  → batched_async_rm(args, microgroup)          # rm_hub/__init__.py
    → custom_rm_path?  user batched function
    → all pickscore?   pickscore_rm (batched)
    → all hps?         hps_rm (batched)
    → all ocr?         ocr_rm (batched, one image per actor call)
    → all api?         api_rm (batched)
    → else             per-sample async_rm → local scorer / api / NotImplementedError
  → sample.reward = score
  → RolloutManager._post_process_rewards()      # GRPO advantage normalization
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
| Prompt text | `--input-key input` | Required non-empty string |
| Per-sample metadata | `--metadata-key metadata` | Optional dict; supports `rm_type` override |

### Dataset subsets

Dataset repo: [`rockdu/miles-diffusion-datasets`](https://huggingface.co/datasets/rockdu/miles-diffusion-datasets)

| Subset | Used by |
|---|---|
| `hpdv2/` | SD3 HPS Flow-GRPO |
| `flowgrpo_pickscore/` | PickScore recipes (SD3 NFT, Qwen-Image, Wan2.2, LTX) |
| `flowgrpo_ocr/` | SD3 OCR Flow-GRPO, NFT smoke test |

### Per-sample rm_type override

JSONL `metadata.rm_type` overrides the global `--rm-type` for that sample:

```python
# rm_hub/__init__.py
metadata.get("rm_type") or args.rm_type
```

Mixed rm_types within one microgroup fall back to per-sample dispatch (no
batched local/API fast path).
