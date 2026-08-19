<h1 align="center">Miles Diffusion</h1>

<p align="center">
  <a href="https://github.com/radixark/miles_diffusion"><img src="https://img.shields.io/badge/github-radixark%2Fmiles__diffusion-black?logo=github" alt="GitHub Repo"></a>
  <a href="https://miles.radixark.com/docs/diffusion"><img src="https://img.shields.io/badge/docs-miles.radixark.com-d55816" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/radixark/miles_diffusion" alt="License"></a>
</p>

<p align="center">
  <a href="#news"><strong>News</strong></a> |
  <a href="#quick-start"><strong>Quick Start</strong></a> |
  <a href="#key-features"><strong>Key Features</strong></a> |
  <a href="https://miles.radixark.com/docs/diffusion"><strong>Documentation</strong></a>
</p>

---

## News

- **[2026/08]** 🎉 **Miles-Diffusion Release**: RL post-training for diffusion models is here — Flow-GRPO, DiffusionNFT, and SFT under one trainer, with end-to-end validated recipes for SD3.5, Qwen-Image, Wan2.2-T2V-A14B, LTX-2.3, and the Cosmos3 MoT omni family. [[Docs]](https://miles.radixark.com/docs/diffusion)

## About

**Miles-diffusion** is currently a standalone repository built on [Miles](https://github.com/radixark/miles)' design philosophy, focused on RL post-training for image and video diffusion models. [SGLang-diffusion](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen) serves the rollout, and the DiT trains under **FSDP2** on a backend that co-evolves with Miles' own. Models load from a diffusers pipeline, or from a native package when a family brings its own modeling. Shipped recipes carry explicit [verification levels](https://miles.radixark.com/docs/diffusion/user-guide/recipe-verification). Custom rewards, losses, and rollout functions plug in through flags.

---

## Key Features

- **Verified recipes for the latest diffusion models.** Launchers for Wan2.2-T2V-A14B, Qwen-Image, LTX-2.3, the Cosmos3 MoT omni family, and SD3.5. `TrainPipelineConfig` allows for easy model support.
- **Quality control on three fronts.** Deterministic mode supports bit-for-bit comparisons for recipes covered by committed E2E standards; SGLang-side monkey patches reduce train/rollout mismatches; and an FSDP2 param-dtype patch provides per-parameter fp32 control under the mixed-precision policy. See [Deterministic Training](https://miles.radixark.com/docs/diffusion/advanced/deterministic) and [Dtype Control](https://miles.radixark.com/docs/diffusion/advanced/dtype-control).
- **SFT, DiffusionNFT, and Flow-GRPO under one trainer.** The loss type, training-batch preparation, rollout function, and reward function are all **replaceable components**, so integrating a new algorithm — or swapping in your own customized component — is easy.
- **SGLang native.** Rollout runs **on the inference engine itself** — the SGLang-diffusion serving stack — with RL support and optimizations living engine-side. An optional curated set of monkey patches aligns selected engine operations with the training-side forward.
- **Multiple parallelisms.** The rollout engines scale with **tensor and sequence parallelism** to support large models and very long contexts; training scales with **USP (Ulysses × Ring)**, built from each family's diffusers `_cp_plan` — or a self-written one — for agile model integration.
- **LoRA training support.** With `--lora-ipc-weight-sync`, PEFT LoRA on the FSDP2 actor ships only `lora_A`/`lora_B` pairs to colocated rollout engines over CUDA IPC and merges them engine-side. See [LoRA Training and Weight Sync](https://miles.radixark.com/docs/diffusion/advanced/lora).

## Supported Models

Each model links to its recipe page; every documented recipe is labeled with a [recipe verification level](https://miles.radixark.com/docs/diffusion/user-guide/recipe-verification).


| Model                                                                                                | Task | Canonical Recipes                                                                                                       |
| ---------------------------------------------------------------------------------------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------- |
| **[Stable Diffusion 3.5](https://miles.radixark.com/docs/diffusion/models/sd3/sd3)**                 | T2I  | Flow-GRPO + OCR, DiffusionNFT + PickScore                                                                               |
| **[Qwen-Image](https://miles.radixark.com/docs/diffusion/models/qwen-image/qwen-image)**             | T2I  | Flow-GRPO + PickScore (flow_grpo-aligned)                                                                               |
| **[Wan2.2-T2V-A14B](https://miles.radixark.com/docs/diffusion/models/wan/wan2-2)**                   | T2V  | Flow-GRPO + PickScore, LoRA SFT                                                                                         |
| **[LTX-2.3](https://miles.radixark.com/docs/diffusion/models/ltx/ltx2)**                             | T2V  | Flow-GRPO + PickScore                                                                                                   |
| **[Cosmos3 (Edge / Nano / Super)](https://miles.radixark.com/docs/diffusion/models/cosmos/cosmos3)** | T2I  | Flow-GRPO + PickScore                                                                                                   |
| **[MiniMax H3](https://miles.radixark.com/docs/diffusion/models/h3/h3)**                             | T2VA | Open ([PR #154](https://github.com/radixark/miles_diffusion/pull/154)); 2-GPU recipe, verified; large-scale coming soon |


---

## Quick Start

- [Installation](https://miles.radixark.com/docs/diffusion/getting-started/installation)
- [Launch Training](https://miles.radixark.com/docs/diffusion/getting-started/quick-start#3-launch-training)

---

## Acknowledgements

Miles-diffusion stands on the shoulders of giants and thanks the following repositories for their outstanding work: [Miles](https://github.com/radixark/miles) · [slime](https://github.com/THUDM/slime) · [SGLang](https://github.com/sgl-project/sglang) · [diffusers](https://github.com/huggingface/diffusers) · [VeOmni](https://github.com/ByteDance-Seed/VeOmni) · [Flow-GRPO](https://github.com/yifan123/flow_grpo) · [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) · [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory)

---

## Links

- **GitHub**: [https://github.com/radixark/miles_diffusion](https://github.com/radixark/miles_diffusion)
- **Miles (LLM RL)**: [https://github.com/radixark/miles](https://github.com/radixark/miles)
- **Documentation**: [https://miles.radixark.com/docs/diffusion](https://miles.radixark.com/docs/diffusion)

*From noise, a world takes shape — one step at a time.*

