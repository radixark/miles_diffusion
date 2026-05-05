# Roadmap

This roadmap describes planned directions for miles-diffusion. It is intended
for external developers and users, and does not represent a strict release
commitment. Priorities may change as the project evolves.

## Text-to-Video RL

We plan to extend miles-diffusion from text-to-image RL to text-to-video RL. The
overall training loop is similar to T2I, but video rollout has higher memory,
compute, and IO pressure.

An initial direction is to start with short videos or few-frame settings, then
extend to longer video generation as the runtime and training recipes mature.
This also makes it possible to study whether policies trained on short temporal
contexts can transfer to longer video generation settings.

## Image Editing and TI2I

We plan to support text-guided image-to-image and image editing workflows. This
includes edit models, condition images, negative images, and other multimodal
conditioning inputs.

Compared with pure T2I, TI2I introduces additional conditioning paths and
model-specific preprocessing. Supporting these workflows will require extending
the diffusion rollout interface and training recipes while keeping the user
experience close to the existing T2I path.

## More Diffusion Backbones

We plan to expand support for more mainstream diffusion and DiT backbones. New
model support should include rollout compatibility, scheduler support, and
log-prob validation against a trusted reference path.

For supported recipes, we aim to keep rollout log-prob drift within an
acceptable tolerance when compared with the reference implementation. The exact
tolerance may depend on the model, scheduler, dtype, and rollout mode.

## Mixed Resolution Training

We plan to explore mixed resolution training and rollout support. Mixed
resolution can make data construction more flexible and may improve hardware
utilization for workloads that naturally contain prompts or tasks at different
image sizes.

This direction may include NaViT-style batching, resolution-aware sampling, and
batch construction strategies that work across T2I, TI2I, and T2V workloads.

## More Tasks and Rewards

We plan to add more task and reward coverage for diffusion RL. The current
direction includes OCR-style rewards, preference or aesthetic rewards, and
custom reward functions provided by users.

The goal is to make it straightforward to attach new reward signals without
rewriting the core rollout and training loop.
