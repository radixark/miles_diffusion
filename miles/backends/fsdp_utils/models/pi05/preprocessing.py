"""OpenPI-compatible PyTorch image augmentation, without importing OpenPI.

Adapted from Physical-Intelligence/openpi (Apache-2.0), commit 215abfb,
models_pytorch/preprocessing_pytorch.py. Keep random draws/order unchanged.
"""

import torch
import torch.nn.functional as F

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def preprocess_images(images, *, train):
    if not set(IMAGE_KEYS) <= images.keys():
        raise ValueError(f"Expected cameras {IMAGE_KEYS}")
    result = {}
    for key in IMAGE_KEYS:
        image = images[key]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0 * 2.0 - 1.0
        if image.shape[1] == 3:
            image = image.permute(0, 2, 3, 1)
        if image.shape[1:3] != (224, 224):
            raise ValueError("π₀.₅ train pairs require images resized to 224×224 before batching")
        if train:
            image = image / 2.0 + 0.5
            if "wrist" not in key:
                height, width = image.shape[1:3]
                crop_height, crop_width = int(height * 0.95), int(width * 0.95)
                start_h = torch.randint(0, height - crop_height + 1, (1,), device=image.device)
                start_w = torch.randint(0, width - crop_width + 1, (1,), device=image.device)
                image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]
                image = F.interpolate(
                    image.permute(0, 3, 1, 2), size=(height, width), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
                angle = torch.rand(1, device=image.device) * 10 - 5
                if torch.abs(angle) > 0.1:
                    radians = angle * torch.pi / 180.0
                    y, x = torch.meshgrid(
                        torch.linspace(-1, 1, height, device=image.device),
                        torch.linspace(-1, 1, width, device=image.device),
                        indexing="ij",
                    )
                    x, y = x[None].expand(image.shape[0], -1, -1), y[None].expand(image.shape[0], -1, -1)
                    grid = torch.stack(
                        [x * radians.cos() - y * radians.sin(), x * radians.sin() + y * radians.cos()], dim=-1
                    )
                    image = F.grid_sample(
                        image.permute(0, 3, 1, 2), grid, mode="bilinear", padding_mode="zeros", align_corners=False
                    ).permute(0, 2, 3, 1)
            image = image * (0.7 + torch.rand(1, device=image.device) * 0.6)
            contrast = 0.6 + torch.rand(1, device=image.device) * 0.8
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast + mean
            saturation = 0.5 + torch.rand(1, device=image.device)
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation
            image = image.clamp(0, 1) * 2.0 - 1.0
        result[key] = image.permute(0, 3, 1, 2)
    return result
