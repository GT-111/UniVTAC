"""Image preprocessing for pi05 inference — PyTorch-only."""

from collections.abc import Sequence
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("openpi")

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
IMAGE_RESOLUTION = (224, 224)


def resize_with_pad_torch(images, height, width, mode="bilinear"):
    """Resize images to target size with padding to avoid distortion."""
    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    batch_size, channels, cur_h, cur_w = images.shape
    ratio = max(cur_w / width, cur_h / height)
    resized_h, resized_w = int(cur_h / ratio), int(cur_w / ratio)

    resized = F.interpolate(images, size=(resized_h, resized_w), mode=mode, align_corners=False if mode == "bilinear" else None)

    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized = resized.clamp(-1.0, 1.0)

    pad_h0, rem_h = divmod(height - resized_h, 2)
    pad_w0, rem_w = divmod(width - resized_w, 2)
    pad_value = 0 if images.dtype == torch.uint8 else -1.0
    padded = F.pad(resized, (pad_w0, pad_w0 + rem_w, pad_h0, pad_h0 + rem_h), mode="constant", value=pad_value)

    if channels_last:
        padded = padded.permute(0, 2, 3, 1)
        if batch_size == 1:
            padded = padded.squeeze(0)
    return padded


def preprocess_observation_pytorch(observation, *, train=False, image_keys=IMAGE_KEYS, image_resolution=IMAGE_RESOLUTION):
    """Preprocess observation for model input."""
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]
    out_images = {}

    for key in image_keys:
        image = observation.images[key]
        is_channels_first = image.shape[1] == 3
        if is_channels_first:
            image = image.permute(0, 2, 3, 1)
        if image.shape[1:3] != image_resolution:
            image = resize_with_pad_torch(image, *image_resolution)
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)
        out_images[key] = image

    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    class ProcessedObs:
        pass
    result = ProcessedObs()
    result.images = out_images
    result.image_masks = out_masks
    result.state = observation.state
    result.tokenized_prompt = observation.tokenized_prompt
    result.tokenized_prompt_mask = observation.tokenized_prompt_mask
    result.token_ar_mask = getattr(observation, 'token_ar_mask', None)
    result.token_loss_mask = getattr(observation, 'token_loss_mask', None)
    return result
