#!/usr/bin/env python3
"""ImageNet-1K preprocessing for already prepared 224x224 RGB images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
BLACK_FILL = (0, 0, 0)


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = IMAGE_SIZE
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


def build_eval_transform(config: PreprocessConfig = PreprocessConfig()):
    """Use the stored square image directly; no crop or letterbox is added."""
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]
    )


def build_train_transform(config: PreprocessConfig = PreprocessConfig()):
    """Apply only conservative online augmentation to the stored image."""
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.01),
            transforms.RandomAffine(
                degrees=8,
                translate=(0.03, 0.03),
                scale=(0.97, 1.03),
                interpolation=InterpolationMode.BILINEAR,
                fill=BLACK_FILL,
            ),
            transforms.Resize((config.image_size, config.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]
    )


def preprocess_rgb_image(image: Image.Image, config: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Prepare one RGB image for the fixed ResNet18 ONNX input."""
    tensor = build_eval_transform(config)(image.convert("RGB"))
    return tensor.unsqueeze(0).numpy().astype(np.float32)


def preprocess_bgr_frame(frame: np.ndarray, config: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Convert an OpenCV BGR frame to the fixed RGB model input."""
    rgb = Image.fromarray(frame[:, :, ::-1].copy(), mode="RGB")
    return preprocess_rgb_image(rgb, config)
