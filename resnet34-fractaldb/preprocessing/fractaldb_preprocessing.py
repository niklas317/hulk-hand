#!/usr/bin/env python3
"""Preprocessing matching the FractalDB ResNet34 model card."""

from __future__ import annotations

from dataclasses import dataclass

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LETTERBOX_FILL = 144


def letterbox_square(image: Image.Image, image_size: int, fill: int = LETTERBOX_FILL) -> Image.Image:
    """Resize without distorting the hand and center it on a square canvas."""
    image = image.convert("RGB")
    width, height = image.size
    scale = image_size / max(width, height)
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resized = image.resize(resized_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), (fill, fill, fill))
    canvas.paste(resized, ((image_size - resized_size[0]) // 2, (image_size - resized_size[1]) // 2))
    return canvas


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 224
    fill: int = LETTERBOX_FILL
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


def build_eval_transform(config: PreprocessConfig = PreprocessConfig()):
    """Build the deterministic resize and ImageNet normalization transform."""
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: letterbox_square(image, config.image_size, config.fill)),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]
    )


def build_train_transform(config: PreprocessConfig = PreprocessConfig(), augment: bool = True):
    """Build FractalDB-style preprocessing with private-session augmentation."""
    operations = []
    if augment:
        operations.extend(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02)],
                    p=0.8,
                ),
                transforms.RandomAffine(
                    degrees=12,
                    translate=(0.06, 0.06),
                    scale=(0.94, 1.06),
                    shear=3,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=(config.fill,) * 3,
                ),
            ]
        )
    operations.extend(
        [
            transforms.Lambda(lambda image: letterbox_square(image, config.image_size, config.fill)),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]
    )
    return transforms.Compose(operations)
