#!/usr/bin/env python3
"""Preprocessing matching the FractalDB ResNet34 model card."""

from __future__ import annotations

from dataclasses import dataclass

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 224
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


def build_eval_transform(config: PreprocessConfig = PreprocessConfig()):
    """Build the deterministic resize and ImageNet normalization transform."""
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size), antialias=True),
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
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=6),
            ]
        )
    operations.extend(
        [
            transforms.Resize((config.image_size, config.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]
    )
    if augment:
        operations.append(
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random")
        )
    return transforms.Compose(operations)
