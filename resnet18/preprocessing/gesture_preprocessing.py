#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from torchvision import transforms


DEFAULT_IMAGE_SIZE = 224
DEFAULT_FILL = 144
DEFAULT_MEAN = (0.54, 0.499, 0.474)
DEFAULT_STD = (0.234, 0.235, 0.231)

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - older Pillow
    RESAMPLE_BILINEAR = Image.BILINEAR


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = DEFAULT_IMAGE_SIZE
    fill: int = DEFAULT_FILL
    mean: tuple[float, float, float] = DEFAULT_MEAN
    std: tuple[float, float, float] = DEFAULT_STD


def letterbox_square(image: Image.Image, image_size: int = DEFAULT_IMAGE_SIZE, fill: int = DEFAULT_FILL) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = image_size / max(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = image.resize((new_width, new_height), RESAMPLE_BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), (fill, fill, fill))
    x0 = (image_size - new_width) // 2
    y0 = (image_size - new_height) // 2
    canvas.paste(resized, (x0, y0))
    return canvas


def build_preprocess_transform(config: PreprocessConfig = PreprocessConfig()):
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: letterbox_square(img, config.image_size, config.fill)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )


def build_train_transform(config: PreprocessConfig = PreprocessConfig(), augment: bool = True):
    ops = []
    if augment:
        ops.extend(
            [
                transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02)], p=0.8),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=6),
            ]
        )
    ops.append(transforms.Lambda(lambda img: letterbox_square(img, config.image_size, config.fill)))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=config.mean, std=config.std))
    if augment:
        ops.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"))
    return transforms.Compose(ops)


def build_eval_transform(config: PreprocessConfig = PreprocessConfig()):
    return build_preprocess_transform(config)


def preprocess_bgr_frame(frame: np.ndarray, config: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    rgb = Image.fromarray(frame[:, :, ::-1].copy())
    tensor = build_preprocess_transform(config)(rgb)
    return tensor.unsqueeze(0).numpy().astype(np.float32)


def preprocess_image_file(input_path: str | Path, output_path: str | Path, config: PreprocessConfig = PreprocessConfig()) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image:
        processed = letterbox_square(image, config.image_size, config.fill)
    processed.save(output_path)
    return output_path


def iter_image_files(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            yield path
