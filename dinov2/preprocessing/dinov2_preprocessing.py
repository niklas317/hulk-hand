#!/usr/bin/env python3

"""Runtime preprocessing for the DINOv2 gesture model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 224
    fill: tuple[int, int, int] = (0, 0, 0)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


def preprocess_bgr_frame(
    frame: np.ndarray,
    config: PreprocessConfig = PreprocessConfig(),
) -> np.ndarray:
    """Convert an OpenCV BGR frame to a normalized batched NCHW tensor."""

    rgb = Image.fromarray(frame[:, :, ::-1].copy()).convert("RGB")
    rgb = ImageOps.pad(
        rgb,
        (config.image_size, config.image_size),
        method=Image.Resampling.LANCZOS,
        color=config.fill,
        centering=(0.5, 0.5),
    )
    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]
    )(rgb)
    return tensor.unsqueeze(0).numpy().astype(np.float32)
