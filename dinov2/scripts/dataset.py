#!/usr/bin/env python3

"""NEW-only dataset loading for the current DINOv2 pipeline."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_SESSIONS = tuple(chr(c) for c in range(ord("A"), ord("R") + 1))
VAL_SESSIONS = tuple(chr(c) for c in range(ord("S"), ord("V") + 1))
TEST_SESSIONS = tuple(chr(c) for c in range(ord("W"), ord("Z") + 1))
SESSION_SPLITS = {
    "train": TRAIN_SESSIONS,
    "val": VAL_SESSIONS,
    "test": TEST_SESSIONS,
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    class_name: str
    session: str


class AddGaussianNoise:
    def __init__(self, probability: float = 0.15, std: float = 0.01) -> None:
        self.probability = probability
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


def build_train_transform(image_size: int = 224, horizontal_flip: bool = False) -> Callable:
    stages = [
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.12, 0.12),
            scale=(0.70, 1.10),
        ),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.25),
        transforms.ColorJitter(
            brightness=0.30,
            contrast=0.30,
            saturation=0.20,
            hue=0.04,
        ),
    ]
    if horizontal_flip:
        stages.append(transforms.RandomHorizontalFlip(p=0.5))
    stages.extend(
        [
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=0.15,
            ),
            transforms.ToTensor(),
            AddGaussianNoise(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(stages)


def build_eval_transform(image_size: int = 224) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def discover_classes(data_dir: Path) -> tuple[list[str], dict[str, int]]:
    classes = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    if not classes:
        raise RuntimeError(f"No class directories found in: {data_dir}")
    return classes, {name: index for index, name in enumerate(classes)}


def collect_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_new_samples(
    data_dir: Path,
    class_to_idx: dict[str, int],
    split: str,
) -> list[Sample]:
    if split not in SESSION_SPLITS:
        raise ValueError(f"Unknown split: {split}")

    samples: list[Sample] = []
    for class_name, label in class_to_idx.items():
        for session in SESSION_SPLITS[split]:
            session_dir = data_dir / class_name / f"session_{session}"
            for path in collect_images(session_dir):
                samples.append(Sample(path, label, class_name, session))
    return samples


class GestureDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: Callable) -> None:
        if not samples:
            raise RuntimeError("Dataset contains no samples.")
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = self.transform(image.convert("RGB"))
        return {
            "image": image,
            "label": sample.label,
            "class_name": sample.class_name,
            "source": "new",
            "session": sample.session,
            "path": str(sample.path),
        }


def build_new_only_datasets(
    data_dir: Path,
    image_size: int = 224,
    horizontal_flip: bool = False,
) -> tuple[dict[str, GestureDataset], list[str], dict[str, int]]:
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    classes, class_to_idx = discover_classes(data_dir)
    train_transform = build_train_transform(image_size, horizontal_flip)
    eval_transform = build_eval_transform(image_size)
    datasets = {
        "new_train": GestureDataset(
            collect_new_samples(data_dir, class_to_idx, "train"), train_transform
        ),
        "new_val": GestureDataset(
            collect_new_samples(data_dir, class_to_idx, "val"), eval_transform
        ),
        "new_test": GestureDataset(
            collect_new_samples(data_dir, class_to_idx, "test"), eval_transform
        ),
    }
    return datasets, classes, class_to_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the NEW-only DINOv2 dataset.")
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    classes, mapping = discover_classes(data_dir)
    print("Class mapping")
    for name, index in mapping.items():
        print(f"{index}: {name}")
    print("\nNEW dataset summary")
    for label, split in (("NEW train", "train"), ("NEW val", "val"), ("NEW test", "test")):
        samples = collect_new_samples(data_dir, mapping, split)
        print(f"{label:<12} {len(samples):>7}")
        for class_name in classes:
            count = sum(sample.class_name == class_name for sample in samples)
            print(f"    {class_name:<20} {count:>7}")


if __name__ == "__main__":
    main()
