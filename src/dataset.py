#!/usr/bin/env python3

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

TRAIN_SESSIONS = tuple(
    chr(c) for c in range(ord("A"), ord("R") + 1)
)

VAL_SESSIONS = tuple(
    chr(c) for c in range(ord("S"), ord("V") + 1)
)

TEST_SESSIONS = tuple(
    chr(c) for c in range(ord("W"), ord("Z") + 1)
)

SESSION_SPLITS = {
    "train": TRAIN_SESSIONS,
    "val": VAL_SESSIONS,
    "test": TEST_SESSIONS,
}

# HaGRID / ImageNet normalization.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Sample metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    class_name: str
    source: str
    session: str | None = None


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class AddGaussianNoise:
    def __init__(
        self,
        probability: float = 0.15,
        std: float = 0.01,
    ) -> None:
        self.probability = probability
        self.std = std

    def __call__(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:

        if random.random() >= self.probability:
            return tensor

        noise = torch.randn_like(tensor) * self.std

        return torch.clamp(
            tensor + noise,
            min=0.0,
            max=1.0,
        )


def build_train_transform(
    image_size: int = 224,
    horizontal_flip: bool = False,
) -> Callable:

    transform_list = [
        transforms.Resize(
            (image_size, image_size),
            antialias=True,
        ),

        transforms.RandomAffine(
            degrees=15,
            translate=(0.12, 0.12),
            scale=(0.70, 1.10),
        ),

        transforms.RandomPerspective(
            distortion_scale=0.15,
            p=0.25,
        ),

        transforms.ColorJitter(
            brightness=0.30,
            contrast=0.30,
            saturation=0.20,
            hue=0.04,
        ),
    ]

    if horizontal_flip:
        transform_list.append(
            transforms.RandomHorizontalFlip(
                p=0.5
            )
        )

    transform_list.extend(
        [
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3,
                        sigma=(0.1, 1.0),
                    )
                ],
                p=0.15,
            ),

            transforms.ToTensor(),

            AddGaussianNoise(
                probability=0.15,
                std=0.01,
            ),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    return transforms.Compose(
        transform_list
    )


def build_eval_transform(
    image_size: int = 224,
) -> Callable:

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                antialias=True,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_classes(
    data_dir: Path,
) -> tuple[list[str], dict[str, int]]:

    classes = sorted(
        path.name
        for path in data_dir.iterdir()
        if path.is_dir()
    )

    if not classes:
        raise RuntimeError(
            f"No class directories found in: {data_dir}"
        )

    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(classes)
    }

    return classes, class_to_idx


def collect_images(
    directory: Path,
) -> list[Path]:

    if not directory.is_dir():
        return []

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# OLD HaGRID data
# ---------------------------------------------------------------------------

def collect_old_samples(
    data_dir: Path,
    class_to_idx: dict[str, int],
) -> list[Sample]:

    samples: list[Sample] = []

    for class_name, label in class_to_idx.items():

        old_dir = (
            data_dir
            / class_name
            / "old"
        )

        for path in collect_images(old_dir):

            samples.append(
                Sample(
                    path=path,
                    label=label,
                    class_name=class_name,
                    source="old",
                )
            )

    return samples


def split_old_samples(
    samples: list[Sample],
    val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """
    Deterministic stratified OLD split.

    Each class is independently split into:
        90% training
        10% validation
    """

    if not samples:
        raise RuntimeError(
            "No OLD samples found."
        )

    samples_by_class: dict[int, list[Sample]] = {}

    for sample in samples:
        samples_by_class.setdefault(
            sample.label,
            [],
        ).append(sample)

    rng = random.Random(seed)

    train_samples: list[Sample] = []
    val_samples: list[Sample] = []

    for label in sorted(samples_by_class):

        class_samples = list(
            samples_by_class[label]
        )

        rng.shuffle(class_samples)

        if len(class_samples) < 2:
            raise RuntimeError(
                "Each OLD class must contain at least "
                f"2 samples. Label {label} contains "
                f"{len(class_samples)}."
            )

        val_count = max(
            1,
            round(
                len(class_samples)
                * val_fraction
            ),
        )

        # Never move the entire class into validation.
        val_count = min(
            val_count,
            len(class_samples) - 1,
        )

        val_samples.extend(
            class_samples[:val_count]
        )

        train_samples.extend(
            class_samples[val_count:]
        )

    return train_samples, val_samples


# ---------------------------------------------------------------------------
# NEW session-based data
# ---------------------------------------------------------------------------

def collect_new_samples(
    data_dir: Path,
    class_to_idx: dict[str, int],
    split: str,
) -> list[Sample]:

    if split not in SESSION_SPLITS:
        raise ValueError(
            f"Unknown split: {split}"
        )

    allowed_sessions = SESSION_SPLITS[split]

    samples: list[Sample] = []

    for class_name, label in class_to_idx.items():

        class_dir = (
            data_dir
            / class_name
        )

        for session in allowed_sessions:

            session_dir = (
                class_dir
                / f"session_{session}"
            )

            for path in collect_images(
                session_dir
            ):

                samples.append(
                    Sample(
                        path=path,
                        label=label,
                        class_name=class_name,
                        source="new",
                        session=session,
                    )
                )

    return samples


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class GestureDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        transform: Callable,
    ) -> None:

        if not samples:
            raise RuntimeError(
                "Dataset contains no samples."
            )

        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict:

        sample = self.samples[index]

        with Image.open(sample.path) as image:

            image = image.convert("RGB")

            image = self.transform(
                image
            )

        return {
            "image": image,
            "label": sample.label,
            "class_name": sample.class_name,
            "source": sample.source,
            "session": sample.session or "",
            "path": str(sample.path),
        }


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_datasets(
    data_dir: Path,
    image_size: int = 224,
    horizontal_flip: bool = False,
    old_val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[
    dict[str, GestureDataset],
    list[str],
    dict[str, int],
]:

    data_dir = (
        data_dir
        .expanduser()
        .resolve()
    )

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{data_dir}"
        )

    classes, class_to_idx = (
        discover_classes(
            data_dir
        )
    )

    train_transform = (
        build_train_transform(
            image_size=image_size,
            horizontal_flip=horizontal_flip,
        )
    )

    eval_transform = (
        build_eval_transform(
            image_size=image_size,
        )
    )

    # OLD
    old_samples = collect_old_samples(
        data_dir,
        class_to_idx,
    )

    old_train_samples, old_val_samples = (
        split_old_samples(
            old_samples,
            val_fraction=old_val_fraction,
            seed=seed,
        )
    )

    # NEW
    new_train_samples = collect_new_samples(
        data_dir,
        class_to_idx,
        split="train",
    )

    new_val_samples = collect_new_samples(
        data_dir,
        class_to_idx,
        split="val",
    )

    new_test_samples = collect_new_samples(
        data_dir,
        class_to_idx,
        split="test",
    )

    datasets = {
        "old_train": GestureDataset(
            old_train_samples,
            transform=train_transform,
        ),

        "old_val": GestureDataset(
            old_val_samples,
            transform=eval_transform,
        ),

        "new_train": GestureDataset(
            new_train_samples,
            transform=train_transform,
        ),

        "new_val": GestureDataset(
            new_val_samples,
            transform=eval_transform,
        ),

        "new_test": GestureDataset(
            new_test_samples,
            transform=eval_transform,
        ),
    }

    return (
        datasets,
        classes,
        class_to_idx,
    )


# ---------------------------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------------------------

def print_sample_counts(
    name: str,
    samples: list[Sample],
    classes: list[str],
) -> None:

    print(
        f"{name:<12} {len(samples):>7}"
    )

    for class_name in classes:

        count = sum(
            sample.class_name == class_name
            for sample in samples
        )

        print(
            f"    {class_name:<20} "
            f"{count:>7}"
        )


def print_dataset_summary(
    data_dir: Path,
    old_val_fraction: float = 0.10,
    seed: int = 42,
) -> None:

    data_dir = (
        data_dir
        .expanduser()
        .resolve()
    )

    classes, class_to_idx = (
        discover_classes(
            data_dir
        )
    )

    old_samples = collect_old_samples(
        data_dir,
        class_to_idx,
    )

    old_train, old_val = (
        split_old_samples(
            old_samples,
            val_fraction=old_val_fraction,
            seed=seed,
        )
    )

    new_train = collect_new_samples(
        data_dir,
        class_to_idx,
        "train",
    )

    new_val = collect_new_samples(
        data_dir,
        class_to_idx,
        "val",
    )

    new_test = collect_new_samples(
        data_dir,
        class_to_idx,
        "test",
    )

    print()
    print("Class mapping")
    print("-------------")

    for class_name, index in (
        class_to_idx.items()
    ):
        print(
            f"{index}: {class_name}"
        )

    print()
    print("Dataset summary")
    print("---------------")

    print_sample_counts(
        "OLD train",
        old_train,
        classes,
    )

    print_sample_counts(
        "OLD val",
        old_val,
        classes,
    )

    print_sample_counts(
        "NEW train",
        new_train,
        classes,
    )

    print_sample_counts(
        "NEW val",
        new_val,
        classes,
    )

    print_sample_counts(
        "NEW test",
        new_test,
        classes,
    )

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Inspect the hulk-hand dataset."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Root directory of the processed dataset."
        ),
    )

    parser.add_argument(
        "--old-val-fraction",
        type=float,
        default=0.10,
        help=(
            "Fraction of OLD samples used for "
            "validation. Default: 0.10"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for the OLD split. "
            "Default: 42"
        ),
    )

    args = parser.parse_args()

    print_dataset_summary(
        data_dir=args.data_dir,
        old_val_fraction=args.old_val_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()