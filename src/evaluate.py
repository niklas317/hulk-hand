#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    GestureDataset,
    build_eval_transform,
    collect_new_samples,
    discover_classes,
)
from model import HulkHandDinoV2, build_model


BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: Path,
) -> dict:

    checkpoint_path = (
        checkpoint_path
        .expanduser()
        .resolve()
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Invalid checkpoint format."
        )

    required_keys = {
        "model_state_dict",
        "classes",
        "class_to_idx",
        "num_classes",
    }

    missing = (
        required_keys
        - checkpoint.keys()
    )

    if missing:
        raise RuntimeError(
            "Checkpoint is missing required keys: "
            f"{sorted(missing)}"
        )

    return checkpoint


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_test_dataset(
    data_dir: Path,
    checkpoint: dict,
) -> tuple[
    GestureDataset,
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
            f"Dataset directory not found: {data_dir}"
        )

    classes, class_to_idx = discover_classes(
        data_dir
    )

    checkpoint_classes = checkpoint[
        "classes"
    ]

    checkpoint_class_to_idx = checkpoint[
        "class_to_idx"
    ]

    if classes != checkpoint_classes:
        raise RuntimeError(
            "Dataset classes do not match the "
            "training checkpoint.\n"
            f"Checkpoint: {checkpoint_classes}\n"
            f"Dataset:    {classes}"
        )

    if class_to_idx != checkpoint_class_to_idx:
        raise RuntimeError(
            "Dataset class mapping does not match "
            "the training checkpoint."
        )

    samples = collect_new_samples(
        data_dir=data_dir,
        class_to_idx=class_to_idx,
        split="test",
    )

    if not samples:
        raise RuntimeError(
            "No NEW test samples found in "
            "sessions W-Z."
        )

    transform = build_eval_transform(
        image_size=224
    )

    dataset = GestureDataset(
        samples=samples,
        transform=transform,
    )

    return (
        dataset,
        classes,
        class_to_idx,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(
    model: HulkHandDinoV2,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[
    float,
    list[float],
    torch.Tensor,
]:

    model.eval()

    total_samples = 0
    total_correct = 0

    class_total = torch.zeros(
        num_classes,
        dtype=torch.long,
    )

    class_correct = torch.zeros(
        num_classes,
        dtype=torch.long,
    )

    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
    )

    progress = tqdm(
        loader,
        desc="Final test",
    )

    for batch in progress:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        outputs = model(
            images
        )

        logits = outputs[
            "logits"
        ]

        predictions = logits.argmax(
            dim=1
        )

        batch_size = labels.size(0)

        total_samples += batch_size

        total_correct += (
            predictions
            .eq(labels)
            .sum()
            .item()
        )

        labels_cpu = labels.cpu()
        predictions_cpu = predictions.cpu()

        for true_label, predicted_label in zip(
            labels_cpu,
            predictions_cpu,
        ):

            true_index = int(
                true_label.item()
            )

            predicted_index = int(
                predicted_label.item()
            )

            class_total[
                true_index
            ] += 1

            if (
                true_index
                == predicted_index
            ):
                class_correct[
                    true_index
                ] += 1

            confusion_matrix[
                true_index,
                predicted_index,
            ] += 1

    if total_samples == 0:
        raise RuntimeError(
            "Test dataset is empty."
        )

    overall_accuracy = (
        total_correct
        / total_samples
    )

    class_accuracies: list[float] = []

    for index in range(
        num_classes
    ):

        count = int(
            class_total[index]
        )

        if count == 0:
            accuracy = float("nan")

        else:
            accuracy = (
                int(class_correct[index])
                / count
            )

        class_accuracies.append(
            accuracy
        )

    return (
        overall_accuracy,
        class_accuracies,
        confusion_matrix,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(
    classes: list[str],
    overall_accuracy: float,
    class_accuracies: list[float],
    confusion_matrix: torch.Tensor,
    total_samples: int,
) -> None:

    print()
    print("Final NEW test evaluation")
    print("=========================")
    print()

    print(
        f"Test samples:     {total_samples:,}"
    )

    print(
        f"Overall accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )

    print()
    print("Accuracy per class")
    print("------------------")

    for index, class_name in enumerate(
        classes
    ):

        accuracy = (
            class_accuracies[index]
        )

        if accuracy != accuracy:
            value = "N/A"
        else:
            value = (
                f"{accuracy * 100:.2f}%"
            )

        print(
            f"{index:>2}  "
            f"{class_name:<20} "
            f"{value:>8}"
        )

    print()
    print("Confusion matrix")
    print("----------------")
    print(
        "Rows = true class"
    )
    print(
        "Columns = predicted class"
    )
    print()

    width = max(
        7,
        max(
            len(str(int(value)))
            for value
            in confusion_matrix.flatten()
        ) + 2,
    )

    header = (
        " " * 8
        + "".join(
            f"{index:>{width}}"
            for index
            in range(len(classes))
        )
    )

    print(header)

    for row_index, row in enumerate(
        confusion_matrix
    ):

        values = "".join(
            f"{int(value):>{width}}"
            for value in row
        )

        print(
            f"{row_index:>6}  "
            f"{values}"
        )

    print()
    print("Class mapping")
    print("-------------")

    for index, class_name in enumerate(
        classes
    ):
        print(
            f"{index}: {class_name}"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Final evaluation of the best "
            "hulk-hand checkpoint."
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
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to best.pt."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help=(
            "DataLoader worker processes. "
            "Default: 4"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = load_checkpoint(
        args.checkpoint
    )

    test_dataset, classes, class_to_idx = (
        build_test_dataset(
            data_dir=args.data_dir,
            checkpoint=checkpoint,
        )
    )

    model = build_model(
        num_classes=len(classes),
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=False,
    )

    print()
    print(
        f"Device:     {device}"
    )

    if device.type == "cuda":
        print(
            "GPU:        "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Checkpoint: {args.checkpoint}"
    )

    print(
        "Test split:  NEW sessions W-Z"
    )

    print()

    (
        overall_accuracy,
        class_accuracies,
        confusion_matrix,
    ) = evaluate(
        model=model,
        loader=loader,
        device=device,
        num_classes=len(classes),
    )

    print_results(
        classes=classes,
        overall_accuracy=overall_accuracy,
        class_accuracies=class_accuracies,
        confusion_matrix=confusion_matrix,
        total_samples=len(test_dataset),
    )


if __name__ == "__main__":
    main()
