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
from model import HulkHandResNet18


VALIDATION_SESSIONS = (
    "S",
    "T",
    "U",
    "V",
)


def load_checkpoint(
    path: Path,
) -> dict:

    path = (
        path
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Invalid checkpoint format."
        )

    required = {
        "model_state_dict",
        "classes",
        "class_to_idx",
        "num_classes",
    }

    missing = (
        required
        - set(checkpoint.keys())
    )

    if missing:
        raise RuntimeError(
            "Checkpoint is missing keys: "
            + ", ".join(
                sorted(missing)
            )
        )

    return checkpoint


def build_loader(
    samples,
    transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:

    dataset = GestureDataset(
        samples=samples,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )


def evaluate_split(
    model: HulkHandResNet18,
    loader: DataLoader,
    device: torch.device,
    classes: list[str],
    description: str,
) -> tuple[
    float,
    list[float],
    torch.Tensor,
    list[int],
]:

    num_classes = len(classes)

    confusion = torch.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=torch.int64,
    )

    class_correct = torch.zeros(
        num_classes,
        dtype=torch.int64,
    )

    class_total = torch.zeros(
        num_classes,
        dtype=torch.int64,
    )

    total_correct = 0
    total_samples = 0

    model.eval()

    with torch.inference_mode():

        for batch in tqdm(
            loader,
            desc=description,
            leave=False,
        ):

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

            if not torch.isfinite(
                logits
            ).all():
                raise FloatingPointError(
                    "Non-finite logits detected "
                    f"in {description}."
                )

            predictions = logits.argmax(
                dim=1
            )

            total_correct += (
                predictions
                .eq(labels)
                .sum()
                .item()
            )

            total_samples += (
                labels.size(0)
            )

            labels_cpu = (
                labels.cpu()
            )

            predictions_cpu = (
                predictions.cpu()
            )

            for target, prediction in zip(
                labels_cpu.tolist(),
                predictions_cpu.tolist(),
            ):

                confusion[
                    target,
                    prediction,
                ] += 1

                class_total[
                    target
                ] += 1

                if target == prediction:
                    class_correct[
                        target
                    ] += 1

    if total_samples == 0:
        raise RuntimeError(
            f"{description} dataset is empty."
        )

    overall_accuracy = (
        total_correct
        / total_samples
    )

    per_class_accuracy = []

    for index in range(
        num_classes
    ):

        if class_total[index] == 0:
            accuracy = float("nan")

        else:
            accuracy = (
                class_correct[
                    index
                ].item()
                / class_total[
                    index
                ].item()
            )

        per_class_accuracy.append(
            accuracy
        )

    return (
        overall_accuracy,
        per_class_accuracy,
        confusion,
        class_total.tolist(),
    )


def print_confusion_matrix(
    confusion: torch.Tensor,
    classes: list[str],
) -> None:

    width = max(
        12,
        max(
            len(name)
            for name in classes
        ) + 2,
    )

    print()
    print("Confusion matrix")
    print("----------------")
    print("Rows = true class")
    print("Columns = predicted class")
    print()

    print(
        " " * width,
        end="",
    )

    for class_name in classes:

        print(
            f"{class_name:>{width}}",
            end="",
        )

    print()

    for row_index, class_name in enumerate(
        classes
    ):

        print(
            f"{class_name:<{width}}",
            end="",
        )

        for column_index in range(
            len(classes)
        ):

            value = int(
                confusion[
                    row_index,
                    column_index,
                ].item()
            )

            print(
                f"{value:>{width}d}",
                end="",
            )

        print()


def print_results(
    title: str,
    overall_accuracy: float,
    per_class_accuracy: list[float],
    confusion: torch.Tensor,
    classes: list[str],
) -> None:

    print()
    print(title)
    print("=" * len(title))
    print()

    print(
        f"Overall accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )

    print()
    print("Per-class accuracy")
    print("------------------")

    for class_name, accuracy in zip(
        classes,
        per_class_accuracy,
    ):

        if accuracy != accuracy:
            text = "N/A"

        else:
            text = (
                f"{accuracy * 100:.2f}%"
            )

        print(
            f"{class_name:<20} "
            f"{text}"
        )

    print_confusion_matrix(
        confusion,
        classes,
    )


def print_session_results(
    session_results: dict,
    classes: list[str],
) -> None:

    print()
    print("VALIDATION BY SESSION")
    print("=====================")
    print()

    header = (
        f"{'Session':<10}"
        f"{'Overall':>12}"
    )

    for class_name in classes:
        header += (
            f"{class_name:>16}"
        )

    print(header)
    print(
        "-" * len(header)
    )

    for session in VALIDATION_SESSIONS:

        result = session_results[
            session
        ]

        line = (
            f"{session:<10}"
            f"{result['overall'] * 100:>11.2f}%"
        )

        for accuracy in result[
            "per_class"
        ]:

            if accuracy != accuracy:
                text = "N/A"
            else:
                text = (
                    f"{accuracy * 100:.2f}%"
                )

            line += (
                f"{text:>16}"
            )

        print(line)

    print()


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Diagnose NEW train versus NEW validation "
            "performance using best.pt, including "
            "per-session validation results."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Path to the processed Dataset."
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
        "--batch-size",
        type=int,
        default=128,
        help=(
            "Evaluation batch size. Default: 128"
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help=(
            "DataLoader workers. Default: 4"
        ),
    )

    args = parser.parse_args()

    data_dir = (
        args.data_dir
        .expanduser()
        .resolve()
    )

    checkpoint_path = (
        args.checkpoint
        .expanduser()
        .resolve()
    )

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset not found: {data_dir}"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0."
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers must be >= 0."
        )

    # ----------------------------------------------------------
    # Classes
    # ----------------------------------------------------------

    classes, class_to_idx = (
        discover_classes(
            data_dir
        )
    )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    saved_classes = checkpoint[
        "classes"
    ]

    saved_class_to_idx = checkpoint[
        "class_to_idx"
    ]

    if saved_classes != classes:
        raise RuntimeError(
            "Checkpoint classes do not match dataset.\n"
            f"Checkpoint: {saved_classes}\n"
            f"Dataset:    {classes}"
        )

    if (
        saved_class_to_idx
        != class_to_idx
    ):
        raise RuntimeError(
            "Checkpoint class mapping does not "
            "match dataset class mapping."
        )

    num_classes = len(
        classes
    )

    # ----------------------------------------------------------
    # Deterministic evaluation preprocessing
    #
    # IMPORTANT:
    # No training augmentation is used here.
    # ----------------------------------------------------------

    eval_transform = (
        build_eval_transform(
            image_size=224
        )
    )

    # ----------------------------------------------------------
    # Collect NEW samples
    # ----------------------------------------------------------

    new_train_samples = (
        collect_new_samples(
            data_dir=data_dir,
            class_to_idx=class_to_idx,
            split="train",
        )
    )

    new_val_samples = (
        collect_new_samples(
            data_dir=data_dir,
            class_to_idx=class_to_idx,
            split="val",
        )
    )

    # ----------------------------------------------------------
    # Check validation session distribution
    # ----------------------------------------------------------

    validation_session_counts = {
        session: 0
        for session in VALIDATION_SESSIONS
    }

    for sample in new_val_samples:

        session = (
            str(sample.session)
            .replace("session_", "")
            .upper()
        )

        if (
            session
            in validation_session_counts
        ):
            validation_session_counts[
                session
            ] += 1

    for session in VALIDATION_SESSIONS:

        if (
            validation_session_counts[
                session
            ] == 0
        ):
            raise RuntimeError(
                "No samples found for validation "
                f"session {session}."
            )

    # ----------------------------------------------------------
    # Loaders
    # ----------------------------------------------------------

    pin_memory = (
        torch.cuda.is_available()
    )

    new_train_loader = (
        build_loader(
            samples=new_train_samples,
            transform=eval_transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    )

    new_val_loader = (
        build_loader(
            samples=new_val_samples,
            transform=eval_transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    )

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = HulkHandResNet18(
        num_classes=num_classes
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    model.eval()

    # ----------------------------------------------------------
    # Header
    # ----------------------------------------------------------

    print()
    print("hulk-hand diagnosis")
    print("===================")
    print()

    print(
        f"Checkpoint:       "
        f"{checkpoint_path}"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    metrics = checkpoint.get(
        "metrics",
        {},
    )

    if (
        "new_val_accuracy"
        in metrics
    ):
        print(
            "Saved NEW val:   "
            f"{metrics['new_val_accuracy'] * 100:.2f}%"
        )

    print(
        f"Device:           {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU:              "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        "Precision:        FP32"
    )

    print(
        f"NEW train:        "
        f"{len(new_train_samples)}"
    )

    print(
        f"NEW val:          "
        f"{len(new_val_samples)}"
    )

    print()

    for session in VALIDATION_SESSIONS:

        print(
            f"Session {session}:        "
            f"{validation_session_counts[session]}"
        )

    # ----------------------------------------------------------
    # NEW train
    # ----------------------------------------------------------

    (
        train_accuracy,
        train_class_accuracy,
        train_confusion,
        _,
    ) = evaluate_split(
        model=model,
        loader=new_train_loader,
        device=device,
        classes=classes,
        description="NEW train",
    )

    # ----------------------------------------------------------
    # NEW validation
    # ----------------------------------------------------------

    (
        val_accuracy,
        val_class_accuracy,
        val_confusion,
        _,
    ) = evaluate_split(
        model=model,
        loader=new_val_loader,
        device=device,
        classes=classes,
        description="NEW val",
    )

    # ----------------------------------------------------------
    # Per-session validation
    # ----------------------------------------------------------

    session_results = {}

    for session in VALIDATION_SESSIONS:

        session_samples = []

        for sample in new_val_samples:

            sample_session = (
                str(sample.session)
                .replace("session_", "")
                .upper()
            )

            if (
                sample_session
                == session
            ):
                session_samples.append(
                    sample
                )

        session_loader = (
            build_loader(
                samples=session_samples,
                transform=eval_transform,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
        )

        (
            session_accuracy,
            session_class_accuracy,
            session_confusion,
            session_class_counts,
        ) = evaluate_split(
            model=model,
            loader=session_loader,
            device=device,
            classes=classes,
            description=(
                f"Session {session}"
            ),
        )

        session_results[
            session
        ] = {
            "overall": (
                session_accuracy
            ),
            "per_class": (
                session_class_accuracy
            ),
            "confusion": (
                session_confusion
            ),
            "class_counts": (
                session_class_counts
            ),
            "samples": (
                len(session_samples)
            ),
        }

    # ----------------------------------------------------------
    # Results
    # ----------------------------------------------------------

    print_results(
        title=(
            "NEW TRAIN (A-R, no augmentation)"
        ),
        overall_accuracy=train_accuracy,
        per_class_accuracy=(
            train_class_accuracy
        ),
        confusion=train_confusion,
        classes=classes,
    )

    print_results(
        title=(
            "NEW VALIDATION (S-V)"
        ),
        overall_accuracy=val_accuracy,
        per_class_accuracy=(
            val_class_accuracy
        ),
        confusion=val_confusion,
        classes=classes,
    )

    # ----------------------------------------------------------
    # Session overview
    # ----------------------------------------------------------

    print_session_results(
        session_results,
        classes,
    )

    # ----------------------------------------------------------
    # Detailed session confusion matrices
    # ----------------------------------------------------------

    for session in VALIDATION_SESSIONS:

        result = session_results[
            session
        ]

        title = (
            f"SESSION {session}"
        )

        print_results(
            title=title,
            overall_accuracy=(
                result["overall"]
            ),
            per_class_accuracy=(
                result["per_class"]
            ),
            confusion=(
                result["confusion"]
            ),
            classes=classes,
        )

    # ----------------------------------------------------------
    # Train / validation comparison
    # ----------------------------------------------------------

    gap = (
        train_accuracy
        - val_accuracy
    )

    print()
    print(
        "Train / validation comparison"
    )
    print(
        "-----------------------------"
    )

    print(
        f"NEW train accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"NEW val accuracy:   "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Generalization gap: "
        f"{gap * 100:.2f} percentage points"
    )

    print()


if __name__ == "__main__":
    main()