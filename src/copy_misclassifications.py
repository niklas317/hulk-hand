#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import (
    GestureDataset,
    build_eval_transform,
    collect_new_samples,
    discover_classes,
)
from diagnose import load_checkpoint, validate_checkpoint
from model import build_model


TARGET_SUBDIR = "misclassifications"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy NEW validation misclassifications into a mirrored "
            "misclassifications/ tree."
        )
    )

    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the trained checkpoint.",
    )

    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the processed Dataset.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory that will receive misclassifications/.",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError(
            "Output directory must not be inside the dataset directory."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    target_root = output_dir / TARGET_SUBDIR

    if target_root.exists():
        shutil.rmtree(target_root)

    checkpoint = load_checkpoint(checkpoint_path)
    classes, class_to_idx = discover_classes(data_dir)
    validate_checkpoint(checkpoint, classes, class_to_idx)

    samples = collect_new_samples(
        data_dir=data_dir,
        class_to_idx=class_to_idx,
        split="val",
    )

    if not samples:
        raise RuntimeError("NEW validation split is empty.")

    dataset = GestureDataset(
        samples=samples,
        transform=build_eval_transform(image_size=224),
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_model(
        num_classes=len(classes),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    manifest_path = output_dir / "misclassifications.jsonl"
    copied = 0
    total = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            logits = model(images)["logits"]
            predictions = logits.argmax(dim=1).cpu()

            for index, prediction in enumerate(predictions.tolist()):
                total += 1

                true_label = int(labels[index])

                if prediction == true_label:
                    continue

                source_path = Path(batch["path"][index])
                relative_path = source_path.relative_to(data_dir)
                destination_path = target_root / relative_path
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)

                record = {
                    "source_path": str(source_path),
                    "destination_path": str(destination_path),
                    "true_class": batch["class_name"][index],
                    "predicted_class": classes[prediction],
                    "session": batch["session"][index],
                }
                manifest_file.write(json.dumps(record) + "\n")
                copied += 1

    print(f"Evaluated samples: {total}")
    print(f"Misclassifications copied: {copied}")
    print(f"Output directory: {target_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
