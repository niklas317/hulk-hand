#!/usr/bin/env python3
"""Diagnose a FractalDB ResNet34 checkpoint on private session splits."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
TRAINER_PATH = SCRIPT_DIR / "train-resnet34-fractaldb.py"

spec = importlib.util.spec_from_file_location("resnet34_fractaldb_trainer", TRAINER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load trainer module: {TRAINER_PATH}")
trainer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trainer
spec.loader.exec_module(trainer)


SPLITS = {
    "validation": trainer.VAL_SESSIONS,
    "test": trainer.TEST_SESSIONS,
}


def load_checkpoint(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("Invalid checkpoint: expected a training checkpoint with model_state_dict.")
    if checkpoint.get("classes") != trainer.CLASS_NAMES:
        raise RuntimeError(
            "Checkpoint classes do not match the FractalDB pipeline: "
            f"{checkpoint.get('classes')} != {trainer.CLASS_NAMES}"
        )
    return checkpoint


def print_accuracy(label: str, correct: int, total: int) -> None:
    accuracy = correct / total if total else 0.0
    print(f"{label:<20} {accuracy * 100:6.2f}% ({correct}/{total})")


@torch.inference_mode()
def diagnose(model, loader, device: torch.device) -> None:
    model.eval()
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    session_correct = defaultdict(int)
    session_total = defaultdict(int)
    confusion = torch.zeros((len(trainer.CLASS_NAMES), len(trainer.CLASS_NAMES)), dtype=torch.long)
    total_correct = 0
    total = 0

    for batch in tqdm(loader, desc="Diagnosing", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits, _ = model(images)
        predictions = logits.argmax(dim=1)

        for label, prediction, session in zip(
            labels.cpu().tolist(), predictions.cpu().tolist(), batch["session"]
        ):
            class_name = trainer.CLASS_NAMES[label]
            confusion[label, prediction] += 1
            class_total[class_name] += 1
            class_correct[class_name] += int(label == prediction)
            session_total[session] += 1
            session_correct[session] += int(label == prediction)

        total_correct += int(predictions.eq(labels).sum())
        total += labels.numel()

    print_accuracy("Overall", total_correct, total)
    print("\nPer-class accuracy")
    for class_name in trainer.CLASS_NAMES:
        print_accuracy(class_name, class_correct[class_name], class_total[class_name])
    print("\nPer-session accuracy")
    for session in sorted(session_total):
        print_accuracy(f"Session {session}", session_correct[session], session_total[session])
    print("\nConfusion matrix (rows=true, columns=predicted)")
    print("                " + " ".join(f"{name:>12}" for name in trainer.CLASS_NAMES))
    for index, class_name in enumerate(trainer.CLASS_NAMES):
        values = " ".join(f"{int(value):12d}" for value in confusion[index])
        print(f"{class_name:>16} {values}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a FractalDB ResNet34 on validation or held-out test sessions."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(SPLITS), default="validation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    checkpoint = load_checkpoint(args.checkpoint)
    dataset = trainer.collect_samples(
        data_dir,
        SPLITS[args.split],
        trainer.build_eval_transform(trainer.PreprocessConfig(image_size=224)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = trainer.ResNet34WithEmbedding(len(trainer.CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Checkpoint: {args.checkpoint.expanduser().resolve()}")
    print(f"Split: {args.split} ({', '.join(SPLITS[args.split])}); samples={len(dataset)}; device={device}\n")
    diagnose(model, loader, device)


if __name__ == "__main__":
    main()
