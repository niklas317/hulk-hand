#!/usr/bin/env python3
"""Evaluate ResNet18-IN1K on validation or untouched test sessions."""

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
TRAIN_PATH = SCRIPT_DIR / "train.py"
spec = importlib.util.spec_from_file_location("resnet18_in1k_train", TRAIN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load training module: {TRAIN_PATH}")
train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train
spec.loader.exec_module(train)


@torch.inference_mode()
def evaluate(model, loader, device) -> dict:
    model.eval()
    correct = total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    session_correct = defaultdict(int)
    session_total = defaultdict(int)
    confusion = torch.zeros((len(train.CLASS_NAMES), len(train.CLASS_NAMES)), dtype=torch.long)
    for batch in tqdm(loader, desc="Evaluation"):
        logits = model(batch["image"].to(device, non_blocking=True))
        labels = batch["label"].to(device, non_blocking=True)
        predictions = logits.argmax(1)
        correct += int(predictions.eq(labels).sum())
        total += labels.numel()
        for label, prediction, session in zip(labels.cpu().tolist(), predictions.cpu().tolist(), batch["session"]):
            name = train.CLASS_NAMES[label]
            class_total[name] += 1
            class_correct[name] += int(label == prediction)
            session_total[session] += 1
            session_correct[session] += int(label == prediction)
            confusion[label, prediction] += 1
    per_class = {name: class_correct[name] / max(1, class_total[name]) for name in train.CLASS_NAMES}
    result = {
        "accuracy": correct / max(1, total),
        "balanced_accuracy": sum(per_class.values()) / len(train.CLASS_NAMES),
        "per_class_accuracy": per_class,
        "per_session_accuracy": {session: session_correct[session] / max(1, session_total[session]) for session in sorted(session_total)},
        "confusion_matrix": confusion.tolist(),
    }
    print(f"Accuracy: {result['accuracy'] * 100:.2f}%")
    print(f"Balanced accuracy: {result['balanced_accuracy'] * 100:.2f}%")
    for name in train.CLASS_NAMES:
        print(f"{name:<16} {per_class[name] * 100:6.2f}% ({class_correct[name]}/{class_total[name]})")
    for session, value in result["per_session_accuracy"].items():
        print(f"Session {session:<8} {value * 100:6.2f}%")
    print("Confusion matrix (rows=true, columns=predicted)")
    print("                 " + " ".join(f"{name:>12}" for name in train.CLASS_NAMES))
    for index, name in enumerate(train.CLASS_NAMES):
        print(f"{name:>16} " + " ".join(f"{value:12d}" for value in confusion[index].tolist()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ResNet18-IN1K checkpoint.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    expected = {name: index for index, name in enumerate(train.CLASS_NAMES)}
    if checkpoint.get("classes") != train.CLASS_NAMES or checkpoint.get("class_to_idx") != expected:
        raise RuntimeError("Checkpoint class mapping does not match the ResNet18-IN1K pipeline.")
    sessions = train.VAL_SESSIONS if args.split == "validation" else train.TEST_SESSIONS
    dataset = train.collect_samples(data_dir, sessions, train.build_eval_transform(train.PreprocessConfig()))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    model = train.build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Checkpoint: {args.checkpoint}; split={args.split}; samples={len(dataset)}; device={device}")
    evaluate(model, loader, device)


if __name__ == "__main__":
    main()
