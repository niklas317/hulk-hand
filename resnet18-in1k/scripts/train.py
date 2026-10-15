#!/usr/bin/env python3
"""Train an ImageNet-1K-pretrained ResNet18 on private NEW sessions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_ROOT / "preprocessing"))
from in1k_preprocessing import PreprocessConfig, build_eval_transform, build_train_transform


CLASS_NAMES = ["gesture_1", "gesture_2", "gesture_3", "no_gesture"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_SESSIONS = tuple(chr(code) for code in range(ord("A"), ord("R") + 1))
VAL_SESSIONS = tuple(chr(code) for code in range(ord("S"), ord("V") + 1))
TEST_SESSIONS = tuple(chr(code) for code in range(ord("W"), ord("Z") + 1))

BATCH_SIZE = 32
BATCHES_PER_EPOCH = 281
MAX_EPOCHS = 30
WARMUP_EPOCHS = 3
WARMUP_START_FACTOR = 0.10
STEM_LR = 1e-5
LAYER1_LR = 1e-5
LAYER2_LR = 2e-5
LAYER3_LR = 3e-5
LAYER4_LR = 5e-5
CLASSIFIER_LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
GRADIENT_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 7
BACKBONE_START_EPOCH = 3


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    class_name: str
    session: str


class SessionDataset(Dataset):
    def __init__(self, samples: list[Sample], transform) -> None:
        if not samples:
            raise RuntimeError("Dataset contains no images.")
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "label": sample.label, "session": sample.session, "path": str(sample.path)}


class BalancedBatchSampler(Sampler[list[int]]):
    """Yield four equally represented classes while cycling through sessions."""

    def __init__(self, dataset: SessionDataset, seed: int) -> None:
        if BATCH_SIZE % len(CLASS_NAMES) != 0:
            raise ValueError("BATCH_SIZE must be divisible by the number of classes.")
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.grouped = {label: [] for label in range(len(CLASS_NAMES))}
        for index, sample in enumerate(dataset.samples):
            self.grouped[sample.label].append(index)
        missing = [CLASS_NAMES[label] for label, items in self.grouped.items() if not items]
        if missing:
            raise RuntimeError("Missing training classes: " + ", ".join(missing))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        per_class = BATCH_SIZE // len(CLASS_NAMES)
        sequences = {}
        required = BATCHES_PER_EPOCH * per_class
        for label, source in self.grouped.items():
            by_session: dict[str, list[int]] = defaultdict(list)
            for index in source:
                by_session[self.dataset.samples[index].session].append(index)
            sessions = list(by_session)
            for pool in by_session.values():
                rng.shuffle(pool)
            cursors = {session: 0 for session in sessions}
            sequence = []
            while len(sequence) < required:
                rng.shuffle(sessions)
                for session in sessions:
                    pool = by_session[session]
                    cursor = cursors[session]
                    if cursor >= len(pool):
                        rng.shuffle(pool)
                        cursor = 0
                    sequence.append(pool[cursor])
                    cursors[session] = cursor + 1
                    if len(sequence) == required:
                        break
            sequences[label] = sequence
        for batch_index in range(BATCHES_PER_EPOCH):
            batch = []
            start = batch_index * per_class
            for label in sorted(sequences):
                batch.extend(sequences[label][start : start + per_class])
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return BATCHES_PER_EPOCH


def build_model(pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def collect_samples(data_dir: Path, sessions: tuple[str, ...], transform) -> SessionDataset:
    samples = []
    for label, class_name in enumerate(CLASS_NAMES):
        for session in sessions:
            directory = data_dir / class_name / f"session_{session}"
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append(Sample(path, label, class_name, session))
    return SessionDataset(samples, transform)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_training_stage(model: nn.Module, epoch: int) -> str:
    backbone_trainable = epoch >= BACKBONE_START_EPOCH
    for module in (model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4):
        for parameter in module.parameters():
            parameter.requires_grad = backbone_trainable
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return "backbone+fc" if backbone_trainable else "fc-only"


def set_module_modes(model: nn.Module) -> None:
    model.train()
    frozen = (model.bn1, model.layer1, model.layer2, model.layer3, model.layer4)
    for module in frozen:
        if not any(parameter.requires_grad for parameter in module.parameters()):
            module.eval()


def build_optimizer(model: nn.Module) -> AdamW:
    return AdamW(
        [
            {"name": "stem", "params": [*model.conv1.parameters(), *model.bn1.parameters()], "lr": STEM_LR, "active_epoch": BACKBONE_START_EPOCH},
            {"name": "layer1", "params": model.layer1.parameters(), "lr": LAYER1_LR, "active_epoch": BACKBONE_START_EPOCH},
            {"name": "layer2", "params": model.layer2.parameters(), "lr": LAYER2_LR, "active_epoch": BACKBONE_START_EPOCH},
            {"name": "layer3", "params": model.layer3.parameters(), "lr": LAYER3_LR, "active_epoch": BACKBONE_START_EPOCH},
            {"name": "layer4", "params": model.layer4.parameters(), "lr": LAYER4_LR, "active_epoch": BACKBONE_START_EPOCH},
            {"name": "fc", "params": model.fc.parameters(), "lr": CLASSIFIER_LR, "active_epoch": 1},
        ],
        weight_decay=WEIGHT_DECAY,
    )


class ProgressiveLRScheduler:
    def __init__(self, optimizer: AdamW) -> None:
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.active_epochs = [int(group["active_epoch"]) for group in optimizer.param_groups]
        self.last_epoch = 0

    def step(self, epoch: int) -> None:
        self.last_epoch = epoch
        if epoch <= WARMUP_EPOCHS:
            progress = (epoch - 1) / max(1, WARMUP_EPOCHS - 1)
            factor = WARMUP_START_FACTOR + (1.0 - WARMUP_START_FACTOR) * progress
        else:
            progress = (epoch - WARMUP_EPOCHS) / max(1, MAX_EPOCHS - WARMUP_EPOCHS)
            factor = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        for group, base_lr, active_epoch in zip(self.optimizer.param_groups, self.base_lrs, self.active_epochs):
            group["lr"] = 0.0 if epoch < active_epoch else base_lr * factor

    def state_dict(self) -> dict:
        return {"last_epoch": self.last_epoch, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict) -> None:
        self.last_epoch = int(state.get("last_epoch", 0))


def run_epoch(model, loader, criterion, optimizer, device, epoch: int, training: bool) -> dict:
    if training:
        set_module_modes(model)
    else:
        model.eval()
    loss_sum = correct = total = 0.0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    session_correct = defaultdict(int)
    session_total = defaultdict(int)
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    for batch in tqdm(loader, desc=f"Epoch {epoch:02d} {'train' if training else 'val'}", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()
        predictions = logits.argmax(1)
        count = labels.size(0)
        loss_sum += float(loss.detach()) * count
        correct += int(predictions.eq(labels).sum())
        total += count
        if not training:
            for label, prediction, session in zip(labels.cpu().tolist(), predictions.cpu().tolist(), batch["session"]):
                class_name = CLASS_NAMES[label]
                class_total[class_name] += 1
                class_correct[class_name] += int(label == prediction)
                session_total[session] += 1
                session_correct[session] += int(label == prediction)
                confusion[label, prediction] += 1
    metrics = {"loss": loss_sum / max(1, total), "accuracy": correct / max(1, total)}
    if not training:
        metrics["per_class_accuracy"] = {name: class_correct[name] / max(1, class_total[name]) for name in CLASS_NAMES}
        metrics["balanced_accuracy"] = sum(metrics["per_class_accuracy"].values()) / len(CLASS_NAMES)
        metrics["per_session_accuracy"] = {session: session_correct[session] / max(1, session_total[session]) for session in sorted(session_total)}
        metrics["confusion_matrix"] = confusion.tolist()
        print(f"\nValidation diagnostics for epoch {epoch:02d}")
        print(f"  loss={metrics['loss']:.5f} accuracy={metrics['accuracy'] * 100:.2f}% balanced={metrics['balanced_accuracy'] * 100:.2f}%")
        for name in CLASS_NAMES:
            print(f"  {name:<16} {metrics['per_class_accuracy'][name] * 100:6.2f}% ({class_correct[name]}/{class_total[name]})")
        for session, value in metrics["per_session_accuracy"].items():
            print(f"  Session {session:<8} {value * 100:6.2f}%")
        print("  Confusion matrix (rows=true, columns=predicted)")
        print("                 " + " ".join(f"{name:>12}" for name in CLASS_NAMES))
        for index, name in enumerate(CLASS_NAMES):
            print(f"  {name:>16} " + " ".join(f"{value:12d}" for value in confusion[index].tolist()))
    return metrics


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: str
    output_dir: str
    seed: int
    batch_size: int
    max_epochs: int
    training_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]
    class_names: tuple[str, ...]
    initialization: str
    preprocessing: str
    learning_rates: dict[str, float]
    label_smoothing: float


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_accuracy: float, best_balanced: float, stale: int, config: TrainingConfig) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_accuracy": best_accuracy,
        "best_val_balanced_accuracy": best_balanced,
        "early_stopping_counter": stale,
        "classes": CLASS_NAMES,
        "class_to_idx": {name: index for index, name in enumerate(CLASS_NAMES)},
        "num_classes": len(CLASS_NAMES),
        "training_config": asdict(config),
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, device: torch.device) -> tuple[int, float, float, int]:
    checkpoint = torch.load(path.expanduser().resolve(), map_location=device, weights_only=False)
    if checkpoint.get("classes") != CLASS_NAMES or checkpoint.get("class_to_idx") != {name: index for index, name in enumerate(CLASS_NAMES)}:
        raise RuntimeError("Checkpoint class mapping does not match the required four classes.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint.get("scheduler_state_dict", {}))
    return checkpoint["epoch"] + 1, checkpoint.get("best_val_accuracy", float("-inf")), checkpoint.get("best_val_balanced_accuracy", float("-inf")), checkpoint.get("early_stopping_counter", 0)


def main() -> None:
    global BATCH_SIZE, BATCHES_PER_EPOCH, MAX_EPOCHS, BACKBONE_START_EPOCH
    global STEM_LR, LAYER1_LR, LAYER2_LR, LAYER3_LR, LAYER4_LR, CLASSIFIER_LR
    parser = argparse.ArgumentParser(description="Train ImageNet-1K-pretrained ResNet18 on private NEW sessions.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--backbone-start-epoch", type=int, default=BACKBONE_START_EPOCH)
    parser.add_argument("--stem-lr", type=float, default=STEM_LR)
    parser.add_argument("--layer1-lr", type=float, default=LAYER1_LR)
    parser.add_argument("--layer2-lr", type=float, default=LAYER2_LR)
    parser.add_argument("--layer3-lr", type=float, default=LAYER3_LR)
    parser.add_argument("--layer4-lr", type=float, default=LAYER4_LR)
    parser.add_argument("--classifier-lr", type=float, default=CLASSIFIER_LR)
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    MAX_EPOCHS = args.max_epochs
    BACKBONE_START_EPOCH = args.backbone_start_epoch
    STEM_LR = args.stem_lr
    LAYER1_LR = args.layer1_lr
    LAYER2_LR = args.layer2_lr
    LAYER3_LR = args.layer3_lr
    LAYER4_LR = args.layer4_lr
    CLASSIFIER_LR = args.classifier_lr
    if BATCH_SIZE <= 0 or MAX_EPOCHS <= 0 or BACKBONE_START_EPOCH < 1:
        raise ValueError("batch size, max epochs, and backbone start epoch must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if (output_dir / "best.pt").exists() and args.resume is None:
        raise FileExistsError(f"{output_dir} contains best.pt; use a fresh directory or --resume.")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PreprocessConfig()
    train_dataset = collect_samples(data_dir, TRAIN_SESSIONS, build_train_transform(config))
    val_dataset = collect_samples(data_dir, VAL_SESSIONS, build_eval_transform(config))
    if not train_dataset.samples or not val_dataset.samples:
        raise RuntimeError("Training A-R and validation S-V must both contain images.")
    BATCHES_PER_EPOCH = max(1, len(train_dataset) // BATCH_SIZE)
    sampler = BalancedBatchSampler(train_dataset, args.seed)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = build_model(pretrained=True).to(device)
    optimizer = build_optimizer(model)
    scheduler = ProgressiveLRScheduler(optimizer)
    start_epoch, best_accuracy, best_balanced, stale = 1, float("-inf"), float("-inf"), 0
    if args.resume:
        start_epoch, best_accuracy, best_balanced, stale = load_checkpoint(args.resume, model, optimizer, scheduler, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    training_config = TrainingConfig(str(data_dir), str(output_dir), args.seed, BATCH_SIZE, MAX_EPOCHS, TRAIN_SESSIONS, VAL_SESSIONS, TEST_SESSIONS, tuple(CLASS_NAMES), "torchvision ResNet18_Weights.IMAGENET1K_V1", "resnet18-in1k/preprocessing/in1k_preprocessing.py", {"stem": STEM_LR, "layer1": LAYER1_LR, "layer2": LAYER2_LR, "layer3": LAYER3_LR, "layer4": LAYER4_LR, "fc": CLASSIFIER_LR}, LABEL_SMOOTHING)
    history = []
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        sampler.set_epoch(epoch)
        stage = set_training_stage(model, epoch)
        scheduler.step(epoch)
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, epoch, True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, epoch, False)
        learning_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        record = {"epoch": epoch, "stage": stage, "learning_rates": learning_rates, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_balanced = val_metrics["balanced_accuracy"]
            stale = 0
        else:
            stale += 1
        print(f"epoch={epoch:02d} stage={stage} train_loss={train_metrics['loss']:.5f} train_acc={train_metrics['accuracy']:.4f} val_loss={val_metrics['loss']:.5f} val_acc={val_metrics['accuracy']:.4f} val_balanced={val_metrics['balanced_accuracy']:.4f}")
        print("  Learning rates: " + ", ".join(f"{name}={value:.3e}" for name, value in learning_rates.items() if value > 0))
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, best_accuracy, best_balanced, stale, training_config)
        if val_metrics["accuracy"] >= best_accuracy:
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, best_accuracy, best_balanced, stale, training_config)
        (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
        (output_dir / "training_config.json").write_text(json.dumps(asdict(training_config), indent=2) + "\n")
        if stale >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
