#!/usr/bin/env python3
"""Train the direct recursive-folder ResNet18 gesture classification pipeline."""

from __future__ import annotations

import argparse
import math
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.models import resnet18
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from gesture_preprocessing import PreprocessConfig, build_eval_transform, build_train_transform


REPO_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ["one", "two", "stop", "no_gesture"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RecursiveLabelDataset(Dataset):
    """Read images recursively from the fixed four-class directory layout."""

    def __init__(self, root: str | Path, transform=None, class_names: Sequence[str] = CLASS_NAMES) -> None:
        self.root = Path(root)
        self.transform = transform
        self.class_names = list(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.samples: list[tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[class_name]))

        if not self.samples:
            raise ValueError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        # Open files per sample so workers do not retain image handles between batches.
        path, class_index = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, class_index


def dataset_labels(dataset: Dataset) -> List[int]:
    """Retrieve labels efficiently from datasets and Subset wrappers."""
    if isinstance(dataset, Subset):
        source = dataset.dataset
        if hasattr(source, "samples"):
            return [int(source.samples[i][1]) for i in dataset.indices]
        return [int(source[i][1]) for i in dataset.indices]

    if hasattr(dataset, "samples"):
        return [int(class_index) for _, class_index in dataset.samples]

    return collect_labels(dataset)


class ResNet18WithEmbedding(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor):
        # Keep the embedding available for the matching ONNX export path.
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        embedding = torch.flatten(x, 1)
        logits = self.backbone.fc(embedding)
        return logits, embedding


def load_checkpoint(checkpoint_path: str | Path) -> Dict[str, torch.Tensor]:
    """Load the model state from the checkpoint produced by the base model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "MODEL_STATE" in checkpoint:
            return checkpoint["MODEL_STATE"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def load_matching_weights(model: ResNet18WithEmbedding, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load only tensors whose names and shapes match the four-class model."""
    model_state = model.backbone.state_dict()
    filtered_state = {}

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        normalized_key = key
        if normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]
        if normalized_key.startswith("backbone."):
            normalized_key = normalized_key[len("backbone.") :]

        if normalized_key in model_state and model_state[normalized_key].shape == tensor.shape:
            filtered_state[normalized_key] = tensor

    model.backbone.load_state_dict(filtered_state, strict=False)


def build_model(num_classes: int) -> ResNet18WithEmbedding:
    return ResNet18WithEmbedding(num_classes=num_classes)


def stratified_split_indices(labels: Sequence[int], val_fraction: float, seed: int) -> tuple[List[int], List[int]]:
    """Split each class independently so validation retains class coverage."""
    by_class: Dict[int, List[int]] = {idx: [] for idx in range(len(CLASS_NAMES))}
    for index, label in enumerate(labels):
        by_class[int(label)].append(index)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for indices in by_class.values():
        if not indices:
            continue

        rng.shuffle(indices)
        if len(indices) == 1:
            train_indices.extend(indices)
            continue

        # Reserve at least one sample while keeping one sample available for training.
        val_count = max(1, int(round(len(indices) * val_fraction)))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """Return inverse-frequency weights for optional imbalanced training."""
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


def collect_labels(dataset: Dataset) -> List[int]:
    labels = []
    for _, label in dataset:
        labels.append(int(label))
    return labels


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, weighted: bool = False) -> DataLoader:
    """Build a standard or class-balanced data loader."""
    sampler = None
    if weighted:
        labels = dataset_labels(dataset)
        class_weights = compute_class_weights(labels, len(CLASS_NAMES))
        sample_weights = torch.tensor([class_weights[label] for label in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def build_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    """Create linear warmup followed by cosine decay over optimizer steps."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, float(step - warmup_steps) / float(decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@dataclass(frozen=True)
class TrainingConfig:
    train_root: str
    val_root: str | None
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    val_fraction: float
    seed: int
    image_size: int
    label_smoothing: float
    warmup_ratio: float
    use_class_weights: bool
    amp: bool


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, use_amp: bool):
    """Run one optimization epoch and return mean loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _ = model(images)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += images.size(0)

    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model without updating parameters."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits, _ = model(images)
            loss = criterion(logits, labels)

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += images.size(0)

    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def save_checkpoint(output_dir: Path, model: nn.Module, config: TrainingConfig, class_names: Sequence[str], best_val_acc: float) -> None:
    """Persist weights and the metadata needed to reproduce the trained model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"MODEL_STATE": model.backbone.state_dict(), "class_names": list(class_names), "config": asdict(config), "best_val_acc": best_val_acc}, output_dir / "ResNet18_finetuned.pth")
    (output_dir / "class_names.txt").write_text("\n".join(class_names) + "\n")
    (output_dir / "training_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")


def main() -> None:
    """Configure datasets and run direct ResNet18 fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 for one, two, stop, no_gesture")
    parser.add_argument("--train-root", required=True, help="Dataset root containing one/, two/, stop/, and no_gesture/")
    parser.add_argument("--val-root", default=None, help="Optional validation dataset root with the same folder layout")
    parser.add_argument("--pretrained-checkpoint", default=str(REPO_ROOT / "resnet18" / "artifacts" / "ResNet18.pth"), help="Path to the pretrained checkpoint")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "resnet18" / "artifacts" / "runs" / "resnet18"), help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the classifier head")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing for cross entropy")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Fraction of total steps used for LR warmup")
    parser.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=False, help="Weight classes by inverse frequency")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use mixed precision when available")
    args = parser.parse_args()

    config = TrainingConfig(
        train_root=args.train_root,
        val_root=args.val_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        image_size=args.image_size,
        label_smoothing=args.label_smoothing,
        warmup_ratio=args.warmup_ratio,
        use_class_weights=args.use_class_weights,
        amp=args.amp,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess_config = PreprocessConfig(image_size=args.image_size)
    train_base = RecursiveLabelDataset(args.train_root, transform=build_train_transform(preprocess_config, augment=True), class_names=CLASS_NAMES)
    eval_base = RecursiveLabelDataset(args.train_root, transform=build_eval_transform(preprocess_config), class_names=CLASS_NAMES)

    if args.val_root:
        train_dataset = train_base
        val_dataset = RecursiveLabelDataset(args.val_root, transform=build_eval_transform(preprocess_config), class_names=CLASS_NAMES)
    else:
        train_indices, val_indices = stratified_split_indices(dataset_labels(train_base), args.val_fraction, args.seed)
        train_dataset = Subset(train_base, train_indices)
        val_dataset = Subset(eval_base, val_indices)

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    load_matching_weights(model, load_checkpoint(args.pretrained_checkpoint))

    if args.freeze_backbone:
        for name, param in model.backbone.named_parameters():
            param.requires_grad = name.startswith("fc.")

    train_labels = dataset_labels(train_dataset)
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_labels, len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True, weighted=False)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, shuffle=False, weighted=False)
    scheduler = build_lr_scheduler(optimizer, max(1, args.epochs * max(1, len(train_loader))), args.warmup_ratio)

    output_dir = Path(args.output_dir)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Save only the best validation checkpoint so later export uses the strongest run.
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler, args.amp and device.type == "cuda")
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(output_dir, model, config, CLASS_NAMES, best_val_acc)

    print(f"best_val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
