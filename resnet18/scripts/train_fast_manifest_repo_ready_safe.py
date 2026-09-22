#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.models import resnet18

from gesture_preprocessing import PreprocessConfig, build_eval_transform, build_train_transform


CLASS_NAMES = ["one", "two", "stop", "no_gesture"]


SCRIPT_DIR = Path(__file__).resolve().parent


def has_dataset_layout(path: Path) -> bool:
    return (path / "manifest.csv").is_file() and all((path / name).is_dir() for name in CLASS_NAMES)


def default_dataset_root() -> Path:
    """Find the preprocessed dataset next to this script/repo root."""
    candidates = [
        SCRIPT_DIR / "gestures_224",
        SCRIPT_DIR / "Gestures",
        SCRIPT_DIR / "gestures",
        SCRIPT_DIR / "dataset",
        SCRIPT_DIR,
    ]
    for candidate in candidates:
        if has_dataset_layout(candidate):
            return candidate
    # Keep the intended default even if it does not exist yet; the later error message explains what is missing.
    return SCRIPT_DIR / "gestures_224"


def resolve_repo_path(value: str | None, default_path: Path) -> Path:
    """Resolve CLI paths relative to the repo/script root instead of the current shell directory."""
    path = Path(value).expanduser() if value else default_path
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def log_status(message: str) -> None:
    print(message, flush=True)


class ManifestImageDataset(Dataset):
    def __init__(self, root: str | Path, manifest: str | Path, transform=None, class_names: Sequence[str] = CLASS_NAMES) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = Path(manifest).expanduser().resolve()
        self.transform = transform
        self.class_names = list(class_names)
        self.samples: list[tuple[Path, int]] = []

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        if not self.manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest}")

        with self.manifest.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"path", "label", "class_name"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

            class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            for line_number, row in enumerate(reader, start=2):
                rel_or_abs = Path(row["path"])
                image_path = rel_or_abs if rel_or_abs.is_absolute() else self.root / rel_or_abs
                label = int(row["label"])
                class_name = row["class_name"]
                if class_name not in class_to_idx:
                    raise ValueError(f"Unknown class_name in manifest line {line_number}: {class_name!r}")
                expected_label = class_to_idx[class_name]
                if label != expected_label:
                    raise ValueError(
                        f"Manifest label mismatch line {line_number}: class_name={class_name!r} "
                        f"has label={label}, expected {expected_label} from CLASS_NAMES={self.class_names}"
                    )
                if label < 0 or label >= len(self.class_names):
                    raise ValueError(f"Invalid label in manifest line {line_number}: {label}")
                self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(f"Manifest has no samples: {self.manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, class_index = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, class_index


def dataset_labels(dataset: Dataset) -> List[int]:
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    # Support common checkpoint layouts. This is intentionally permissive because
    # older training scripts often save under different top-level keys.
    for key in ("MODEL_STATE", "state_dict", "model_state_dict", "model", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    # If the checkpoint itself already looks like a state_dict, use it directly.
    if any(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint

    raise ValueError(
        f"Could not find a model state_dict in {checkpoint_path}. "
        "Expected one of: MODEL_STATE, state_dict, model_state_dict, model, net."
    )


def load_matching_weights(model: ResNet18WithEmbedding, state_dict: Dict[str, torch.Tensor]) -> tuple[int, int, list[str]]:
    model_state = model.backbone.state_dict()
    filtered_state = {}
    skipped: list[str] = []

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            skipped.append(key)
            continue
        normalized_key = key
        for prefix in ("module.", "backbone.", "model.", "net."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]

        if normalized_key in model_state and model_state[normalized_key].shape == tensor.shape:
            filtered_state[normalized_key] = tensor
        else:
            skipped.append(key)

    model.backbone.load_state_dict(filtered_state, strict=False)
    return len(filtered_state), len(state_dict), skipped[:12]


def build_model(num_classes: int) -> ResNet18WithEmbedding:
    return ResNet18WithEmbedding(num_classes=num_classes)


def stratified_split_indices(labels: Sequence[int], val_fraction: float, seed: int) -> tuple[List[int], List[int]]:
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

        val_count = max(1, int(round(len(indices) * val_fraction)))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


def collect_labels(dataset: Dataset) -> List[int]:
    labels = []
    for _, label in dataset:
        labels.append(int(label))
    return labels


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    weighted: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> DataLoader:
    sampler = None
    if weighted:
        labels = dataset_labels(dataset)
        class_weights = compute_class_weights(labels, len(CLASS_NAMES))
        sample_weights = torch.tensor([class_weights[label] for label in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)

    return DataLoader(dataset, **loader_kwargs)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
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
    train_manifest: str
    val_root: str | None
    val_manifest: str | None
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


def describe_device(device: torch.device, use_amp: bool) -> list[str]:
    if device.type == "cuda":
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        gpu_props = torch.cuda.get_device_properties(gpu_index)
        total_memory_gb = gpu_props.total_memory / (1024 ** 3)
        return [
            f"NVIDIA GPU acceleration: enabled ({gpu_name}, cuda:{gpu_index}, {total_memory_gb:.1f} GB VRAM)",
            f"Mixed precision AMP: {'enabled' if use_amp else 'disabled'}",
        ]

    return [
        "NVIDIA GPU acceleration: disabled - using CPU",
        "Hint: install a CUDA-enabled PyTorch build and check your NVIDIA driver if this should use the GPU.",
    ]


def print_progress(stage: str, epoch: int, epochs: int, batch_index: int, total_batches: int, loss: float, acc: float) -> None:
    total_batches = max(1, total_batches)
    percent = 100.0 * batch_index / total_batches
    bar_width = 30
    filled = int(bar_width * batch_index / total_batches)
    bar = "█" * filled + "-" * (bar_width - filled)

    sys.stdout.write(
        f"\r{stage:<5} epoch={epoch}/{epochs} "
        f"[{bar}] {batch_index}/{total_batches} "
        f"({percent:5.1f}%) loss={loss:.4f} acc={acc:.4f}"
    )
    sys.stdout.flush()


def set_frozen_backbone_eval(model: ResNet18WithEmbedding) -> None:
    """Keep the frozen feature extractor stable while training only fc."""
    model.backbone.conv1.eval()
    model.backbone.bn1.eval()
    model.backbone.relu.eval()
    model.backbone.maxpool.eval()
    model.backbone.layer1.eval()
    model.backbone.layer2.eval()
    model.backbone.layer3.eval()
    model.backbone.layer4.eval()
    model.backbone.avgpool.eval()
    model.backbone.fc.train()


def assert_finite_tensor(name: str, tensor: torch.Tensor, epoch: int, batch_index: int) -> None:
    if not torch.isfinite(tensor).all().item():
        finite_ratio = torch.isfinite(tensor).float().mean().item()
        raise FloatingPointError(
            f"Non-finite values detected in {name} at epoch={epoch}, batch={batch_index}. "
            f"finite_ratio={finite_ratio:.6f}. Stop training; try --no-amp and a lower --lr."
        )


def assert_finite_model(model: nn.Module, where: str) -> None:
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all().item():
            raise FloatingPointError(f"Model parameter contains NaN/Inf at {where}: {name}")


def format_class_distribution(labels: Sequence[int]) -> str:
    counts = torch.bincount(torch.tensor(labels), minlength=len(CLASS_NAMES)).tolist()
    total = max(1, sum(counts))
    return ", ".join(f"{CLASS_NAMES[i]}={counts[i]:,} ({100.0 * counts[i] / total:.1f}%)" for i in range(len(CLASS_NAMES)))


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, use_amp: bool, epoch: int, epochs: int, log_interval: int, freeze_backbone: bool, strict_finite: bool):
    model.train()
    if freeze_backbone:
        set_frozen_backbone_eval(model)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = len(loader)

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _ = model(images)
            loss = criterion(logits, labels)

        if strict_finite:
            assert_finite_tensor("train logits", logits, epoch, batch_index)
            assert_finite_tensor("train loss", loss, epoch, batch_index)

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

        if batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches:
            running_loss = total_loss / max(1, total_samples)
            running_acc = total_correct / max(1, total_samples)
            print_progress("Train", epoch, epochs, batch_index, total_batches, running_loss, running_acc)

    print()
    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch: int, epochs: int, log_interval: int, use_amp: bool, strict_finite: bool):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = len(loader)

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _ = model(images)
            loss = criterion(logits, labels)

        if strict_finite:
            assert_finite_tensor("val logits", logits, epoch, batch_index)
            assert_finite_tensor("val loss", loss, epoch, batch_index)

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += images.size(0)

        if batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches:
            running_loss = total_loss / max(1, total_samples)
            running_acc = total_correct / max(1, total_samples)
            print_progress("Val", epoch, epochs, batch_index, total_batches, running_loss, running_acc)

    print()
    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def save_checkpoint(output_dir: Path, model: nn.Module, config: TrainingConfig, class_names: Sequence[str], best_val_acc: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"MODEL_STATE": model.backbone.state_dict(), "class_names": list(class_names), "config": asdict(config), "best_val_acc": best_val_acc}, output_dir / "ResNet18_finetuned.pth")
    (output_dir / "class_names.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    (output_dir / "training_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 from a preprocessed local manifest dataset")
    parser.add_argument("--train-root", default=None, help="Preprocessed dataset root; default: auto-detect next to this script")
    parser.add_argument("--train-manifest", default="manifest.csv", help="Manifest CSV path, absolute or relative to --train-root")
    parser.add_argument("--val-root", default=None, help="Optional preprocessed validation root")
    parser.add_argument("--val-manifest", default=None, help="Optional validation manifest CSV")
    parser.add_argument("--pretrained-checkpoint", default=None, help="Path to the pretrained 34-class checkpoint; default: ResNet18.pth next to this script")
    parser.add_argument("--output-dir", default=None, help="Directory for checkpoints and logs; default: resnet18_4class_head next to this script")
    parser.add_argument("--epochs", type=int, default=3, help="Default is 3 for fast classifier-head training on the 1M-image dataset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4, help="Safer default after NaN; increase only after stable runs")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.05, help="Smaller val split saves time with very large datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True, help="Train only the classifier head by default; use --no-freeze-backbone for full fine-tuning")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing for cross entropy")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Fraction of total steps used for LR warmup")
    parser.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=False, help="Weight classes by inverse frequency")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False, help="Use mixed precision when available; default off because NaN was observed")
    parser.add_argument("--strict-finite", action=argparse.BooleanOptionalAction, default=True, help="Abort immediately if NaN/Inf appears in inputs, logits, loss, or weights")
    parser.add_argument("--log-interval", type=int, default=25, help="Print progress every N batches")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    args = parser.parse_args()

    print("Start training", flush=True)
    print("Mode: SAFE classifier-head training from 34-class ResNet18 checkpoint -> 4 classes", flush=True)
    print(f"Repo/script root: {SCRIPT_DIR}", flush=True)

    train_root = resolve_repo_path(args.train_root, default_dataset_root())
    train_manifest = Path(args.train_manifest).expanduser()
    if not train_manifest.is_absolute():
        train_manifest = train_root / train_manifest
    train_manifest = train_manifest.resolve()

    val_root = resolve_repo_path(args.val_root, SCRIPT_DIR) if args.val_root else None
    val_manifest = Path(args.val_manifest).expanduser() if args.val_manifest else None
    if val_manifest is not None and not val_manifest.is_absolute():
        if val_root is None:
            raise ValueError("--val-manifest can be relative only when --val-root is set")
        val_manifest = val_root / val_manifest

    config = TrainingConfig(
        train_root=str(train_root),
        train_manifest=str(train_manifest),
        val_root=str(val_root) if val_root else None,
        val_manifest=str(val_manifest) if val_manifest else None,
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
    use_amp = args.amp and device.type == "cuda"
    for line in describe_device(device, use_amp):
        print(line, flush=True)

    preprocess_config = PreprocessConfig(image_size=args.image_size)

    log_status(f"Loading train manifest: {train_manifest}")
    train_all_for_train_transform = ManifestImageDataset(train_root, train_manifest, transform=build_train_transform(preprocess_config, augment=True), class_names=CLASS_NAMES)
    train_all_for_eval_transform = ManifestImageDataset(train_root, train_manifest, transform=build_eval_transform(preprocess_config), class_names=CLASS_NAMES)

    if val_root and val_manifest:
        train_dataset = train_all_for_train_transform
        log_status(f"Loading val manifest: {val_manifest}")
        val_dataset = ManifestImageDataset(val_root, val_manifest, transform=build_eval_transform(preprocess_config), class_names=CLASS_NAMES)
    else:
        train_indices, val_indices = stratified_split_indices(dataset_labels(train_all_for_train_transform), args.val_fraction, args.seed)
        train_dataset = Subset(train_all_for_train_transform, train_indices)
        val_dataset = Subset(train_all_for_eval_transform, val_indices)

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)

    checkpoint_path = resolve_repo_path(args.pretrained_checkpoint, SCRIPT_DIR / "ResNet18.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}. "
            "Put your 34-class ResNet18 checkpoint in the repo root next to this script as ResNet18.pth, "
            "or pass --pretrained-checkpoint /path/to/checkpoint.pth"
        )
    log_status(f"Loading pretrained checkpoint: {checkpoint_path}")
    loaded_count, total_count, skipped_examples = load_matching_weights(model, load_checkpoint(checkpoint_path))
    log_status(f"Loaded matching checkpoint tensors: {loaded_count}/{total_count}")
    if skipped_examples:
        log_status(f"Skipped examples: {skipped_examples}")
    if loaded_count < 20:
        raise RuntimeError(
            "Almost no pretrained weights were loaded. This usually means the checkpoint format/key names do not match. "
            "Do not train with --freeze-backbone until the pretrained backbone loads correctly."
        )
    assert_finite_model(model, "after checkpoint load")

    if args.freeze_backbone:
        for name, param in model.backbone.named_parameters():
            param.requires_grad = name.startswith("fc.")
        log_status("Backbone frozen: training classifier head only")

    train_labels = dataset_labels(train_dataset)
    val_labels_for_print = dataset_labels(val_dataset)
    log_status(f"Train class distribution: {format_class_distribution(train_labels)}")
    log_status(f"Val class distribution:   {format_class_distribution(val_labels_for_print)}")
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_labels, len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True, weighted=False, persistent_workers=args.persistent_workers, prefetch_factor=args.prefetch_factor)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, shuffle=False, weighted=False, persistent_workers=args.persistent_workers, prefetch_factor=args.prefetch_factor)
    scheduler = build_lr_scheduler(optimizer, max(1, args.epochs * max(1, len(train_loader))), args.warmup_ratio)

    output_dir = resolve_repo_path(args.output_dir, SCRIPT_DIR / "resnet18_4class_head")
    best_val_acc = 0.0

    print(
        f"train_root={train_root} train_manifest={train_manifest} output_dir={output_dir}",
        flush=True,
    )
    print(
        f"train_samples={len(train_dataset):,} val_samples={len(val_dataset):,} "
        f"train_batches={len(train_loader):,} val_batches={len(val_loader):,} epochs={args.epochs}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler, use_amp, epoch, args.epochs, max(1, args.log_interval), args.freeze_backbone, args.strict_finite)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch, args.epochs, max(1, args.log_interval), False, args.strict_finite)

        print(f"Result epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

        if math.isfinite(val_loss) and math.isfinite(val_acc) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(output_dir, model, config, CLASS_NAMES, best_val_acc)
            print(f"Saved new best checkpoint: val_acc={best_val_acc:.4f}", flush=True)
        else:
            print("Checkpoint not saved because validation metrics are not finite or did not improve.", flush=True)

        if args.strict_finite:
            assert_finite_model(model, f"after epoch {epoch}")

    print(f"best_val_acc={best_val_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
