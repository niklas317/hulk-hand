#!/usr/bin/env python3
"""Fine-tune a FractalDB-pretrained ResNet18 on the private NEW session split.

This pipeline loads a FractalDB-pretrained ResNet18, uses FractalDB/ImageNet
preprocessing, and uses the DINOv2 pipeline's NEW-only
session split, balanced batches, warmup/cosine schedule, and early stopping.
"""

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.models import resnet18
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PIPELINE_ROOT / "preprocessing"))
from fractaldb_preprocessing import PreprocessConfig, build_eval_transform, build_train_transform


CLASS_NAMES = ["gesture_1", "gesture_2", "gesture_3", "no_gesture"]
FRACTALDB_MODEL_ID = "totti0223/resnet18_fractaldb_10000"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_SESSIONS = tuple(chr(code) for code in range(ord("A"), ord("R") + 1))
VAL_SESSIONS = tuple(chr(code) for code in range(ord("S"), ord("V") + 1))
TEST_SESSIONS = tuple(chr(code) for code in range(ord("W"), ord("Z") + 1))

BATCH_SIZE = 32
BATCHES_PER_EPOCH = 281
MAX_EPOCHS = 40
WARMUP_EPOCHS = 3
WARMUP_START_FACTOR = 0.10
BACKBONE_LR = 3e-5
CLASSIFIER_LR = 3e-4
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 4
HEAD_ONLY_EPOCHS = 3
LAYER4_START_EPOCH = HEAD_ONLY_EPOCHS + 1
FULL_BACKBONE_START_EPOCH = 7


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
            image_tensor = self.transform(image.convert("RGB"))
        return {"image": image_tensor, "label": sample.label, "session": sample.session}


class BalancedBatchSampler(Sampler[list[int]]):
    """Yield exactly four equally represented classes in every batch."""

    def __init__(self, dataset: SessionDataset, seed: int) -> None:
        if BATCH_SIZE % len(CLASS_NAMES) != 0:
            raise ValueError("BATCH_SIZE must be divisible by the number of classes.")
        grouped: dict[int, list[int]] = {label: [] for label in range(len(CLASS_NAMES))}
        for index, sample in enumerate(dataset.samples):
            grouped[sample.label].append(index)
        missing = [CLASS_NAMES[label] for label, indices in grouped.items() if not indices]
        if missing:
            raise RuntimeError(
                "Every class must have at least one training image in sessions A-R; "
                f"missing: {', '.join(missing)}."
            )
        self.dataset = dataset
        self.grouped = grouped
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        per_class = BATCH_SIZE // len(CLASS_NAMES)
        sequences: dict[int, list[int]] = {}
        required = BATCHES_PER_EPOCH * per_class
        for label, source in self.grouped.items():
            by_session: dict[str, list[int]] = defaultdict(list)
            for index in source:
                by_session[self.dataset.samples[index].session].append(index)
            session_order = list(by_session)
            cursors = {session: 0 for session in session_order}
            for pool in by_session.values():
                rng.shuffle(pool)
            sequence = []
            while len(sequence) < required:
                rng.shuffle(session_order)
                for session in session_order:
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


class ResNet18WithEmbedding(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone
        x = features.maxpool(features.relu(features.bn1(features.conv1(images))))
        x = features.layer1(x)
        x = features.layer2(x)
        x = features.layer3(x)
        x = features.layer4(x)
        embedding = torch.flatten(features.avgpool(x), 1)
        return features.fc(embedding), embedding


def collect_samples(data_dir: Path, sessions: tuple[str, ...], transform) -> SessionDataset:
    samples: list[Sample] = []
    for label, class_name in enumerate(CLASS_NAMES):
        for session in sessions:
            session_dir = data_dir / class_name / f"session_{session}"
            if not session_dir.is_dir():
                continue
            for path in sorted(session_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append(Sample(path, label, class_name, session))
    return SessionDataset(samples, transform)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: ResNet18WithEmbedding) -> AdamW:
    return AdamW(
        [
            {"params": model.backbone.layer2.parameters(), "lr": BACKBONE_LR, "name": "backbone"},
            {"params": model.backbone.layer3.parameters(), "lr": BACKBONE_LR, "name": "backbone"},
            {"params": model.backbone.layer4.parameters(), "lr": BACKBONE_LR, "name": "backbone"},
            {"params": model.backbone.conv1.parameters(), "lr": BACKBONE_LR, "name": "backbone"},
            {"params": model.backbone.bn1.parameters(), "lr": BACKBONE_LR, "name": "backbone"},
            {"params": model.backbone.fc.parameters(), "lr": CLASSIFIER_LR, "name": "classifier"},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def set_training_stage(model: ResNet18WithEmbedding, epoch: int) -> str:
    """Progressively unfreeze the backbone while keeping layer1 frozen."""
    layer4_trainable = epoch >= LAYER4_START_EPOCH
    full_backbone_trainable = epoch >= FULL_BACKBONE_START_EPOCH
    for parameter in model.backbone.layer1.parameters():
        parameter.requires_grad = False
    for module in (model.backbone.layer2, model.backbone.layer3, model.backbone.conv1, model.backbone.bn1):
        for parameter in module.parameters():
            parameter.requires_grad = full_backbone_trainable
    for parameter in model.backbone.layer4.parameters():
        parameter.requires_grad = layer4_trainable
    if not layer4_trainable:
        return "classifier-only"
    if not full_backbone_trainable:
        return "layer4+classifier"
    return "layer2-4+stem+classifier"


def build_scheduler(optimizer: AdamW) -> SequentialLR:
    return SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=WARMUP_START_FACTOR, end_factor=1.0, total_iters=WARMUP_EPOCHS),
            CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS),
        ],
        milestones=[WARMUP_EPOCHS],
    )


def run_epoch(model, loader, criterion, optimizer, device, epoch: int, training: bool) -> tuple[float, float]:
    model.train(training)
    if training:
        for module in (
            model.backbone.bn1,
            model.backbone.layer1,
            model.backbone.layer2,
            model.backbone.layer3,
            model.backbone.layer4,
        ):
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()
    total_loss = total_correct = total_samples = 0
    phase = "train" if training else "val"
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch:02d} {phase}",
        leave=False,
    )
    for batch_index, batch in enumerate(progress, 1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits, embedding = model(images)
            if not torch.isfinite(logits).all() or not torch.isfinite(embedding).all():
                raise FloatingPointError(f"Non-finite model output at epoch {epoch}, batch {batch_index}.")
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()
        count = labels.size(0)
        total_loss += loss.detach().item() * count
        total_correct += int(logits.argmax(1).eq(labels).sum())
        total_samples += count
        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples * 100:.2f}%",
        )
    if not total_samples:
        raise RuntimeError("DataLoader produced no samples.")
    return total_loss / total_samples, total_correct / total_samples


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: str
    output_dir: str
    seed: int
    image_size: int
    batch_size: int
    batches_per_epoch: int
    max_epochs: int
    training_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]
    initialization: str
    preprocessing: str


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_accuracy: float, counter: int, classes: list[str], config: TrainingConfig) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_accuracy": best_accuracy,
            "early_stopping_counter": counter,
            "classes": classes,
            "class_to_idx": {name: index for index, name in enumerate(classes)},
            "training_config": asdict(config),
        },
        path,
    )


def load_fractaldb_backbone(model: ResNet18WithEmbedding) -> None:
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "FractalDB loading requires huggingface-hub and safetensors. "
            "Install resnet18-fractaldb/requirements.txt."
        ) from exc

    checkpoint_path = hf_hub_download(
        repo_id=FRACTALDB_MODEL_ID,
        filename="model.safetensors",
    )
    state_dict = load_file(checkpoint_path, device="cpu")
    normalized = {}
    for key, tensor in state_dict.items():
        key = key.removeprefix("model.")
        if key.startswith("fc."):
            continue
        normalized[key] = tensor
    missing, unexpected = model.backbone.load_state_dict(normalized, strict=False)
    if set(missing) != {"fc.weight", "fc.bias"} or unexpected:
        raise RuntimeError(
            "FractalDB checkpoint did not match torchvision ResNet18. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )


def freeze_first_stage(model: ResNet18WithEmbedding) -> None:
    for parameter in model.backbone.layer1.parameters():
        parameter.requires_grad = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a FractalDB-pretrained ResNet18 on private NEW sessions.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Private dataset containing class/session directories")
    parser.add_argument("--output-dir", type=Path, default=PIPELINE_ROOT / "artifacts" / "runs" / "resnet18-fractaldb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None, help="Optional matching last.pt checkpoint")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = PreprocessConfig(image_size=224)
    train_dataset = collect_samples(data_dir, TRAIN_SESSIONS, build_train_transform(preprocess, augment=True))
    val_dataset = collect_samples(data_dir, VAL_SESSIONS, build_eval_transform(preprocess))
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Both NEW training sessions A-R and validation sessions S-V must contain images.")

    sampler = BalancedBatchSampler(train_dataset, args.seed)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = ResNet18WithEmbedding(len(CLASS_NAMES))
    load_fractaldb_backbone(model)
    freeze_first_stage(model)
    set_training_stage(model, 1)
    model.to(device)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    start_epoch = 1
    best_accuracy = float("-inf")
    stopping_counter = 0
    config = TrainingConfig(str(data_dir), str(output_dir), args.seed, 224, BATCH_SIZE, BATCHES_PER_EPOCH, MAX_EPOCHS, TRAIN_SESSIONS, VAL_SESSIONS, TEST_SESSIONS, f"FractalDB pretrained weights ({FRACTALDB_MODEL_ID}), staged fine-tuning with layer1 frozen", "resnet18-fractaldb/preprocessing/fractaldb_preprocessing.py")

    if args.resume:
        checkpoint = torch.load(args.resume.expanduser().resolve(), map_location=device)
        if checkpoint.get("training_config", {}).get("initialization") != config.initialization:
            raise ValueError("Resume checkpoint is not a matching FractalDB ResNet18 checkpoint.")
        if checkpoint.get("classes") != CLASS_NAMES:
            raise ValueError(
                "Resume checkpoint has incompatible class names; expected "
                f"{CLASS_NAMES}."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_accuracy = float(checkpoint["best_val_accuracy"])
        stopping_counter = int(checkpoint["early_stopping_counter"])

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    print(f"Training FractalDB ResNet18 on NEW sessions A-R; device={device}; train={len(train_dataset)}; val={len(val_dataset)}")
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        stage = set_training_stage(model, epoch)
        sampler.set_epoch(epoch - 1)
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device, epoch, True)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, optimizer, device, epoch, False)
        scheduler.step()
        improved = val_accuracy > best_accuracy
        if improved:
            best_accuracy = val_accuracy
            stopping_counter = 0
        else:
            stopping_counter += 1
        print(f"epoch={epoch:02d} stage={stage} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}")
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, best_accuracy, stopping_counter, CLASS_NAMES, config)
        if improved:
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, best_accuracy, stopping_counter, CLASS_NAMES, config)
        if stopping_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break
    (output_dir / "class_names.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    (output_dir / "training_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
