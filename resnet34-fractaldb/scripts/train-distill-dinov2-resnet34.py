#!/usr/bin/env python3
"""Distill a finished DINOv2 gesture classifier into FractalDB ResNet34."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
DINOV2_SCRIPTS = REPO_ROOT / "dinov2" / "scripts"
sys.path.insert(0, str(DINOV2_SCRIPTS))

from model import build_model as build_dino_model  # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def load_student_trainer():
    path = SCRIPT_DIR / "train-resnet34-fractaldb.py"
    spec = importlib.util.spec_from_file_location("resnet34_fractaldb_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load student trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


student_api = load_student_trainer()

CLASS_NAMES = ["gesture_1", "gesture_2", "gesture_3", "no_gesture"]
TRAIN_SESSIONS = tuple(chr(code) for code in range(ord("A"), ord("R") + 1))
VAL_SESSIONS = tuple(chr(code) for code in range(ord("S"), ord("V") + 1))
TEST_SESSIONS = tuple(chr(code) for code in range(ord("W"), ord("Z") + 1))

BATCH_SIZE = 32
MAX_EPOCHS = 40
HEAD_ONLY_EPOCHS = 3
LAYER4_START_EPOCH = HEAD_ONLY_EPOCHS + 1
WARMUP_EPOCHS = 3
WARMUP_START_FACTOR = 0.10
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 3e-4
WEIGHT_DECAY = 5e-4
GRADIENT_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 7
ALPHA = 0.5
TEMPERATURE = 2.0
LABEL_SMOOTHING = 0.05


class PairedDataset(Dataset):
    """Apply one stochastic image transform, then create teacher/student tensors."""

    def __init__(self, samples, train: bool, image_size: int = 224) -> None:
        if not samples:
            raise RuntimeError("Paired dataset contains no samples.")
        self.samples = samples
        self.train = train
        config = student_api.PreprocessConfig(image_size=image_size)
        self.shared_augmentation = student_api.build_shared_augmentation(config) if train else None
        self.student_normalize = transforms.Normalize(config.mean, config.std)
        self.teacher_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.resize = transforms.Resize((image_size, image_size), antialias=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if self.train:
                image = self.shared_augmentation(image)
            else:
                image = self.resize(image)
            tensor = transforms.ToTensor()(image)
        return {
            "teacher_image": self.teacher_normalize(tensor.clone()),
            "student_image": self.student_normalize(tensor),
            "label": sample.label,
            "session": sample.session,
        }


def load_checkpoint(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid teacher checkpoint format.")
    required = {"model_state_dict", "classes", "class_to_idx", "num_classes"}
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(f"Teacher checkpoint is missing keys: {sorted(missing)}")
    if checkpoint["classes"] != CLASS_NAMES:
        raise RuntimeError(
            f"Teacher class order mismatch: {checkpoint['classes']} != {CLASS_NAMES}"
        )
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if checkpoint["class_to_idx"] != expected_mapping:
        raise RuntimeError("Teacher class_to_idx does not match the required class order.")
    if int(checkpoint["num_classes"]) != 4:
        raise RuntimeError("Teacher must have exactly four output classes.")
    return checkpoint


def load_teacher(path: Path, checkpoint: dict, device: torch.device) -> nn.Module:
    config = checkpoint.get("training_config", {})
    if config.get("backbone") != "dinov2_vits14":
        raise RuntimeError(
            "Teacher checkpoint was not produced by the expected DINOv2 ViT-S/14 pipeline. "
            f"Found backbone={config.get('backbone')!r}."
        )
    if int(config.get("embedding_dim", -1)) != 384:
        raise RuntimeError("Teacher checkpoint must contain the 384-dimensional DINOv2 embedding.")
    if "num_trainable_blocks" not in config:
        raise RuntimeError("Teacher checkpoint is missing training_config.num_trainable_blocks.")
    num_blocks = int(config.get("num_trainable_blocks", 10))
    teacher = build_dino_model(num_classes=4, pretrained=False, num_trainable_blocks=num_blocks)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.to(device)
    return teacher


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = ALPHA,
    temperature: float = TEMPERATURE,
    label_smoothing: float = LABEL_SMOOTHING,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce_loss = nn.functional.cross_entropy(
        student_logits, labels, label_smoothing=label_smoothing
    )
    kd_loss = nn.functional.kl_div(
        nn.functional.log_softmax(student_logits / temperature, dim=1),
        nn.functional.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)
    total_loss = (1.0 - alpha) * ce_loss + alpha * kd_loss
    return total_loss, ce_loss, kd_loss


def build_student_optimizer(model: nn.Module) -> AdamW:
    return AdamW(
        [
            {"params": model.backbone.layer2.parameters(), "lr": BACKBONE_LR},
            {"params": model.backbone.layer3.parameters(), "lr": BACKBONE_LR},
            {"params": model.backbone.layer4.parameters(), "lr": BACKBONE_LR},
            {"params": model.backbone.conv1.parameters(), "lr": BACKBONE_LR},
            {"params": model.backbone.bn1.parameters(), "lr": BACKBONE_LR},
            {"params": model.backbone.fc.parameters(), "lr": CLASSIFIER_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def set_student_stage(model: nn.Module, epoch: int) -> str:
    for parameter in model.backbone.layer1.parameters():
        parameter.requires_grad = False
    for module in (model.backbone.layer2, model.backbone.layer3, model.backbone.conv1, model.backbone.bn1):
        for parameter in module.parameters():
            parameter.requires_grad = False
    layer4_trainable = epoch >= LAYER4_START_EPOCH
    for parameter in model.backbone.layer4.parameters():
        parameter.requires_grad = layer4_trainable
    return "layer4+classifier" if layer4_trainable else "classifier-only"


def set_student_modes(model: nn.Module, freeze_bn: bool) -> None:
    model.train()
    for module in (model.backbone.layer1, model.backbone.layer2, model.backbone.layer3, model.backbone.layer4):
        if not any(parameter.requires_grad for parameter in module.parameters()):
            module.eval()
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


@torch.inference_mode()
def evaluate_teacher(teacher, loader, device) -> float:
    correct = total = 0
    for batch in tqdm(loader, desc="Teacher validation", leave=False):
        logits = teacher(batch["teacher_image"].to(device, non_blocking=True))["logits"]
        labels = batch["label"].to(device, non_blocking=True)
        correct += int(logits.argmax(1).eq(labels).sum())
        total += labels.numel()
    return correct / max(1, total)


def print_validation_metrics(class_correct, class_total, session_correct, session_total, confusion):
    print("  Validation per-class accuracy")
    for name in CLASS_NAMES:
        print(f"    {name:<16} {class_correct[name] / max(1, class_total[name]) * 100:6.2f}% ({class_correct[name]}/{class_total[name]})")
    balanced = sum(class_correct[name] / max(1, class_total[name]) for name in CLASS_NAMES) / 4
    print(f"  Balanced validation accuracy: {balanced * 100:.2f}%")
    print("  Validation per-session accuracy")
    for session in sorted(session_total):
        print(f"    Session {session:<10} {session_correct[session] / max(1, session_total[session]) * 100:6.2f}% ({session_correct[session]}/{session_total[session]})")
    print("  Confusion matrix (rows=true, columns=predicted)")
    print("                   " + " ".join(f"{name:>12}" for name in CLASS_NAMES))
    for index, name in enumerate(CLASS_NAMES):
        print(f"    {name:>16} " + " ".join(f"{int(value):12d}" for value in confusion[index]))
    return balanced


def run_epoch(student, teacher, loader, optimizer, device, training: bool, freeze_bn: bool):
    if training:
        set_student_modes(student, freeze_bn)
    else:
        student.eval()
    total_loss = total_ce = total_kd = 0.0
    correct = total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    session_correct = defaultdict(int)
    session_total = defaultdict(int)
    confusion = torch.zeros((4, 4), dtype=torch.long)
    progress = tqdm(loader, desc="Train" if training else "Validation", leave=False)
    for batch in progress:
        labels = batch["label"].to(device, non_blocking=True)
        student_images = batch["student_image"].to(device, non_blocking=True)
        teacher_images = batch["teacher_image"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.inference_mode():
            teacher_logits = teacher(teacher_images)["logits"]
        with torch.set_grad_enabled(training):
            student_logits, _ = student(student_images)
            loss, ce_loss, kd_loss = distillation_loss(student_logits, teacher_logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()
        count = labels.size(0)
        predictions = student_logits.argmax(1)
        total_loss += loss.detach().item() * count
        total_ce += ce_loss.detach().item() * count
        total_kd += kd_loss.detach().item() * count
        correct += int(predictions.eq(labels).sum())
        total += count
        if not training:
            for label, prediction, session in zip(labels.cpu().tolist(), predictions.cpu().tolist(), batch["session"]):
                class_total[CLASS_NAMES[label]] += 1
                class_correct[CLASS_NAMES[label]] += int(label == prediction)
                session_total[session] += 1
                session_correct[session] += int(label == prediction)
                confusion[label, prediction] += 1
        progress.set_postfix(loss=f"{total_loss / max(1, total):.4f}", acc=f"{correct / max(1, total) * 100:.2f}%")
    metrics = {
        "total_loss": total_loss / max(1, total),
        "ce_loss": total_ce / max(1, total),
        "kd_loss": total_kd / max(1, total),
        "accuracy": correct / max(1, total),
    }
    if not training:
        metrics["balanced_accuracy"] = print_validation_metrics(class_correct, class_total, session_correct, session_total, confusion)
    return metrics


def save_student_checkpoint(path, student, optimizer, scheduler, epoch, best_accuracy, history, config):
    torch.save({
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_accuracy": best_accuracy,
        "classes": CLASS_NAMES,
        "class_to_idx": {name: index for index, name in enumerate(CLASS_NAMES)},
        "training_config": config,
        "metrics": history,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill DINOv2 into FractalDB ResNet34.")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-bn", action="store_true")
    parser.add_argument("--minimum-teacher-val-accuracy", type=float, default=0.80)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if (args.output_dir / "best.pt").exists():
        raise FileExistsError("Output directory already contains best.pt; use a new directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    missing_classes = [name for name in CLASS_NAMES if not (data_dir / name).is_dir()]
    if missing_classes:
        raise RuntimeError(
            "Dataset is missing required class directories: " + ", ".join(missing_classes)
        )
    set_seed = student_api.set_seed
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_checkpoint = load_checkpoint(args.teacher_checkpoint)
    teacher = load_teacher(args.teacher_checkpoint, teacher_checkpoint, device)
    train_raw = student_api.collect_samples(data_dir, TRAIN_SESSIONS, lambda image: image)
    val_raw = student_api.collect_samples(data_dir, VAL_SESSIONS, lambda image: image)
    train_dataset = PairedDataset(train_raw.samples, train=True)
    val_dataset = PairedDataset(val_raw.samples, train=False)
    train_sampler = student_api.BalancedBatchSampler(train_raw, seed=42)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    teacher_val_accuracy = evaluate_teacher(teacher, val_loader, device)
    print(f"Teacher validation accuracy: {teacher_val_accuracy * 100:.2f}%")
    if teacher_val_accuracy < args.minimum_teacher_val_accuracy:
        raise RuntimeError("Teacher validation accuracy is below the required threshold; student training stopped.")

    student = student_api.ResNet34WithEmbedding(4)
    student_api.load_fractaldb_backbone(student)
    student_api.freeze_first_stage(student)
    set_student_stage(student, 1)
    student.to(device)
    optimizer = build_student_optimizer(student)
    scheduler = SequentialLR(optimizer, [LinearLR(optimizer, start_factor=WARMUP_START_FACTOR, total_iters=WARMUP_EPOCHS), CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS)], milestones=[WARMUP_EPOCHS])

    # Required preflight: prove teacher is frozen and the student can update.
    smoke_batch = next(iter(train_loader))
    teacher_before = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
    student_before = {name: value.detach().clone() for name, value in student.state_dict().items()}
    smoke_teacher = teacher(smoke_batch["teacher_image"].to(device))["logits"]
    smoke_student, _ = student(smoke_batch["student_image"].to(device))
    smoke_loss, _, _ = distillation_loss(smoke_student, smoke_teacher, smoke_batch["label"].to(device))
    optimizer.zero_grad(set_to_none=True)
    smoke_loss.backward()
    optimizer.step()
    assert all(torch.equal(value, teacher.state_dict()[name]) for name, value in teacher_before.items())
    assert any(not torch.equal(value, student.state_dict()[name]) for name, value in student_before.items())
    print(f"Smoke test passed: teacher/student logits={tuple(smoke_student.shape)}, loss={smoke_loss.item():.4f}")
    if args.smoke_test:
        return

    # The smoke step is deliberately not part of the real optimization run.
    student = student_api.ResNet34WithEmbedding(4)
    student_api.load_fractaldb_backbone(student)
    student_api.freeze_first_stage(student)
    set_student_stage(student, 1)
    student.to(device)
    optimizer = build_student_optimizer(student)
    scheduler = SequentialLR(optimizer, [LinearLR(optimizer, start_factor=WARMUP_START_FACTOR, total_iters=WARMUP_EPOCHS), CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS)], milestones=[WARMUP_EPOCHS])

    history = []
    best_accuracy = float("-inf")
    stale_epochs = 0
    config = asdict(DistillationConfig(str(args.teacher_checkpoint.expanduser().resolve()), str(data_dir), str(args.output_dir.resolve()), ALPHA, TEMPERATURE, LABEL_SMOOTHING, BACKBONE_LR, CLASSIFIER_LR, WEIGHT_DECAY, MAX_EPOCHS, EARLY_STOPPING_PATIENCE, args.freeze_bn, teacher_val_accuracy))
    for epoch in range(1, MAX_EPOCHS + 1):
        stage = set_student_stage(student, epoch)
        train_metrics = run_epoch(student, teacher, train_loader, optimizer, device, True, args.freeze_bn)
        val_metrics = run_epoch(student, teacher, val_loader, optimizer, device, False, args.freeze_bn)
        scheduler.step()
        record = {"epoch": epoch, "stage": stage, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        improved = val_metrics["accuracy"] > best_accuracy
        if improved:
            best_accuracy = val_metrics["accuracy"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(f"epoch={epoch:02d} stage={stage} train_acc={train_metrics['accuracy']:.4f} train_ce={train_metrics['ce_loss']:.4f} train_kd={train_metrics['kd_loss']:.4f} train_total={train_metrics['total_loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_balanced={val_metrics['balanced_accuracy']:.4f}")
        save_student_checkpoint(args.output_dir / "last.pt", student, optimizer, scheduler, epoch, best_accuracy, history, config)
        if improved:
            save_student_checkpoint(args.output_dir / "best.pt", student, optimizer, scheduler, epoch, best_accuracy, history, config)
        (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
        (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break


@dataclass(frozen=True)
class DistillationConfig:
    teacher_checkpoint: str
    data_dir: str
    output_dir: str
    alpha: float
    temperature: float
    label_smoothing: float
    backbone_lr: float
    classifier_lr: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    freeze_bn: bool
    teacher_validation_accuracy: float


if __name__ == "__main__":
    main()
