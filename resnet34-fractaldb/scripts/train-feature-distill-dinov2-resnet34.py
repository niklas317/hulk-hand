#!/usr/bin/env python3
"""Feature-distill DINOv2 into the unchanged FractalDB ResNet34.

The linear projection exists only for training. Deployment checkpoints contain
the ResNet34 state separately; the projection is not part of the student model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "train-distill-dinov2-resnet34.py"
spec = importlib.util.spec_from_file_location("distillation_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load distillation base: {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

CLASS_NAMES = base.CLASS_NAMES
TRAIN_SESSIONS = base.TRAIN_SESSIONS
VAL_SESSIONS = base.VAL_SESSIONS
BATCH_SIZE = base.BATCH_SIZE
MAX_EPOCHS = base.MAX_EPOCHS
LAYER4_START_EPOCH = base.LAYER4_START_EPOCH
FULL_BACKBONE_START_EPOCH = base.FULL_BACKBONE_START_EPOCH
WARMUP_EPOCHS = base.WARMUP_EPOCHS
WARMUP_START_FACTOR = base.WARMUP_START_FACTOR
STEM_LR = base.STEM_LR
LAYER2_LR = base.LAYER2_LR
LAYER3_LR = base.LAYER3_LR
LAYER4_LR = base.LAYER4_LR
CLASSIFIER_LR = base.CLASSIFIER_LR
WEIGHT_DECAY = base.WEIGHT_DECAY
GRADIENT_CLIP_NORM = base.GRADIENT_CLIP_NORM
EARLY_STOPPING_PATIENCE = base.EARLY_STOPPING_PATIENCE
TEMPERATURE = 2.0
CE_WEIGHT = 0.7
KD_WEIGHT = 0.2
FEATURE_WEIGHT = 0.1
LABEL_SMOOTHING = base.LABEL_SMOOTHING


def build_optimizer(student: nn.Module, projection: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {"name": "stem", "params": [*student.backbone.conv1.parameters(), *student.backbone.bn1.parameters()], "lr": STEM_LR, "active_epoch": FULL_BACKBONE_START_EPOCH},
            {"name": "layer2", "params": student.backbone.layer2.parameters(), "lr": LAYER2_LR, "active_epoch": FULL_BACKBONE_START_EPOCH},
            {"name": "layer3", "params": student.backbone.layer3.parameters(), "lr": LAYER3_LR, "active_epoch": FULL_BACKBONE_START_EPOCH},
            {"name": "layer4", "params": student.backbone.layer4.parameters(), "lr": LAYER4_LR, "active_epoch": LAYER4_START_EPOCH},
            {"name": "classifier", "params": student.backbone.fc.parameters(), "lr": CLASSIFIER_LR, "active_epoch": 1},
            {"name": "projection", "params": projection.parameters(), "lr": CLASSIFIER_LR, "active_epoch": 1},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def feature_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ce_loss = F.cross_entropy(student_logits, labels, label_smoothing=LABEL_SMOOTHING)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / TEMPERATURE, dim=1),
        F.softmax(teacher_logits.detach() / TEMPERATURE, dim=1),
        reduction="batchmean",
    ) * TEMPERATURE * TEMPERATURE
    feature_loss = 1.0 - (
        F.normalize(student_embedding, dim=1) * F.normalize(teacher_embedding.detach(), dim=1)
    ).sum(dim=1).mean()
    total_loss = CE_WEIGHT * ce_loss + KD_WEIGHT * kd_loss + FEATURE_WEIGHT * feature_loss
    return total_loss, ce_loss, kd_loss, feature_loss


def set_modes(student: nn.Module, projection: nn.Module) -> None:
    base.set_student_modes(student)
    projection.train(True)


def run_epoch(student, projection, teacher, loader, optimizer, device, epoch: int, training: bool):
    if training:
        set_modes(student, projection)
    else:
        student.eval()
        projection.eval()
    totals = {"total_loss": 0.0, "ce_loss": 0.0, "kd_loss": 0.0, "feature_loss": 0.0}
    correct = total = 0
    for batch in tqdm(loader, desc=f"{'Train' if training else 'Validation'} {epoch:02d}", leave=False):
        labels = batch["label"].to(device, non_blocking=True)
        student_images = batch["student_image"].to(device, non_blocking=True)
        teacher_images = batch["teacher_image"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.inference_mode():
            teacher_output = teacher(teacher_images)
        with torch.set_grad_enabled(training):
            student_logits, student_embedding = student(student_images)
            projected_embedding = projection(student_embedding)
            loss, ce_loss, kd_loss, feature_loss = feature_distillation_loss(
                student_logits, teacher_output["logits"], labels, projected_embedding, teacher_output["embedding"]
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([*student.parameters(), *projection.parameters()], GRADIENT_CLIP_NORM)
                optimizer.step()
        count = labels.size(0)
        totals["total_loss"] += float(loss.detach()) * count
        totals["ce_loss"] += float(ce_loss.detach()) * count
        totals["kd_loss"] += float(kd_loss.detach()) * count
        totals["feature_loss"] += float(feature_loss.detach()) * count
        correct += int(student_logits.argmax(1).eq(labels).sum())
        total += count
    return {**{name: value / max(1, total) for name, value in totals.items()}, "accuracy": correct / max(1, total)}


def save_checkpoint(path, student, projection, optimizer, scheduler, epoch, best_accuracy, history, config):
    torch.save({
        "model_state_dict": student.state_dict(),
        "projection_state_dict": projection.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_accuracy": best_accuracy,
        "classes": CLASS_NAMES,
        "class_to_idx": {name: index for index, name in enumerate(CLASS_NAMES)},
        "training_config": config,
        "metrics": history,
    }, path)


@dataclass(frozen=True)
class FeatureDistillationConfig:
    teacher_checkpoint: str
    data_dir: str
    output_dir: str
    teacher_embedding_dim: int
    student_embedding_dim: int
    ce_weight: float
    kd_weight: float
    feature_weight: float
    temperature: float
    label_smoothing: float
    learning_rates: dict[str, float]
    max_epochs: int
    early_stopping_patience: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-distill DINOv2 into FractalDB ResNet34.")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if (args.output_dir / "best.pt").exists():
        raise FileExistsError("Output directory already contains best.pt; use a fresh output directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.expanduser().resolve()
    base.student_api.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_checkpoint = base.load_checkpoint(args.teacher_checkpoint)
    teacher = base.load_teacher(args.teacher_checkpoint, teacher_checkpoint, device)
    teacher_dim = int(teacher_checkpoint["training_config"]["embedding_dim"])
    student_dim = 512
    train_raw = base.student_api.collect_samples(data_dir, TRAIN_SESSIONS, lambda image: image)
    val_raw = base.student_api.collect_samples(data_dir, VAL_SESSIONS, lambda image: image)
    train_dataset = base.PairedDataset(train_raw.samples, train=True)
    val_dataset = base.PairedDataset(val_raw.samples, train=False)
    train_sampler = base.student_api.BalancedBatchSampler(train_raw, seed=42)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    student = base.student_api.ResNet34WithEmbedding(4)
    base.student_api.load_fractaldb_backbone(student)
    base.student_api.freeze_first_stage(student)
    base.set_student_stage(student, 1)
    projection = nn.Linear(student_dim, teacher_dim)
    student.to(device)
    projection.to(device)
    optimizer = build_optimizer(student, projection)
    scheduler = base.ProgressiveLRScheduler(optimizer, MAX_EPOCHS, WARMUP_EPOCHS)

    # Preflight smoke test: dimensions, losses, gradients, frozen teacher, and stages.
    batch = next(iter(train_loader))
    teacher_before = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
    set_modes(student, projection)
    with torch.inference_mode():
        teacher_output = teacher(batch["teacher_image"].to(device))
    student_logits, student_embedding = student(batch["student_image"].to(device))
    projected = projection(student_embedding)
    smoke_loss, smoke_ce, smoke_kd, smoke_feature = feature_distillation_loss(
        student_logits, teacher_output["logits"], batch["label"].to(device), projected, teacher_output["embedding"]
    )
    optimizer.zero_grad(set_to_none=True)
    smoke_loss.backward()
    assert student_logits.shape == (batch["label"].shape[0], 4)
    assert student_embedding.shape[1] == student_dim and teacher_output["embedding"].shape[1] == teacher_dim
    assert smoke_loss.isfinite() and smoke_ce.isfinite() and smoke_kd.isfinite() and smoke_feature.isfinite()
    assert any(parameter.grad is not None for parameter in student.parameters() if parameter.requires_grad)
    assert any(parameter.grad is not None for parameter in projection.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(torch.equal(value, teacher.state_dict()[name]) for name, value in teacher_before.items())
    for check_epoch, expected in ((1, "classifier-only"), (2, "layer4+classifier"), (4, "layer2+layer3+layer4+stem+classifier")):
        assert base.set_student_stage(student, check_epoch) == expected
    print(f"Feature smoke test passed: student={student_dim}, teacher={teacher_dim}, logits={tuple(student_logits.shape)}, total={smoke_loss.item():.6f}, ce={smoke_ce.item():.6f}, kd={smoke_kd.item():.6f}, feature={smoke_feature.item():.6f}")
    if args.smoke_test:
        return

    # Reinitialize student and projection after smoke; the smoke backward pass is not training.
    student = base.student_api.ResNet34WithEmbedding(4)
    base.student_api.load_fractaldb_backbone(student)
    base.student_api.freeze_first_stage(student)
    base.set_student_stage(student, 1)
    projection = nn.Linear(student_dim, teacher_dim)
    student.to(device)
    projection.to(device)
    optimizer = build_optimizer(student, projection)
    scheduler = base.ProgressiveLRScheduler(optimizer, MAX_EPOCHS, WARMUP_EPOCHS)
    config = asdict(FeatureDistillationConfig(
        str(args.teacher_checkpoint.expanduser().resolve()), str(data_dir), str(args.output_dir.resolve()),
        teacher_dim, student_dim, CE_WEIGHT, KD_WEIGHT, FEATURE_WEIGHT, TEMPERATURE, LABEL_SMOOTHING,
        {"stem": STEM_LR, "layer2": LAYER2_LR, "layer3": LAYER3_LR, "layer4": LAYER4_LR, "classifier": CLASSIFIER_LR, "projection": CLASSIFIER_LR},
        MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
    ))
    history = []
    best_accuracy = float("-inf")
    stale_epochs = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        stage = base.set_student_stage(student, epoch)
        scheduler.step(epoch)
        train_metrics = run_epoch(student, projection, teacher, train_loader, optimizer, device, epoch, True)
        val_metrics = run_epoch(student, projection, teacher, val_loader, optimizer, device, epoch, False)
        learning_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        record = {"epoch": epoch, "stage": stage, "learning_rates": learning_rates, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        improved = val_metrics["accuracy"] > best_accuracy
        if improved:
            best_accuracy = val_metrics["accuracy"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(f"epoch={epoch:02d} stage={stage} train_acc={train_metrics['accuracy']:.4f} train_ce={train_metrics['ce_loss']:.4f} train_kd={train_metrics['kd_loss']:.4f} train_feature={train_metrics['feature_loss']:.4f} train_total={train_metrics['total_loss']:.4f} val_acc={val_metrics['accuracy']:.4f}")
        save_checkpoint(args.output_dir / "last.pt", student, projection, optimizer, scheduler, epoch, best_accuracy, history, config)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", student, projection, optimizer, scheduler, epoch, best_accuracy, history, config)
        (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
        (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
