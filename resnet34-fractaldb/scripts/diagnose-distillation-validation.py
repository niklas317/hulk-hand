#!/usr/bin/env python3
"""Compare a DINOv2 teacher and ResNet34 student on identical S-V images.

This script only performs deterministic validation diagnosis. It never creates
an optimizer, modifies a checkpoint, or starts training.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
DINOV2_SCRIPTS = REPO_ROOT / "dinov2" / "scripts"
sys.path.insert(0, str(DINOV2_SCRIPTS))

from dataset import build_eval_transform as build_dino_eval_transform  # noqa: E402
from model import build_model as build_dino_model  # noqa: E402


def load_student_api():
    path = SCRIPT_DIR / "train-resnet34-fractaldb.py"
    spec = importlib.util.spec_from_file_location("resnet34_fractaldb_diagnose_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load student trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


student_api = load_student_api()
CLASS_NAMES = student_api.CLASS_NAMES
VAL_SESSIONS = student_api.VAL_SESSIONS
TEMPERATURE = 2.0


class PairedValidationDataset(Dataset):
    """One file record, two deterministic checkpoint-specific preprocessing paths."""

    def __init__(self, samples) -> None:
        if not samples:
            raise RuntimeError("No validation samples found in sessions S-V.")
        self.samples = samples
        self.student_transform = student_api.build_eval_transform(
            student_api.PreprocessConfig(image_size=224)
        )
        self.teacher_transform = build_dino_eval_transform(image_size=224)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            return {
                "student_image": self.student_transform(image),
                "teacher_image": self.teacher_transform(image),
                "label": sample.label,
                "class_name": sample.class_name,
                "session": sample.session,
                "path": str(sample.path),
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
        raise RuntimeError(f"Invalid checkpoint: {path}")
    return checkpoint


def validate_class_mapping(checkpoint: dict, name: str) -> None:
    expected = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
    if checkpoint.get("classes") != CLASS_NAMES or checkpoint.get("class_to_idx") != expected:
        raise RuntimeError(
            f"{name} class order mismatch. Expected {CLASS_NAMES} / {expected}; "
            f"got {checkpoint.get('classes')} / {checkpoint.get('class_to_idx')}"
        )
    if int(checkpoint.get("num_classes", len(CLASS_NAMES))) != len(CLASS_NAMES):
        raise RuntimeError(f"{name} must have exactly four output classes.")


def load_models(student_checkpoint: dict, teacher_checkpoint: dict, device: torch.device):
    validate_class_mapping(student_checkpoint, "Student checkpoint")
    validate_class_mapping(teacher_checkpoint, "Teacher checkpoint")
    teacher_config = teacher_checkpoint.get("training_config", {})
    if teacher_config.get("backbone") != "dinov2_vits14" or int(teacher_config.get("embedding_dim", -1)) != 384:
        raise RuntimeError("Teacher checkpoint is not the expected DINOv2 ViT-S/14 384-D model.")
    num_blocks = int(teacher_config.get("num_trainable_blocks", 10))
    teacher = build_dino_model(num_classes=4, pretrained=False, num_trainable_blocks=num_blocks)
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = student_api.ResNet34WithEmbedding(4)
    student.load_state_dict(student_checkpoint["model_state_dict"], strict=True)
    student.eval()
    teacher.to(device)
    student.to(device)
    return teacher, student


def accuracy_stats(correct: dict, total: dict) -> tuple[float, dict[str, float]]:
    per_class = {name: correct[name] / max(1, total[name]) for name in CLASS_NAMES}
    return sum(per_class.values()) / len(CLASS_NAMES), per_class


def print_model_stats(name: str, correct: dict, total: dict, session_correct: dict, session_total: dict) -> None:
    overall_correct = sum(correct.values())
    overall_total = sum(total.values())
    balanced, per_class = accuracy_stats(correct, total)
    print(f"\n{name} accuracy: {overall_correct / max(1, overall_total) * 100:.2f}% ({overall_correct}/{overall_total})")
    print(f"{name} balanced accuracy: {balanced * 100:.2f}%")
    print(f"{name} per-class accuracy")
    for class_name in CLASS_NAMES:
        print(f"  {class_name:<16} {per_class[class_name] * 100:6.2f}% ({correct[class_name]}/{total[class_name]})")
    print(f"{name} per-session accuracy")
    for session in sorted(session_total):
        print(f"  Session {session:<8} {session_correct[session] / max(1, session_total[session]) * 100:6.2f}% ({session_correct[session]}/{session_total[session]})")


def save_error_grid(errors: list[dict], output_path: Path, limit: int = 30) -> None:
    selected = errors[:limit]
    if not selected:
        print("No teacher-correct/student-wrong validation examples found; no grid written.")
        return
    columns, cell_width, cell_height = 5, 260, 250
    rows = (len(selected) + columns - 1) // columns
    grid = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for index, item in enumerate(selected):
        with Image.open(item["path"]) as image:
            image.thumbnail((cell_width - 12, 185))
            image = image.convert("RGB")
            x = (index % columns) * cell_width + (cell_width - image.width) // 2
            y = (index // columns) * cell_height + 4
            grid.paste(image, (x, y))
        text = (
            f"{item['session']} | true: {item['true']}\n"
            f"T: {item['teacher']} ({item['teacher_conf']:.1%})\n"
            f"S: {item['student']} ({item['student_conf']:.1%})"
        )
        draw.multiline_text(((index % columns) * cell_width + 6, (index // columns) * cell_height + 192), text, fill="black", font=font, spacing=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"Error grid written: {output_path} ({len(selected)} images)")


@torch.inference_mode()
def diagnose(teacher, student, loader, device: torch.device, grid_path: Path) -> None:
    teacher_correct = defaultdict(int)
    teacher_total = defaultdict(int)
    student_correct = defaultdict(int)
    student_total = defaultdict(int)
    teacher_session_correct = defaultdict(int)
    teacher_session_total = defaultdict(int)
    student_session_correct = defaultdict(int)
    student_session_total = defaultdict(int)
    agreement = defaultdict(int)
    agreement_total = defaultdict(int)
    agreement_session = defaultdict(int)
    agreement_session_total = defaultdict(int)
    no_gesture_confusion = 0
    total_kl = 0.0
    total_items = 0
    errors = []

    for batch in tqdm(loader, desc="Validation diagnosis"):
        teacher_logits = teacher(batch["teacher_image"].to(device, non_blocking=True))["logits"]
        student_logits, _ = student(batch["student_image"].to(device, non_blocking=True))
        labels = batch["label"].to(device, non_blocking=True)
        teacher_probs = F.softmax(teacher_logits, dim=1)
        student_probs = F.softmax(student_logits, dim=1)
        teacher_predictions = teacher_probs.argmax(1)
        student_predictions = student_probs.argmax(1)
        total_kl += float(F.kl_div(
            F.log_softmax(student_logits / TEMPERATURE, dim=1),
            F.softmax(teacher_logits / TEMPERATURE, dim=1),
            reduction="batchmean",
        )) * labels.size(0)
        total_items += labels.size(0)
        for index, (label, teacher_prediction, student_prediction) in enumerate(zip(labels.cpu().tolist(), teacher_predictions.cpu().tolist(), student_predictions.cpu().tolist())):
            class_name = CLASS_NAMES[label]
            session = batch["session"][index]
            teacher_total[class_name] += 1
            student_total[class_name] += 1
            teacher_correct[class_name] += int(label == teacher_prediction)
            student_correct[class_name] += int(label == student_prediction)
            teacher_session_total[session] += 1
            student_session_total[session] += 1
            teacher_session_correct[session] += int(label == teacher_prediction)
            student_session_correct[session] += int(label == student_prediction)
            agreement_total[class_name] += 1
            agreement[class_name] += int(teacher_prediction == student_prediction)
            agreement_session_total[session] += 1
            agreement_session[session] += int(teacher_prediction == student_prediction)
            if label != 3 and teacher_prediction == label and student_prediction == 3:
                no_gesture_confusion += 1
            if teacher_prediction == label and student_prediction != label:
                errors.append({
                    "path": batch["path"][index],
                    "session": session,
                    "true": class_name,
                    "teacher": CLASS_NAMES[teacher_prediction],
                    "student": CLASS_NAMES[student_prediction],
                    "teacher_conf": float(teacher_probs[index, teacher_prediction]),
                    "student_conf": float(student_probs[index, student_prediction]),
                    "priority": int(label != 3 and student_prediction == 3),
                })
    errors.sort(key=lambda item: (-item["priority"], item["session"], item["path"]))

    print_model_stats("Teacher", teacher_correct, teacher_total, teacher_session_correct, teacher_session_total)
    print_model_stats("Student", student_correct, student_total, student_session_correct, student_session_total)
    print(f"\nTeacher-student agreement: {sum(agreement.values()) / max(1, sum(agreement_total.values())) * 100:.2f}%")
    print("Agreement by class")
    for name in CLASS_NAMES:
        print(f"  {name:<16} {agreement[name] / max(1, agreement_total[name]) * 100:6.2f}% ({agreement[name]}/{agreement_total[name]})")
    print("Agreement by session")
    for session in sorted(agreement_session_total):
        print(f"  Session {session:<8} {agreement_session[session] / max(1, agreement_session_total[session]) * 100:6.2f}% ({agreement_session[session]}/{agreement_session_total[session]})")
    print(f"Mean temperature-scaled KL divergence (T={TEMPERATURE}): {total_kl / max(1, total_items):.6f}")
    print(f"Teacher-correct gesture -> student no_gesture: {no_gesture_confusion}")
    save_error_grid(errors, grid_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose DINOv2 teacher vs ResNet34 student on validation S-V.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--grid-output", type=Path, default=Path("distillation-validation-errors.png"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    student_checkpoint = load_checkpoint(args.student_checkpoint)
    teacher_checkpoint = load_checkpoint(args.teacher_checkpoint)
    samples = student_api.collect_samples(data_dir, VAL_SESSIONS, lambda image: image).samples
    dataset = PairedValidationDataset(samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher, student = load_models(student_checkpoint, teacher_checkpoint, device)
    print(f"Device: {device}; validation sessions: {', '.join(VAL_SESSIONS)}; samples: {len(dataset)}")
    print("Teacher preprocessing: DINOv2 deterministic Resize(224,224) + ImageNet normalization")
    print("Student preprocessing: FractalDB deterministic letterbox(224) + ImageNet normalization")
    diagnose(teacher, student, loader, device, args.grid_output.expanduser().resolve())


if __name__ == "__main__":
    main()
