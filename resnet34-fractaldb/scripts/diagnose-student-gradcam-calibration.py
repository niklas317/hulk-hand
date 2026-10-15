#!/usr/bin/env python3
"""Diagnose a ResNet34 student on S-V with Grad-CAM and logit calibration.

This script never trains, creates an optimizer, or modifies checkpoints.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DIAGNOSE_PATH = SCRIPT_DIR / "diagnose-distillation-validation.py"
spec = importlib.util.spec_from_file_location("distillation_validation_diagnose", DIAGNOSE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load diagnosis module: {DIAGNOSE_PATH}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

CLASS_NAMES = base.CLASS_NAMES
VAL_SESSIONS = base.VAL_SESSIONS


def make_heatmap(cam: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    cam = F.interpolate(cam[None, None], size=(size[1], size[0]), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min()).clamp_min(1e-8)
    values = (cam.cpu().clamp(0, 1) * 255).byte()
    # Blue-to-red heatmap without requiring matplotlib.
    red = values
    green = torch.zeros_like(values)
    blue = 255 - values
    rgb = torch.stack((red, green, blue), dim=-1).numpy()
    return Image.fromarray(rgb, mode="RGB")


def font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def save_gradcam_group(items: list[dict], output_path: Path, student, dataset, device: torch.device) -> None:
    """Write original, heatmap, and overlay columns for one class/session group."""
    selected = items[:5]
    if not selected:
        return
    cell_width, cell_height = 224, 270
    grid = Image.new("RGB", (cell_width * 3, cell_height * len(selected)), "white")
    draw = ImageDraw.Draw(grid)
    text_font = font()
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda gradient: gradients.__setitem__("value", gradient))

    handle = student.backbone.layer4.register_forward_hook(forward_hook)
    try:
        for row, item in enumerate(selected):
            sample_index = item["index"]
            batch = dataset[sample_index]
            image_tensor = batch["student_image"].unsqueeze(0).to(device)
            student.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits, _ = student(image_tensor)
                target = logits[0, item["prediction"]]
                target.backward()
            activation = activations["value"]
            gradient = gradients["value"]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1).relu()[0].detach().cpu()
            with Image.open(item["path"]) as source:
                original = source.convert("RGB").resize((cell_width, 224))
            heatmap = make_heatmap(cam, original.size)
            overlay = Image.blend(original, heatmap, alpha=0.45)
            y = row * cell_height
            grid.paste(original, (0, y))
            grid.paste(heatmap, (cell_width, y))
            grid.paste(overlay, (cell_width * 2, y))
            draw.text((4, y + 228), "Original", fill="black", font=text_font)
            draw.text((cell_width + 4, y + 228), "Grad-CAM layer4", fill="black", font=text_font)
            draw.multiline_text(
                (cell_width * 2 + 4, y + 228),
                f"true: {item['true']}\npred: {item['prediction_name']} ({item['confidence']:.1%})",
                fill="black",
                font=text_font,
                spacing=2,
            )
    finally:
        handle.remove()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


@torch.inference_mode()
def collect_student_outputs(student, loader, device: torch.device) -> tuple[dict, list[dict]]:
    logits = []
    labels = []
    paths = []
    sessions = []
    wrong_by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    correct = total = 0
    for batch in tqdm(loader, desc="Student validation"):
        student_logits, _ = student(batch["student_image"].to(device, non_blocking=True))
        probabilities = F.softmax(student_logits, dim=1)
        predictions = probabilities.argmax(1).cpu()
        batch_labels = batch["label"].cpu()
        logits.append(student_logits.cpu())
        labels.append(batch_labels)
        paths.extend(batch["path"])
        sessions.extend(batch["session"])
        correct += int(predictions.eq(batch_labels).sum())
        total += len(batch_labels)
        for index, (label, prediction) in enumerate(zip(batch_labels.tolist(), predictions.tolist())):
            if label != prediction:
                confidence = float(probabilities[index, prediction])
                wrong_by_group[(CLASS_NAMES[label], batch["session"][index])].append({
                    "index": len(paths) - len(batch_labels) + index,
                    "path": batch["path"][index],
                    "true": CLASS_NAMES[label],
                    "prediction_name": CLASS_NAMES[prediction],
                    "prediction": prediction,
                    "confidence": confidence,
                })
    output = {
        "logits": torch.cat(logits),
        "labels": torch.cat(labels),
        "paths": paths,
        "sessions": sessions,
        "accuracy": correct / max(1, total),
    }
    return output, wrong_by_group


def print_calibration(outputs: dict, output_path: Path, minimum: float, maximum: float, step: float) -> None:
    logits = outputs["logits"]
    labels = outputs["labels"]
    rows = []
    bias = minimum
    while bias <= maximum + step * 0.01:
        adjusted = logits.clone()
        adjusted[:, CLASS_NAMES.index("no_gesture")] += bias
        predictions = adjusted.argmax(1)
        accuracy = float(predictions.eq(labels).float().mean())
        recalls = []
        for class_index in range(len(CLASS_NAMES)):
            mask = labels == class_index
            recalls.append(float(predictions[mask].eq(labels[mask]).float().mean()) if mask.any() else 0.0)
        rows.append({"bias": bias, "accuracy": accuracy, "balanced_accuracy": sum(recalls) / len(recalls), "per_class_accuracy": recalls})
        bias += step
    best_accuracy = max(rows, key=lambda row: (row["accuracy"], row["balanced_accuracy"]))
    best_balanced = max(rows, key=lambda row: (row["balanced_accuracy"], row["accuracy"]))
    print("\nNo-gesture logit-bias calibration (diagnostic only)")
    print(f"Original accuracy: {outputs['accuracy'] * 100:.2f}%")
    print(f"Best accuracy: bias={best_accuracy['bias']:.3f}, accuracy={best_accuracy['accuracy'] * 100:.2f}%, balanced={best_accuracy['balanced_accuracy'] * 100:.2f}%")
    print(f"Best balanced accuracy: bias={best_balanced['bias']:.3f}, accuracy={best_balanced['accuracy'] * 100:.2f}%, balanced={best_balanced['balanced_accuracy'] * 100:.2f}%")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"original_accuracy": outputs["accuracy"], "grid": rows, "best_accuracy": best_accuracy, "best_balanced_accuracy": best_balanced}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM and logit calibration diagnosis for the ResNet34 student.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bias-min", type=float, default=-3.0)
    parser.add_argument("--bias-max", type=float, default=3.0)
    parser.add_argument("--bias-step", type=float, default=0.05)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint = base.load_checkpoint(args.student_checkpoint)
    base.validate_class_mapping(checkpoint, "Student checkpoint")
    samples = base.student_api.collect_samples(args.data_dir.expanduser().resolve(), VAL_SESSIONS, lambda image: image).samples
    dataset = base.PairedValidationDataset(samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = base.student_api.ResNet34WithEmbedding(4)
    student.load_state_dict(checkpoint["model_state_dict"], strict=True)
    student.to(device).eval()
    outputs, wrong_by_group = collect_student_outputs(student, loader, device)
    logits_path = output_dir / "validation_logits.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, logits_path)
    print(f"Saved validation logits: {logits_path}")
    missing = []
    for class_name in CLASS_NAMES:
        for session in VAL_SESSIONS:
            group = wrong_by_group.get((class_name, session), [])
            if len(group) < 5:
                missing.append(f"{class_name}/session_{session} ({len(group)}/5)")
            save_gradcam_group(group, output_dir / "gradcam" / f"{class_name}_session_{session}.png", student, dataset, device)
    if missing:
        print("WARNING: fewer than five student errors available for: " + ", ".join(missing))
    print_calibration(outputs, output_dir / "no_gesture_bias_calibration.json", args.bias_min, args.bias_max, args.bias_step)


if __name__ == "__main__":
    main()
