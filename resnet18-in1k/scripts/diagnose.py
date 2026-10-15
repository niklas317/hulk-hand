#!/usr/bin/env python3
"""Session-aware ResNet18-IN1K validation diagnosis with Grad-CAM.

This script does not train or alter checkpoints. It records logits, session
metrics, input statistics, and layer4 Grad-CAM grids for validation errors.
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
TRAIN_PATH = SCRIPT_DIR / "train.py"
spec = importlib.util.spec_from_file_location("resnet18_in1k_train", TRAIN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load training module: {TRAIN_PATH}")
train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train
spec.loader.exec_module(train)


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    expected = {name: index for index, name in enumerate(train.CLASS_NAMES)}
    if checkpoint.get("classes") != train.CLASS_NAMES or checkpoint.get("class_to_idx") != expected:
        raise RuntimeError("Checkpoint class mapping does not match ResNet18-IN1K.")
    return checkpoint


def font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def save_gradcam_group(model, dataset, items: list[dict], output_path: Path, device: torch.device) -> None:
    selected = items[:5]
    if not selected:
        return
    width, height = 224, 270
    grid = Image.new("RGB", (width * 3, height * len(selected)), "white")
    draw = ImageDraw.Draw(grid)
    text_font = font()
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))

    handle = model.layer4.register_forward_hook(hook)
    try:
        for row, item in enumerate(selected):
            batch = dataset[item["index"]]
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = model(batch["image"].unsqueeze(0).to(device))
                logits[0, item["prediction"]].backward()
            activation = activations["value"]
            gradient = gradients["value"]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1).relu()[0]
            cam = F.interpolate(cam[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
            cam = (cam - cam.min()) / (cam.max() - cam.min()).clamp_min(1e-8)
            values = (cam.detach().cpu().clamp(0, 1) * 255).byte()
            heatmap = Image.fromarray(torch.stack((values, torch.zeros_like(values), 255 - values), dim=-1).numpy(), mode="RGB")
            with Image.open(item["path"]) as source:
                original = source.convert("RGB").resize((224, 224))
            overlay = Image.blend(original, heatmap, 0.45)
            y = row * height
            grid.paste(original, (0, y))
            grid.paste(heatmap, (width, y))
            grid.paste(overlay, (width * 2, y))
            draw.text((4, y + 228), "Original", fill="black", font=text_font)
            draw.text((width + 4, y + 228), "Grad-CAM layer4", fill="black", font=text_font)
            draw.multiline_text((width * 2 + 4, y + 228), f"true: {item['true']}\npred: {item['prediction_name']} ({item['confidence']:.1%})", fill="black", font=text_font, spacing=2)
    finally:
        handle.remove()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


@torch.inference_mode()
def collect_outputs(model, loader, device):
    model.eval()
    logits = []
    labels = []
    paths = []
    sessions = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    session_correct = defaultdict(int)
    session_total = defaultdict(int)
    confusion = torch.zeros((len(train.CLASS_NAMES), len(train.CLASS_NAMES)), dtype=torch.long)
    errors = defaultdict(list)
    input_stats = defaultdict(lambda: {"count": 0, "mean_sum": torch.zeros(3), "std_sum": torch.zeros(3)})

    for batch in tqdm(loader, desc="ResNet18 validation diagnosis"):
        batch_logits = model(batch["image"].to(device, non_blocking=True))
        predictions = batch_logits.argmax(1).cpu()
        batch_labels = batch["label"].cpu()
        logits.append(batch_logits.cpu())
        labels.append(batch_labels)
        paths.extend(batch["path"])
        sessions.extend(batch["session"])
        for index, (label, prediction) in enumerate(zip(batch_labels.tolist(), predictions.tolist())):
            name = train.CLASS_NAMES[label]
            session = batch["session"][index]
            class_total[name] += 1
            class_correct[name] += int(label == prediction)
            session_total[session] += 1
            session_correct[session] += int(label == prediction)
            confusion[label, prediction] += 1
            if label != prediction:
                errors[(name, session)].append({
                    "index": len(paths) - len(batch_labels) + index,
                    "path": batch["path"][index],
                    "true": name,
                    "prediction": prediction,
                    "prediction_name": train.CLASS_NAMES[prediction],
                    "confidence": float(F.softmax(batch_logits[index], dim=0)[prediction]),
                })
            image = batch["image"][index].float()
            stats = input_stats[session]
            stats["count"] += 1
            stats["mean_sum"] += image.mean(dim=(1, 2))
            stats["std_sum"] += image.std(dim=(1, 2))
    total = sum(class_total.values())
    per_class = {name: class_correct[name] / max(1, class_total[name]) for name in train.CLASS_NAMES}
    metrics = {
        "accuracy": sum(class_correct.values()) / max(1, total),
        "balanced_accuracy": sum(per_class.values()) / len(train.CLASS_NAMES),
        "per_class_accuracy": per_class,
        "per_session_accuracy": {session: session_correct[session] / max(1, session_total[session]) for session in sorted(session_total)},
        "confusion_matrix": confusion.tolist(),
        "input_statistics_by_session": {
            session: {
                "mean": (stats["mean_sum"] / max(1, stats["count"])).tolist(),
                "std": (stats["std_sum"] / max(1, stats["count"])).tolist(),
            }
            for session, stats in input_stats.items()
        },
        "logits": torch.cat(logits),
        "labels": torch.cat(labels),
        "paths": paths,
        "sessions": sessions,
    }
    return metrics, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose ResNet18-IN1K validation errors by class and session.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(args.checkpoint)
    dataset = train.collect_samples(args.data_dir.expanduser().resolve(), train.VAL_SESSIONS, train.build_eval_transform(train.PreprocessConfig()))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    model = train.build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics, errors = collect_outputs(model, loader, device)
    logits = metrics.pop("logits")
    labels = metrics.pop("labels")
    metrics.pop("paths")
    metrics.pop("sessions")
    torch.save({"logits": logits, "labels": labels}, output_dir / "validation_logits.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")
    for name, value in metrics["per_class_accuracy"].items():
        print(f"{name:<16} {value * 100:6.2f}%")
    for session, value in metrics["per_session_accuracy"].items():
        print(f"Session {session:<8} {value * 100:6.2f}%")
    missing = []
    for name in train.CLASS_NAMES:
        for session in train.VAL_SESSIONS:
            group = errors.get((name, session), [])
            if len(group) < 5:
                missing.append(f"{name}/session_{session} ({len(group)}/5)")
            save_gradcam_group(model, dataset, group, output_dir / "gradcam" / f"{name}_session_{session}.png", device)
    if missing:
        print("Groups with fewer than five errors: " + ", ".join(missing))
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
