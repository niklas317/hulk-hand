#!/usr/bin/env python3
"""Export a fine-tuned ResNet18 to dual-output ONNX and optional TFLite."""

from __future__ import annotations

import argparse
import inspect
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torchvision.models import resnet18


CLASS_NAMES = ["one", "two", "stop", "no_gesture"]
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


class ResNet18WithEmbedding(nn.Module):
    """ResNet18 that returns both class logits and the penultimate embedding."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor):
        # Run the backbone explicitly so the penultimate feature vector is exportable.
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        embedding = torch.flatten(x, 1)  # [batch, 512] for ResNet18
        logits = self.backbone.fc(embedding)  # [batch, num_classes]
        return logits, embedding


class DualOutputWrapper(nn.Module):
    """Deployment wrapper: export logits plus penultimate embedding."""

    def __init__(self, model: ResNet18WithEmbedding) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        logits, embedding = self.model(x)
        return logits, embedding


class LogitsOnlyWrapper(nn.Module):
    """Optional deployment wrapper: export only class logits."""

    def __init__(self, model: ResNet18WithEmbedding) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        return logits


def resolve_repo_path(value: str | None, default_path: Path) -> Path:
    """Resolve relative CLI paths from the repository rather than the shell cwd."""
    path = Path(value).expanduser() if value else default_path
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_checkpoint_state(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    """Extract a state dictionary from common PyTorch checkpoint layouts."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    for key in ("MODEL_STATE", "state_dict", "model_state_dict", "model", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if any(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint

    raise ValueError(
        f"Could not find a model state_dict in {checkpoint_path}. "
        "Expected one of: MODEL_STATE, state_dict, model_state_dict, model, net."
    )


def normalize_state_dict_for_backbone(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove wrapper prefixes left by distributed or nested model training."""
    normalized: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        name = key
        for prefix in ("module.", "backbone.", "model.", "net."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        normalized[name] = tensor
    return normalized


def read_class_names(path: Path, fallback: Sequence[str]) -> list[str]:
    """Read checkpoint labels while retaining the standard four-class fallback."""
    if not path.exists():
        return list(fallback)
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return names or list(fallback)


def load_model(checkpoint_path: Path, class_names: Sequence[str]) -> ResNet18WithEmbedding:
    """Build the deployment model, load compatible weights, and switch to eval mode."""
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    print(f"Class names: {list(class_names)}", flush=True)

    model = ResNet18WithEmbedding(num_classes=len(class_names))
    state_dict = normalize_state_dict_for_backbone(load_checkpoint_state(checkpoint_path))
    missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Warning: missing keys: {missing}", flush=True)
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}", flush=True)

    model.eval()
    return model


def export_onnx(
    model: ResNet18WithEmbedding,
    output_path: Path,
    image_size: int,
    batch_size: int,
    opset: int,
    dynamic_batch: bool,
    dual_output: bool,
) -> None:
    """Export either logits-only or dual-output inference with an optional batch axis."""
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dual_output:
        export_model = DualOutputWrapper(model).eval()
        output_names = ["logits", "embedding"]
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                "input": {0: "batch"},
                "logits": {0: "batch"},
                "embedding": {0: "batch"},
            }
        print("ONNX outputs: logits [batch, 4], embedding [batch, 512]", flush=True)
    else:
        export_model = LogitsOnlyWrapper(model).eval()
        output_names = ["logits"]
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}
        print("ONNX output: logits [batch, 4]", flush=True)

    print(f"Exporting ONNX opset={opset}: {output_path}", flush=True)
    print("Using legacy TorchScript ONNX exporter when available: dynamo=False", flush=True)

    export_kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Newer PyTorch versions use the torch.export/Dynamo ONNX exporter by default.
    # That path may internally emit opset 18 and then try to downgrade. For exact
    # opset 13 exports, the legacy TorchScript exporter is usually the safer path
    # for this simple ResNet18 graph.
    try:
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False
    except (TypeError, ValueError):
        pass

    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        **export_kwargs,
    )
    print(f"Saved ONNX: {output_path}", flush=True)


def verify_onnx_if_available(onnx_path: Path) -> None:
    """Run ONNX structural validation when the optional checker is installed."""
    try:
        import onnx  # type: ignore
    except Exception:
        print("ONNX package not installed; skipping onnx.checker verification.", flush=True)
        return

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    outputs = [output.name for output in model.graph.output]
    opsets = {opset.domain or "ai.onnx": opset.version for opset in model.opset_import}
    print(f"ONNX check: OK. Graph outputs: {outputs}", flush=True)
    print(f"ONNX opset imports: {opsets}", flush=True)


def convert_to_tflite_with_onnx2tf(onnx_path: Path, saved_model_dir: Path, tflite_path: Path) -> bool:
    """Convert ONNX -> TensorFlow SavedModel via onnx2tf, then SavedModel -> TFLite.

    The two ONNX outputs are preserved as two TensorFlow/TFLite outputs when the
    converter supports this graph. TFLite may rename or reorder output tensors
    internally, so inspect output names and shapes after conversion.
    """
    onnx2tf_bin = shutil.which("onnx2tf")
    if onnx2tf_bin is None:
        print("onnx2tf command not found; skipping TFLite conversion.", flush=True)
        print("Install if needed: python3 -m pip install onnx onnx2tf tensorflow", flush=True)
        return False

    print(f"Converting ONNX -> TensorFlow SavedModel: {saved_model_dir}", flush=True)
    saved_model_dir.parent.mkdir(parents=True, exist_ok=True)
    # Keep conversion external so the exporter can report a useful missing-tool error.
    subprocess.run(
        [onnx2tf_bin, "-i", str(onnx_path), "-o", str(saved_model_dir)],
        check=True,
    )

    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        print("TensorFlow not installed; SavedModel was created, but skipping .tflite export.", flush=True)
        print("Install if needed: python3 -m pip install tensorflow", flush=True)
        return False

    print(f"Converting TensorFlow SavedModel -> TFLite: {tflite_path}", flush=True)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = []
    tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    print(f"Saved TFLite: {tflite_path}", flush=True)

    try:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        output_details = interpreter.get_output_details()
        print("TFLite output tensors:", flush=True)
        for idx, detail in enumerate(output_details):
            print(f"  output {idx}: name={detail.get('name')} shape={detail.get('shape')}", flush=True)
    except Exception as exc:
        print(f"TFLite was saved, but output inspection failed: {exc}", flush=True)

    return True


def main() -> None:
    """Parse export options and produce the requested deployment artifacts."""
    parser = argparse.ArgumentParser(description="Export trained 4-class ResNet18 to ONNX opset 13 and optionally TFLite")
    parser.add_argument(
        "--checkpoint",
        default="resnet18/artifacts/ResNet18_finetuned.pth",
        help="Best trained checkpoint path, relative to the repository root by default",
    )
    parser.add_argument(
        "--output-dir",
        default="resnet18/artifacts/export",
        help="Output directory for ONNX/TFLite artifacts, relative to the repository root by default",
    )
    parser.add_argument("--onnx-name", default="ResNet18_4class_opset13_dual_output.onnx")
    parser.add_argument("--tflite-name", default="ResNet18_4class_dual_output.tflite")
    parser.add_argument("--saved-model-name", default="saved_model_dual_output")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset. Default: 13")
    parser.add_argument("--fixed-batch", action="store_true", help="Use fixed batch size instead of dynamic batch axis")
    parser.add_argument("--logits-only", action="store_true", help="Export only logits instead of logits + embedding")
    parser.add_argument("--skip-tflite", action="store_true", help="Only export ONNX")
    parser.add_argument("--skip-onnx-check", action="store_true", help="Skip optional ONNX checker")
    args = parser.parse_args()

    checkpoint_path = resolve_repo_path(args.checkpoint, REPO_ROOT / "resnet18" / "artifacts" / "ResNet18_finetuned.pth")
    output_dir = resolve_repo_path(args.output_dir, REPO_ROOT / "resnet18" / "artifacts" / "export")
    onnx_path = output_dir / args.onnx_name
    saved_model_dir = output_dir / args.saved_model_name
    tflite_path = output_dir / args.tflite_name

    class_names_path = checkpoint_path.parent / "class_names.txt"
    class_names = read_class_names(class_names_path, CLASS_NAMES)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("Start export", flush=True)
    print(f"Repo/script root: {SCRIPT_DIR}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Export mode: {'logits only' if args.logits_only else 'dual output: logits + embedding'}", flush=True)

    model = load_model(checkpoint_path, class_names)
    export_onnx(
        model=model,
        output_path=onnx_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        opset=args.opset,
        dynamic_batch=not args.fixed_batch,
        dual_output=not args.logits_only,
    )

    if not args.skip_onnx_check:
        verify_onnx_if_available(onnx_path)

    if not args.skip_tflite:
        convert_to_tflite_with_onnx2tf(onnx_path, saved_model_dir, tflite_path)

    print("Export finished", flush=True)
    print("Input:   float32 tensor [batch, 3, 224, 224]", flush=True)
    if args.logits_only:
        print("Output:  logits [batch, 4] in class_names.txt order", flush=True)
    else:
        print("Output0: logits [batch, 4] in class_names.txt order", flush=True)
        print("Output1: embedding [batch, 512] from the penultimate layer before fc", flush=True)
        print("TFLite note: tensor names and output order may change; identify logits by shape [batch, 4] and embedding by [batch, 512].", flush=True)
    print("Remember: preprocessing is not embedded; apply the same RGB/224/to-tensor/normalization before inference.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"External conversion command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr, flush=True)
        raise
