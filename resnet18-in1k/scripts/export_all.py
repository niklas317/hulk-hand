#!/usr/bin/env python3
"""Export a ResNet18-IN1K checkpoint to ONNX and TensorFlow Lite.

The TFLite conversion uses the external onnx2tf converter. Preprocessing is
not embedded in either artifact: callers must provide ImageNet-normalized
float32 RGB tensors with shape [1, 3, 224, 224].
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(SCRIPT_DIR))
import train  # noqa: E402


IMAGE_SHAPE = (1, 3, 224, 224)
OPSET = 13


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path.expanduser().resolve(), map_location="cpu", weights_only=False)
    expected_mapping = {name: index for index, name in enumerate(train.CLASS_NAMES)}
    if checkpoint.get("classes") != train.CLASS_NAMES or checkpoint.get("class_to_idx") != expected_mapping:
        raise RuntimeError("Checkpoint class mapping does not match the ResNet18-IN1K pipeline.")
    model = train.build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().cpu()
    return model


def export_onnx(model: torch.nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.onnx")
    if temporary.exists():
        temporary.unlink()
    dummy = torch.randn(IMAGE_SHAPE, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        str(temporary),
        export_params=True,
        input_names=["input"],
        output_names=["logits"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    import onnx

    graph = onnx.load(str(temporary))
    onnx.checker.check_model(graph)
    shape = [dimension.dim_value for dimension in graph.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [dimension.dim_value for dimension in graph.graph.output[0].type.tensor_type.shape.dim]
    if shape != list(IMAGE_SHAPE) or output_shape != [1, len(train.CLASS_NAMES)]:
        raise RuntimeError(f"Unexpected ONNX shapes: input={shape}, output={output_shape}")
    temporary.replace(output_path)
    print(f"ONNX exported: {output_path}")


def convert_tflite(onnx_path: Path, saved_model_dir: Path, tflite_path: Path) -> None:
    converter = shutil.which("onnx2tf")
    if converter is None:
        raise RuntimeError(
            "onnx2tf is required for TFLite export. Install optional dependencies with "
            "`python3 -m pip install -r resnet18-in1k/requirements-export.txt`."
        )
    if saved_model_dir.exists():
        raise FileExistsError(f"SavedModel directory already exists: {saved_model_dir}")
    saved_model_dir.parent.mkdir(parents=True, exist_ok=True)
    # onnx2tf downloads a calibration sample by default. Supplying a local
    # valid sample avoids network/cache failures and keeps export reproducible.
    calibration_name = "calibration_image_sample_data_20x128x128x3_float32.npy"
    with tempfile.TemporaryDirectory(prefix="resnet18-onnx2tf-") as work_dir:
        calibration_path = Path(work_dir) / calibration_name
        np.save(calibration_path, np.random.default_rng(0).standard_normal((20, 128, 128, 3), dtype=np.float32))
        subprocess.run(
            [converter, "-i", str(onnx_path), "-o", str(saved_model_dir), "-nuo", "-n"],
            check=True,
            cwd=work_dir,
        )

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required after onnx2tf conversion to create TFLite.") from exc

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tflite_path.with_suffix(".tmp.tflite")
    converter_options = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_bytes = converter_options.convert()
    temporary.write_bytes(tflite_bytes)

    interpreter = tf.lite.Interpreter(model_path=str(temporary))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise RuntimeError(f"Expected one TFLite input/output, got {len(input_details)}/{len(output_details)}")
    tflite_input_shape = list(input_details[0]["shape"])
    accepted_input_shapes = [list(IMAGE_SHAPE), [1, 224, 224, 3]]
    if tflite_input_shape not in accepted_input_shapes:
        raise RuntimeError(f"Unexpected TFLite input shape: {tflite_input_shape}")
    if list(output_details[0]["shape"]) != [1, len(train.CLASS_NAMES)]:
        raise RuntimeError(f"Unexpected TFLite output shape: {output_details[0]['shape']}")
    temporary.replace(tflite_path)
    print(f"TFLite exported: {tflite_path} (input shape {tflite_input_shape})")


def compare_onnx(model: torch.nn.Module, onnx_path: Path, seed: int) -> None:
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    sample = rng.standard_normal(IMAGE_SHAPE, dtype=np.float32)
    with torch.inference_mode():
        torch_logits = model(torch.from_numpy(sample)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"input": sample})[0]
    difference = float(np.max(np.abs(torch_logits - onnx_logits)))
    print(f"PyTorch/ONNX max logit difference: {difference:.8f}")
    if difference > 1e-4:
        raise RuntimeError(f"ONNX numerical comparison failed: max difference={difference}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the latest ResNet18-IN1K checkpoint to ONNX and TFLite.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt or last.pt from a ResNet18-IN1K run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--onnx-name", default="resnet18-in1k.onnx")
    parser.add_argument("--tflite-name", default="resnet18-in1k.tflite")
    parser.add_argument("--saved-model-name", default="resnet18-in1k-saved-model")
    parser.add_argument("--skip-tflite", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint)
    onnx_path = output_dir / args.onnx_name
    export_onnx(model, onnx_path)
    if not args.skip_compare:
        compare_onnx(model, onnx_path, args.seed)
    if not args.skip_tflite:
        convert_tflite(
            onnx_path,
            output_dir / args.saved_model_name,
            output_dir / args.tflite_name,
        )
    print(f"Checkpoint: {checkpoint}")
    print(f"ONNX input: float32 RGB ImageNet-normalized tensor {list(IMAGE_SHAPE)}")
    print("TFLite input: float32 tensor; onnx2tf may expose NHWC shape [1, 224, 224, 3]")
    print("Output: logits [1, 4] in gesture_1, gesture_2, gesture_3, no_gesture order")


if __name__ == "__main__":
    main()
