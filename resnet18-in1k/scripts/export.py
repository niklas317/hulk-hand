#!/usr/bin/env python3
"""Export a trained ResNet18-IN1K checkpoint to fixed-shape ONNX."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = SCRIPT_DIR / "train.py"
spec = importlib.util.spec_from_file_location("resnet18_in1k_train", TRAIN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load training module: {TRAIN_PATH}")
train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train
spec.loader.exec_module(train)

IMAGE_SHAPE = (1, 3, 224, 224)
OPSET = 13


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    expected = {name: index for index, name in enumerate(train.CLASS_NAMES)}
    if checkpoint.get("classes") != train.CLASS_NAMES or checkpoint.get("class_to_idx") != expected:
        raise RuntimeError("Checkpoint class mapping does not match the ResNet18-IN1K pipeline.")
    return checkpoint


def validate_graph(path: Path) -> None:
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    opsets = [int(item.version) for item in graph.opset_import if item.domain in ("", "ai.onnx")]
    if opsets != [OPSET]:
        raise RuntimeError(f"Expected Opset 13, found {opsets}")
    input_shape = [dimension.dim_value for dimension in graph.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [dimension.dim_value for dimension in graph.graph.output[0].type.tensor_type.shape.dim]
    if input_shape != list(IMAGE_SHAPE) or output_shape != [1, 4]:
        raise RuntimeError(f"Unexpected graph shapes: input={input_shape}, output={output_shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ResNet18-IN1K as fixed-shape ONNX Opset 13.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", action="store_true", help="Compare one random input with ONNX Runtime.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and args.output.is_dir():
        raise ValueError("--output must be a file path, not a directory")
    checkpoint = load_checkpoint(args.checkpoint)
    model = train.build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.onnx")
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
    validate_graph(temporary)
    if args.compare:
        np.random.seed(args.seed)
        comparison_input = np.random.randn(*IMAGE_SHAPE).astype(np.float32)
        with torch.inference_mode():
            torch_logits = model(torch.from_numpy(comparison_input)).numpy()
        session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
        onnx_logits = session.run(["logits"], {"input": comparison_input})[0]
        print(f"PyTorch/ONNX max logit difference: {np.max(np.abs(torch_logits - onnx_logits)):.8f}")
    temporary.replace(args.output)
    print(f"Exported: {args.output}")


if __name__ == "__main__":
    main()
