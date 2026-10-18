#!/usr/bin/env python3

"""Export trained hulk-hand checkpoints through compatibility variants.

The checkpoint and training model are kept unchanged. Compatibility changes
are applied only to the export graph so each result can be tested by the
existing, fixed quantization pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import build_model  # noqa: E402


IMAGE_SHAPE = (1, 3, 224, 224)
OPSET = 13
EMBEDDING_DIM = 384
class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(x)
        return outputs["logits"], outputs["embedding"]


def load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(
            path.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path.expanduser().resolve(), map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid checkpoint format.")
    required = {"model_state_dict", "num_classes", "classes"}
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(f"Checkpoint is missing keys: {sorted(missing)}")
    return checkpoint


def build_export_model(checkpoint: dict, attention_variant: str) -> ExportWrapper:
    model = build_model(
        num_classes=int(checkpoint["num_classes"]),
        pretrained=False,
        attention_variant=attention_variant,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    model.eval().cpu()
    return ExportWrapper(model).eval().cpu()


def validate_graph(path: Path, num_classes: int, attention_variant: str) -> dict:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    opsets = [
        int(item.version)
        for item in model.opset_import
        if item.domain in ("", "ai.onnx")
    ]
    if opsets != [OPSET]:
        raise RuntimeError(f"Expected Opset 13, found {opsets}")

    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    if input_shape != list(IMAGE_SHAPE):
        raise RuntimeError(f"Unexpected input shape: {input_shape}")

    outputs = {output.name: output for output in model.graph.output}
    expected = {
        "logits": [1, num_classes],
        "embedding": [1, EMBEDDING_DIM],
    }
    for name, shape in expected.items():
        actual = [dim.dim_value for dim in outputs[name].type.tensor_type.shape.dim]
        if actual != shape:
            raise RuntimeError(f"Unexpected {name} shape: {actual}")

    report = inspect_attention_graph(model)
    max_matmul_rank = 4
    rank_warnings = []
    if attention_variant == "split-qkv" and report["attention_max_tensor_rank"] > 4:
        rank_warnings.append("split-qkv still exposes a tensor above rank 4")
    if attention_variant == "split-qkv" and report["attention_matmul_ranks"]:
        if max(report["attention_matmul_ranks"]) > max_matmul_rank:
            rank_warnings.append(
                f"attention MatMul ranks exceed the target rank {max_matmul_rank}: "
                f"{report['attention_matmul_ranks']}"
            )
    report["rank_warnings"] = rank_warnings
    return report


def inspect_attention_graph(model: onnx.ModelProto) -> dict:
    """Return operator counts and inferred ranks for attention nodes."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    shape_map = {}
    values = list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)
    for value in values:
        dimensions = value.type.tensor_type.shape.dim
        if dimensions:
            shape_map[value.name] = [dimension.dim_value for dimension in dimensions]
    attention_nodes = [
        node for node in inferred.graph.node
        if "attn" in node.name.lower() or "attention" in node.name.lower()
    ]
    matmul_ranks = []
    max_tensor_rank = 0
    for node in attention_nodes:
        for value_name in list(node.input) + list(node.output):
            max_tensor_rank = max(max_tensor_rank, len(shape_map.get(value_name, [])))
        if node.op_type == "MatMul":
            matmul_ranks.extend(
                len(shape_map.get(value_name, []))
                for value_name in node.input
                if shape_map.get(value_name)
            )
    return {
        "attention_node_count": len(attention_nodes),
        "attention_max_tensor_rank": max_tensor_rank,
        "attention_matmul_ranks": matmul_ranks,
        "attention_matmul_details": [
            {
                "name": node.name,
                "input_ranks": [len(shape_map.get(value_name, [])) for value_name in node.input],
                "output_ranks": [len(shape_map.get(value_name, [])) for value_name in node.output],
            }
            for node in attention_nodes
            if node.op_type == "MatMul"
        ],
        "op_counts": {
            op: sum(node.op_type == op for node in inferred.graph.node)
            for op in sorted({node.op_type for node in inferred.graph.node})
        },
    }


@torch.inference_mode()
def compare_with_onnx(
    wrapper: ExportWrapper,
    path: Path,
    dummy_input: torch.Tensor,
) -> tuple[float, float]:
    torch_logits, torch_embedding = wrapper(dummy_input)
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    onnx_logits, onnx_embedding = session.run(
        ["logits", "embedding"],
        {"input": dummy_input.numpy().astype(np.float32)},
    )
    return (
        float(np.max(np.abs(torch_logits.numpy() - onnx_logits))),
        float(np.max(np.abs(torch_embedding.numpy() - onnx_embedding))),
    )


def compare_onnx_variants(paths: dict[str, Path], dummy_input: torch.Tensor) -> dict[str, float]:
    """Compare logits across already exported ONNX variants without exporting them."""
    outputs = {}
    for name, path in paths.items():
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        outputs[name] = session.run(
            ["logits", "embedding"],
            {input_name: dummy_input.numpy().astype(np.float32)},
        )
    reference_name = next(iter(outputs))
    reference_logits, reference_embedding = outputs[reference_name]
    differences = {}
    for name, (logits, embedding) in outputs.items():
        if name != reference_name:
            differences[f"{name}:logits"] = float(np.max(np.abs(logits - reference_logits)))
            differences[f"{name}:embedding"] = float(np.max(np.abs(embedding - reference_embedding)))
    return differences


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Calculate accuracy for later PyTorch/ONNX validation comparisons."""
    if len(labels) == 0:
        return 0.0
    return float(np.mean(np.argmax(logits, axis=1) == labels))


def export_variant(
    checkpoint: dict,
    attention_variant: str,
    output_path: Path,
    seed: int,
) -> None:
    wrapper = build_export_model(checkpoint, attention_variant)
    dummy_input = torch.randn(IMAGE_SHAPE, dtype=torch.float32)
    temporary_path = output_path.with_suffix(".tmp.onnx")
    if temporary_path.exists():
        temporary_path.unlink()

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(temporary_path),
        export_params=True,
        input_names=["input"],
        output_names=["logits", "embedding"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )

    graph_report = validate_graph(temporary_path, int(checkpoint["num_classes"]), attention_variant)
    torch.manual_seed(seed)
    dummy_input = torch.randn(IMAGE_SHAPE, dtype=torch.float32)
    logits_diff, embedding_diff = compare_with_onnx(
        wrapper, temporary_path, dummy_input
    )
    if logits_diff > 1e-4 or embedding_diff > 1e-4:
        raise RuntimeError(
            f"{attention_variant} is not numerically equivalent: "
            f"logits max diff {logits_diff:.8f}, embedding max diff {embedding_diff:.8f}"
        )
    temporary_path.replace(output_path)

    print(f"Exported: {output_path}")
    print(f"  Logits max diff:  {logits_diff:.8f}")
    print(f"  Embed max diff:   {embedding_diff:.8f}")
    print(f"  Attention graph:  {graph_report}")
    if graph_report["rank_warnings"]:
        print("  Attention rank warnings: " + " | ".join(graph_report["rank_warnings"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the trained DINOv2 model to fixed-shape ONNX Opset 13."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--qkv-split",
        action="store_true",
        help="Use split Q/K/V attention instead of the standard combined-QKV export.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output.exists() and args.output.is_dir():
        raise ValueError("--output must be a file path, for example ~/Downloads/dinov2_original.onnx")

    checkpoint = load_checkpoint(args.checkpoint)
    attention_variant = "split-qkv" if args.qkv_split else "original"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_variant(checkpoint, attention_variant, args.output, args.seed)


if __name__ == "__main__":
    main()
