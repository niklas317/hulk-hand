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
from onnx import helper, numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import ATTENTION_VARIANTS, build_model  # noqa: E402


IMAGE_SHAPE = (1, 3, 224, 224)
OPSET = 13
EMBEDDING_DIM = 384
GRAPH_VARIANTS = ("baseline", "gelu", "pow", "resize", "all")


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs["logits"]


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


def replace_gelu_modules(module: nn.Module) -> int:
    """Use PyTorch's tanh GELU approximation instead of Erf-based GELU."""

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GELU):
            setattr(module, name, nn.GELU(approximate="tanh"))
            replaced += 1
        else:
            replaced += replace_gelu_modules(child)
    return replaced


def build_export_model(checkpoint: dict, attention_variant: str, graph_variant: str) -> ExportWrapper:
    model = build_model(
        num_classes=int(checkpoint["num_classes"]),
        pretrained=False,
        attention_variant=attention_variant,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if "gelu" in graph_variant or graph_variant == "all":
        replaced = replace_gelu_modules(model)
        print(f"GELU modules replaced: {replaced}")

    model.eval().cpu()
    return ExportWrapper(model).eval().cpu()


def _constant_value(graph: onnx.GraphProto, name: str) -> float | None:
    for initializer in graph.initializer:
        if initializer.name == name:
            values = numpy_helper.to_array(initializer).reshape(-1)
            return float(values[0]) if values.size == 1 else None

    for node in graph.node:
        if node.op_type != "Constant" or not node.output or node.output[0] != name:
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                values = numpy_helper.to_array(attribute.t).reshape(-1)
                return float(values[0]) if values.size == 1 else None
    return None


def rewrite_pow_as_mul(model: onnx.ModelProto) -> int:
    rewritten = 0
    nodes = list(model.graph.node)
    replacement_nodes = []
    for node in nodes:
        if node.op_type != "Pow" or len(node.input) != 2:
            replacement_nodes.append(node)
            continue
        exponent = _constant_value(model.graph, node.input[1])
        if exponent is None or abs(exponent - 2.0) > 1e-6:
            replacement_nodes.append(node)
            continue
        replacement = helper.make_node(
            "Mul",
            inputs=[node.input[0], node.input[0]],
            outputs=list(node.output),
            name=node.name,
        )
        replacement_nodes.append(replacement)
        rewritten += 1

    del model.graph.node[:]
    model.graph.node.extend(replacement_nodes)
    return rewritten


def rewrite_resize_as_linear(model: onnx.ModelProto) -> int:
    rewritten = 0
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        for attribute in node.attribute:
            if attribute.name == "mode" and attribute.s.decode() == "cubic":
                attribute.s = b"linear"
                rewritten += 1
    return rewritten


def rewrite_graph(path: Path, graph_variant: str) -> tuple[int, int]:
    model = onnx.load(str(path))
    pow_count = 0
    resize_count = 0
    if "pow" in graph_variant or graph_variant == "all":
        pow_count = rewrite_pow_as_mul(model)
    if "resize" in graph_variant or graph_variant == "all":
        resize_count = rewrite_resize_as_linear(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return pow_count, resize_count


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
    expected = {"logits": [1, num_classes]}
    for name, shape in expected.items():
        actual = [dim.dim_value for dim in outputs[name].type.tensor_type.shape.dim]
        if actual != shape:
            raise RuntimeError(f"Unexpected {name} shape: {actual}")

    report = inspect_attention_graph(model)
    max_matmul_rank = {"original": 4, "split-qkv": 4, "flatten-heads": 3, "separate-heads": 2}[attention_variant]
    if attention_variant == "split-qkv" and report["attention_max_tensor_rank"] > 4:
        raise RuntimeError("split-qkv graph still contains a 5-D attention tensor")
    if attention_variant in ("flatten-heads", "separate-heads") and report["attention_matmul_ranks"]:
        if max(report["attention_matmul_ranks"]) > max_matmul_rank:
            raise RuntimeError(
                f"{attention_variant} graph contains an attention MatMul above rank {max_matmul_rank}: "
                f"{report['attention_matmul_ranks']}"
            )
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
) -> float:
    torch_logits = wrapper(dummy_input)
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    onnx_logits = session.run(
        ["logits"],
        {"input": dummy_input.numpy().astype(np.float32)},
    )[0]
    return float(np.max(np.abs(torch_logits.numpy() - onnx_logits)))


def compare_onnx_variants(paths: dict[str, Path], dummy_input: torch.Tensor) -> dict[str, float]:
    """Compare logits across already exported ONNX variants without exporting them."""
    outputs = {}
    for name, path in paths.items():
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        outputs[name] = session.run(["logits"], {input_name: dummy_input.numpy().astype(np.float32)})[0]
    reference_name = next(iter(outputs))
    reference = outputs[reference_name]
    return {
        name: float(np.max(np.abs(values - reference)))
        for name, values in outputs.items()
        if name != reference_name
    }


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Calculate accuracy for later PyTorch/ONNX validation comparisons."""
    if len(labels) == 0:
        return 0.0
    return float(np.mean(np.argmax(logits, axis=1) == labels))


def export_variant(
    checkpoint: dict,
    attention_variant: str,
    graph_variant: str,
    output_path: Path,
    seed: int,
) -> None:
    wrapper = build_export_model(checkpoint, attention_variant, graph_variant)
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
        output_names=["logits"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )

    pow_count, resize_count = rewrite_graph(temporary_path, graph_variant)
    graph_report = validate_graph(temporary_path, int(checkpoint["num_classes"]), attention_variant)
    torch.manual_seed(seed)
    dummy_input = torch.randn(IMAGE_SHAPE, dtype=torch.float32)
    logits_diff = compare_with_onnx(
        wrapper, temporary_path, dummy_input
    )
    temporary_path.replace(output_path)

    print(f"Exported: {output_path}")
    print(f"  Pow rewrites:    {pow_count}")
    print(f"  Resize rewrites: {resize_count}")
    print(f"  Logits max diff:  {logits_diff:.8f}")
    print(f"  Attention graph:  {graph_report}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the trained DINOv2 model to fixed-shape ONNX Opset 13."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--attention-variant",
        choices=ATTENTION_VARIANTS,
        default="original",
        help="Attention implementation used in the export graph.",
    )
    parser.add_argument(
        "--gelu-approximation",
        action="store_true",
        help="Replace exact Erf-based GELU with the tanh approximation.",
    )
    parser.add_argument(
        "--rewrite-pow",
        action="store_true",
        help="Replace Pow(x, 2) LayerNorm nodes with Mul(x, x).",
    )
    parser.add_argument(
        "--linear-resize",
        action="store_true",
        help="Replace cubic resize with linear resize; may reduce accuracy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    selected = []
    if args.gelu_approximation:
        selected.append("gelu")
    if args.rewrite_pow:
        selected.append("pow")
    if args.linear_resize:
        selected.append("resize")
    graph_variant = "_".join(selected) if selected else "baseline"
    if graph_variant == "gelu_pow_resize":
        graph_variant = "all"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_variant(checkpoint, args.attention_variant, graph_variant, args.output, args.seed)


if __name__ == "__main__":
    main()
