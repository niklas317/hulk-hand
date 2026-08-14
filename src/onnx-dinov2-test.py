#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before DINOv2 is loaded.
os.environ["XFORMERS_DISABLED"] = "1"

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from model import (
    EMBEDDING_DIM,
    IMAGE_SIZE,
    build_model,
)


OPSET_VERSION = 13

RTOL = 1e-4
ATOL = 1e-5


class OnnxWrapper(nn.Module):
    """
    Convert the normal hulk-hand dict output into two
    plain Tensor outputs for ONNX export.
    """

    def __init__(
        self,
        model: nn.Module,
    ) -> None:

        super().__init__()

        self.model = model

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        outputs = self.model(x)

        return (
            outputs["logits"],
            outputs["embedding"],
        )


def get_onnx_shape(
    value_info,
) -> list[int | str | None]:

    shape = []

    tensor_type = (
        value_info
        .type
        .tensor_type
    )

    for dimension in (
        tensor_type.shape.dim
    ):

        if dimension.HasField(
            "dim_value"
        ):
            shape.append(
                dimension.dim_value
            )

        elif dimension.HasField(
            "dim_param"
        ):
            shape.append(
                dimension.dim_param
            )

        else:
            shape.append(
                None
            )

    return shape


def verify_graph_interface(
    model: onnx.ModelProto,
    num_classes: int,
) -> None:

    if len(model.graph.input) != 1:
        raise RuntimeError(
            "Expected exactly one ONNX input."
        )

    if len(model.graph.output) != 2:
        raise RuntimeError(
            "Expected exactly two ONNX outputs."
        )

    input_info = model.graph.input[0]

    if input_info.name != "input":
        raise RuntimeError(
            "Unexpected ONNX input name: "
            f"{input_info.name}"
        )

    input_shape = get_onnx_shape(
        input_info
    )

    expected_input_shape = [
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ]

    if input_shape != expected_input_shape:
        raise RuntimeError(
            "Unexpected ONNX input shape.\n"
            f"Expected: {expected_input_shape}\n"
            f"Actual:   {input_shape}"
        )

    output_by_name = {
        output.name: output
        for output in model.graph.output
    }

    expected_outputs = {
        "logits": [
            1,
            num_classes,
        ],
        "embedding": [
            1,
            EMBEDDING_DIM,
        ],
    }

    if set(output_by_name) != set(
        expected_outputs
    ):
        raise RuntimeError(
            "Unexpected ONNX output names.\n"
            f"Expected: {sorted(expected_outputs)}\n"
            f"Actual:   {sorted(output_by_name)}"
        )

    for name, expected_shape in (
        expected_outputs.items()
    ):

        actual_shape = get_onnx_shape(
            output_by_name[name]
        )

        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Unexpected shape for '{name}'.\n"
                f"Expected: {expected_shape}\n"
                f"Actual:   {actual_shape}"
            )


def verify_opset(
    model: onnx.ModelProto,
) -> None:

    standard_opsets = [
        item.version
        for item in model.opset_import
        if item.domain in (
            "",
            "ai.onnx",
        )
    ]

    if len(standard_opsets) != 1:
        raise RuntimeError(
            "Expected exactly one standard ONNX "
            "opset import.\n"
            f"Found: {standard_opsets}"
        )

    actual_opset = (
        standard_opsets[0]
    )

    if actual_opset != OPSET_VERSION:
        raise RuntimeError(
            "Incorrect ONNX opset.\n"
            f"Expected: {OPSET_VERSION}\n"
            f"Actual:   {actual_opset}"
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Test DINOv2 ViT-S/14 export to "
            "fixed-shape ONNX Opset 13."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "models/"
            "dinov2_vits14_opset13_smoke.onnx"
        ),
        help=(
            "Final ONNX output path. "
            "The file is only created after all "
            "validation checks pass."
        ),
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help=(
            "Number of classifier outputs. "
            "Default: 4"
        ),
    )

    args = parser.parse_args()

    if args.num_classes <= 0:
        raise ValueError(
            "--num-classes must be > 0."
        )

    output_path = (
        args.output
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        output_path.stem
        + ".tmp"
        + output_path.suffix
    )

    if temporary_path.exists():
        temporary_path.unlink()

    # ----------------------------------------------------------
    # Header
    # ----------------------------------------------------------

    print()
    print("DINOv2 ONNX Opset 13 smoke test")
    print("===============================")
    print()

    print(
        f"PyTorch:       {torch.__version__}"
    )

    print(
        f"ONNX:          {onnx.__version__}"
    )

    print(
        f"ONNX Runtime:  {ort.__version__}"
    )

    print(
        f"Opset target:  {OPSET_VERSION}"
    )

    print(
        "Input:         "
        f"[1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]"
    )

    print(
        "Logits:        "
        f"[1, {args.num_classes}]"
    )

    print(
        "Embedding:     "
        f"[1, {EMBEDDING_DIM}]"
    )

    print(
        "xFormers:      disabled"
    )

    print(
        f"Output:        {output_path}"
    )

    print()

    # ----------------------------------------------------------
    # Build model on CPU
    #
    # Export on CPU intentionally. The resulting ONNX graph
    # contains no dependency on CUDA.
    # ----------------------------------------------------------

    print(
        "Loading DINOv2 ViT-S/14..."
    )

    model = build_model(
        num_classes=args.num_classes,
        pretrained=True,
    )

    model.cpu()
    model.eval()

    wrapper = OnnxWrapper(
        model
    )

    wrapper.cpu()
    wrapper.eval()

    # ----------------------------------------------------------
    # Deterministic input
    # ----------------------------------------------------------

    torch.manual_seed(
        42
    )

    dummy_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        dtype=torch.float32,
    )

    # ----------------------------------------------------------
    # PyTorch reference outputs
    # ----------------------------------------------------------

    print(
        "Running PyTorch reference..."
    )

    with torch.inference_mode():

        (
            torch_logits,
            torch_embedding,
        ) = wrapper(
            dummy_input
        )

    if not torch.isfinite(
        torch_logits
    ).all():
        raise FloatingPointError(
            "PyTorch logits contain "
            "non-finite values."
        )

    if not torch.isfinite(
        torch_embedding
    ).all():
        raise FloatingPointError(
            "PyTorch embedding contains "
            "non-finite values."
        )

    torch_logits_np = (
        torch_logits
        .detach()
        .cpu()
        .numpy()
    )

    torch_embedding_np = (
        torch_embedding
        .detach()
        .cpu()
        .numpy()
    )

    # ----------------------------------------------------------
    # ONNX export
    #
    # dynamo=False deliberately uses the classic exporter here.
    # We want a simple fixed-shape Opset-13 graph and then verify
    # the actual resulting file independently.
    # ----------------------------------------------------------

    print(
        "Exporting ONNX Opset 13..."
    )

    try:

        torch.onnx.export(
            wrapper,
            dummy_input,
            temporary_path,
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=[
                "input",
            ],
            output_names=[
                "logits",
                "embedding",
            ],
            dynamic_axes=None,
            dynamo=False,
        )

    except Exception:

        if temporary_path.exists():
            temporary_path.unlink()

        raise

    if not temporary_path.is_file():
        raise RuntimeError(
            "ONNX exporter did not create "
            "the temporary file."
        )

    # ----------------------------------------------------------
    # Load ONNX
    # ----------------------------------------------------------

    print(
        "Loading exported ONNX model..."
    )

    onnx_model = onnx.load(
        str(
            temporary_path
        )
    )

    # ----------------------------------------------------------
    # ONNX checker
    # ----------------------------------------------------------

    print(
        "Running ONNX checker..."
    )

    onnx.checker.check_model(
        onnx_model
    )

    # ----------------------------------------------------------
    # Verify actual Opset
    # ----------------------------------------------------------

    print(
        "Checking actual opset_import..."
    )

    verify_opset(
        onnx_model
    )

    # ----------------------------------------------------------
    # Verify fixed graph interface
    # ----------------------------------------------------------

    print(
        "Checking ONNX names and shapes..."
    )

    verify_graph_interface(
        model=onnx_model,
        num_classes=args.num_classes,
    )

    # ----------------------------------------------------------
    # ONNX Runtime
    # ----------------------------------------------------------

    print(
        "Loading model in ONNX Runtime..."
    )

    session = ort.InferenceSession(
        str(
            temporary_path
        ),
        providers=[
            "CPUExecutionProvider",
        ],
    )

    ort_inputs = (
        session.get_inputs()
    )

    ort_outputs = (
        session.get_outputs()
    )

    if len(ort_inputs) != 1:
        raise RuntimeError(
            "ONNX Runtime reports an "
            "unexpected input count."
        )

    if ort_inputs[0].name != "input":
        raise RuntimeError(
            "ONNX Runtime input name "
            "does not match."
        )

    output_names = [
        output.name
        for output in ort_outputs
    ]

    if output_names != [
        "logits",
        "embedding",
    ]:
        raise RuntimeError(
            "ONNX Runtime output names "
            "do not match.\n"
            f"Actual: {output_names}"
        )

    # ----------------------------------------------------------
    # ONNX Runtime inference
    # ----------------------------------------------------------

    print(
        "Running ONNX Runtime inference..."
    )

    input_numpy = (
        dummy_input
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    (
        ort_logits,
        ort_embedding,
    ) = session.run(
        [
            "logits",
            "embedding",
        ],
        {
            "input": input_numpy,
        },
    )

    if not np.isfinite(
        ort_logits
    ).all():
        raise FloatingPointError(
            "ONNX Runtime logits contain "
            "non-finite values."
        )

    if not np.isfinite(
        ort_embedding
    ).all():
        raise FloatingPointError(
            "ONNX Runtime embedding contains "
            "non-finite values."
        )

    # ----------------------------------------------------------
    # Numerical comparison
    # ----------------------------------------------------------

    print(
        "Comparing logits..."
    )

    np.testing.assert_allclose(
        ort_logits,
        torch_logits_np,
        rtol=RTOL,
        atol=ATOL,
    )

    print(
        "Comparing embedding..."
    )

    np.testing.assert_allclose(
        ort_embedding,
        torch_embedding_np,
        rtol=RTOL,
        atol=ATOL,
    )

    logits_max_error = float(
        np.max(
            np.abs(
                ort_logits
                - torch_logits_np
            )
        )
    )

    embedding_max_error = float(
        np.max(
            np.abs(
                ort_embedding
                - torch_embedding_np
            )
        )
    )

    # ----------------------------------------------------------
    # Only create final output after every check passed
    # ----------------------------------------------------------

    if output_path.exists():
        output_path.unlink()

    temporary_path.replace(
        output_path
    )

    # ----------------------------------------------------------
    # Re-open final file
    # ----------------------------------------------------------

    final_model = onnx.load(
        str(
            output_path
        )
    )

    onnx.checker.check_model(
        final_model
    )

    verify_opset(
        final_model
    )

    verify_graph_interface(
        model=final_model,
        num_classes=args.num_classes,
    )

    # ----------------------------------------------------------
    # Result
    # ----------------------------------------------------------

    print()
    print("Results")
    print("-------")

    print(
        "ONNX export:         PASS"
    )

    print(
        "ONNX checker:        PASS"
    )

    print(
        "Fixed input shape:   PASS"
    )

    print(
        "Output shapes:       PASS"
    )

    print(
        "ONNX Runtime:        PASS"
    )

    print(
        "Numerical logits:    PASS"
    )

    print(
        "Numerical embedding: PASS"
    )

    print(
        "Opset 13:            PASS"
    )

    print()

    print(
        "Max logits error:    "
        f"{logits_max_error:.10f}"
    )

    print(
        "Max embedding error: "
        f"{embedding_max_error:.10f}"
    )

    print()

    print(
        f"Final model: {output_path}"
    )

    print()
    print("=======================")
    print("ONNX SMOKE TEST PASSED")
    print("=======================")
    print()


if __name__ == "__main__":
    main()