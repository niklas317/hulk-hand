#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'onnx'. "
        "Install it with: pip install onnx"
    ) from exc

try:
    import onnxruntime as ort
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'onnxruntime'. "
        "Install it with: pip install onnxruntime"
    ) from exc

from model import HulkHandResNet18


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ONNX_OPSET = 13

INPUT_SHAPE = (
    1,
    3,
    224,
    224,
)

EMBEDDING_DIM = 512

RTOL = 1e-4
ATOL = 1e-5


# ---------------------------------------------------------------------------
# Export wrapper
# ---------------------------------------------------------------------------

class ONNXExportWrapper(nn.Module):
    """
    Convert the model's dictionary output into a fixed tuple:

        logits
        embedding

    This gives ONNX an explicit and stable output order.
    """

    def __init__(
        self,
        model: HulkHandResNet18,
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


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: Path,
) -> dict:

    checkpoint_path = (
        checkpoint_path
        .expanduser()
        .resolve()
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Invalid checkpoint format."
        )

    required_keys = {
        "model_state_dict",
        "num_classes",
        "classes",
    }

    missing = (
        required_keys
        - checkpoint.keys()
    )

    if missing:
        raise RuntimeError(
            "Checkpoint is missing required keys: "
            f"{sorted(missing)}"
        )

    num_classes = int(
        checkpoint["num_classes"]
    )

    classes = checkpoint["classes"]

    if len(classes) != num_classes:
        raise RuntimeError(
            "Checkpoint class count is inconsistent."
        )

    return checkpoint


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_export_model(
    checkpoint: dict,
) -> ONNXExportWrapper:

    num_classes = int(
        checkpoint["num_classes"]
    )

    model = HulkHandResNet18(
        num_classes=num_classes
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()
    model.cpu()

    wrapper = ONNXExportWrapper(
        model
    )

    wrapper.eval()
    wrapper.cpu()

    return wrapper


# ---------------------------------------------------------------------------
# ONNX shape utilities
# ---------------------------------------------------------------------------

def get_value_info_shape(
    value_info,
) -> list[int | str | None]:

    tensor_type = (
        value_info
        .type
        .tensor_type
    )

    shape: list[
        int | str | None
    ] = []

    for dimension in (
        tensor_type.shape.dim
    ):

        if dimension.HasField(
            "dim_value"
        ):
            shape.append(
                int(
                    dimension.dim_value
                )
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


# ---------------------------------------------------------------------------
# Structural ONNX checks
# ---------------------------------------------------------------------------

def check_onnx_structure(
    onnx_path: Path,
    num_classes: int,
) -> onnx.ModelProto:

    print()
    print("Checking ONNX structure...")
    print()

    model = onnx.load(
        str(onnx_path)
    )

    # --------------------------------------------------------------
    # ONNX checker
    # --------------------------------------------------------------

    onnx.checker.check_model(
        model
    )

    print(
        "[OK] onnx.checker.check_model()"
    )

    # --------------------------------------------------------------
    # Opset
    # --------------------------------------------------------------

    standard_opsets = [
        int(opset.version)
        for opset in model.opset_import
        if opset.domain in (
            "",
            "ai.onnx",
        )
    ]

    if not standard_opsets:
        raise RuntimeError(
            "ONNX model does not declare "
            "a standard ONNX opset."
        )

    if len(standard_opsets) != 1:
        raise RuntimeError(
            "Unexpected multiple standard "
            "ONNX opset declarations: "
            f"{standard_opsets}"
        )

    actual_opset = (
        standard_opsets[0]
    )

    if actual_opset != ONNX_OPSET:
        raise RuntimeError(
            "Wrong ONNX opset.\n"
            f"Expected: {ONNX_OPSET}\n"
            f"Actual:   {actual_opset}"
        )

    print(
        f"[OK] ONNX opset = "
        f"{actual_opset}"
    )

    # --------------------------------------------------------------
    # Input
    # --------------------------------------------------------------

    graph_inputs = (
        model.graph.input
    )

    if len(graph_inputs) != 1:
        raise RuntimeError(
            "Expected exactly one model input, "
            f"found {len(graph_inputs)}."
        )

    model_input = (
        graph_inputs[0]
    )

    if model_input.name != "input":
        raise RuntimeError(
            "Unexpected ONNX input name: "
            f"{model_input.name}"
        )

    input_shape = (
        get_value_info_shape(
            model_input
        )
    )

    expected_input_shape = list(
        INPUT_SHAPE
    )

    if input_shape != expected_input_shape:
        raise RuntimeError(
            "Wrong ONNX input shape.\n"
            f"Expected: {expected_input_shape}\n"
            f"Actual:   {input_shape}"
        )

    print(
        f"[OK] input shape = "
        f"{input_shape}"
    )

    # --------------------------------------------------------------
    # Outputs
    # --------------------------------------------------------------

    graph_outputs = list(
        model.graph.output
    )

    if len(graph_outputs) != 2:
        raise RuntimeError(
            "Expected exactly two outputs, "
            f"found {len(graph_outputs)}."
        )

    output_by_name = {
        output.name: output
        for output in graph_outputs
    }

    required_outputs = {
        "logits",
        "embedding",
    }

    if (
        set(output_by_name.keys())
        != required_outputs
    ):
        raise RuntimeError(
            "Unexpected ONNX outputs.\n"
            f"Expected: {sorted(required_outputs)}\n"
            f"Actual:   "
            f"{sorted(output_by_name.keys())}"
        )

    logits_shape = (
        get_value_info_shape(
            output_by_name[
                "logits"
            ]
        )
    )

    expected_logits_shape = [
        1,
        num_classes,
    ]

    if (
        logits_shape
        != expected_logits_shape
    ):
        raise RuntimeError(
            "Wrong logits shape.\n"
            f"Expected: "
            f"{expected_logits_shape}\n"
            f"Actual:   {logits_shape}"
        )

    print(
        f"[OK] logits shape = "
        f"{logits_shape}"
    )

    embedding_shape = (
        get_value_info_shape(
            output_by_name[
                "embedding"
            ]
        )
    )

    expected_embedding_shape = [
        1,
        EMBEDDING_DIM,
    ]

    if (
        embedding_shape
        != expected_embedding_shape
    ):
        raise RuntimeError(
            "Wrong embedding shape.\n"
            f"Expected: "
            f"{expected_embedding_shape}\n"
            f"Actual:   {embedding_shape}"
        )

    print(
        f"[OK] embedding shape = "
        f"{embedding_shape}"
    )

    return model


# ---------------------------------------------------------------------------
# ONNX Runtime checks
# ---------------------------------------------------------------------------

def check_onnx_runtime_structure(
    onnx_path: Path,
    num_classes: int,
) -> ort.InferenceSession:

    print()
    print("Loading with ONNX Runtime...")
    print()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1:
        raise RuntimeError(
            "ONNX Runtime reports unexpected "
            f"input count: {len(inputs)}"
        )

    if inputs[0].name != "input":
        raise RuntimeError(
            "ONNX Runtime input name mismatch."
        )

    if list(inputs[0].shape) != list(
        INPUT_SHAPE
    ):
        raise RuntimeError(
            "ONNX Runtime input shape mismatch.\n"
            f"Expected: {list(INPUT_SHAPE)}\n"
            f"Actual:   {inputs[0].shape}"
        )

    output_by_name = {
        output.name: output
        for output in outputs
    }

    if set(
        output_by_name.keys()
    ) != {
        "logits",
        "embedding",
    }:
        raise RuntimeError(
            "ONNX Runtime output names mismatch."
        )

    if list(
        output_by_name[
            "logits"
        ].shape
    ) != [
        1,
        num_classes,
    ]:
        raise RuntimeError(
            "ONNX Runtime logits "
            "shape mismatch."
        )

    if list(
        output_by_name[
            "embedding"
        ].shape
    ) != [
        1,
        EMBEDDING_DIM,
    ]:
        raise RuntimeError(
            "ONNX Runtime embedding "
            "shape mismatch."
        )

    print(
        "[OK] ONNX Runtime loaded model"
    )

    print(
        "[OK] ONNX Runtime input/output shapes"
    )

    return session


# ---------------------------------------------------------------------------
# Numerical comparison
# ---------------------------------------------------------------------------

@torch.inference_mode()
def check_numerical_equivalence(
    pytorch_model: ONNXExportWrapper,
    session: ort.InferenceSession,
    dummy_input: torch.Tensor,
) -> None:

    print()
    print(
        "Comparing PyTorch and ONNX Runtime..."
    )
    print()

    # --------------------------------------------------------------
    # PyTorch
    # --------------------------------------------------------------

    pytorch_logits, pytorch_embedding = (
        pytorch_model(
            dummy_input
        )
    )

    pytorch_logits_np = (
        pytorch_logits
        .detach()
        .cpu()
        .numpy()
    )

    pytorch_embedding_np = (
        pytorch_embedding
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------------
    # ONNX Runtime
    # --------------------------------------------------------------

    input_numpy = (
        dummy_input
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    (
        onnx_logits,
        onnx_embedding,
    ) = session.run(
        [
            "logits",
            "embedding",
        ],
        {
            "input": input_numpy
        },
    )

    # --------------------------------------------------------------
    # Differences
    # --------------------------------------------------------------

    logits_max_abs_diff = float(
        np.max(
            np.abs(
                pytorch_logits_np
                - onnx_logits
            )
        )
    )

    embedding_max_abs_diff = float(
        np.max(
            np.abs(
                pytorch_embedding_np
                - onnx_embedding
            )
        )
    )

    print(
        "Logits max abs diff:    "
        f"{logits_max_abs_diff:.10f}"
    )

    print(
        "Embedding max abs diff: "
        f"{embedding_max_abs_diff:.10f}"
    )

    # --------------------------------------------------------------
    # Hard assertions
    # --------------------------------------------------------------

    try:

        np.testing.assert_allclose(
            onnx_logits,
            pytorch_logits_np,
            rtol=RTOL,
            atol=ATOL,
        )

    except AssertionError as exc:
        raise RuntimeError(
            "Numerical validation failed "
            "for logits."
        ) from exc

    print(
        "[OK] logits numerically equivalent"
    )

    try:

        np.testing.assert_allclose(
            onnx_embedding,
            pytorch_embedding_np,
            rtol=RTOL,
            atol=ATOL,
        )

    except AssertionError as exc:
        raise RuntimeError(
            "Numerical validation failed "
            "for embedding."
        ) from exc

    print(
        "[OK] embedding numerically equivalent"
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(
    model: ONNXExportWrapper,
    dummy_input: torch.Tensor,
    output_path: Path,
) -> None:

    print()
    print("Exporting ONNX...")
    print()

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),

        input_names=[
            "input"
        ],

        output_names=[
            "logits",
            "embedding",
        ],

        opset_version=ONNX_OPSET,

        # Fixed input shape:
        # [1, 3, 224, 224]
        dynamic_shapes=None,

        # Keep the entire model in one .onnx file.
        external_data=False,

        # Current torch.export-based exporter.
        dynamo=True,
    )

    if not output_path.is_file():
        raise RuntimeError(
            "ONNX exporter did not create "
            "the output file."
        )

    print(
        f"[OK] exported: {output_path}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Export the best hulk-hand checkpoint "
            "to ONNX Opset 13 and validate it."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to the trained best.pt checkpoint."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output path for the ONNX model."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed used for the numerical "
            "validation input. Default: 42"
        ),
    )

    args = parser.parse_args()

    checkpoint_path = (
        args.checkpoint
        .expanduser()
        .resolve()
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

    # Export to a temporary ONNX file first.
    # The requested output is only created after
    # ALL validation steps succeed.
    temporary_path = (
        output_path.with_name(
            output_path.stem
            + ".tmp.onnx"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    # --------------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------------

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    num_classes = int(
        checkpoint["num_classes"]
    )

    classes = checkpoint[
        "classes"
    ]

    print()
    print("hulk-hand ONNX export")
    print("=====================")
    print()

    print(
        f"Checkpoint:    {checkpoint_path}"
    )

    print(
        f"Output:        {output_path}"
    )

    print(
        f"Classes:       {num_classes}"
    )

    print(
        f"Input:         {list(INPUT_SHAPE)}"
    )

    print(
        f"Opset target:  {ONNX_OPSET}"
    )

    print(
        f"RTOL:          {RTOL}"
    )

    print(
        f"ATOL:          {ATOL}"
    )

    print()
    print("Class mapping:")

    for index, class_name in enumerate(
        classes
    ):
        print(
            f"  {index}: {class_name}"
        )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model = build_export_model(
        checkpoint
    )

    # Fixed batch size 1.
    torch.manual_seed(
        args.seed
    )

    dummy_input = torch.randn(
        INPUT_SHAPE,
        dtype=torch.float32,
        device="cpu",
    )

    # --------------------------------------------------------------
    # PyTorch sanity check
    # --------------------------------------------------------------

    with torch.inference_mode():

        (
            test_logits,
            test_embedding,
        ) = model(
            dummy_input
        )

    if list(
        test_logits.shape
    ) != [
        1,
        num_classes,
    ]:
        raise RuntimeError(
            "Unexpected PyTorch logits shape: "
            f"{list(test_logits.shape)}"
        )

    if list(
        test_embedding.shape
    ) != [
        1,
        EMBEDDING_DIM,
    ]:
        raise RuntimeError(
            "Unexpected PyTorch embedding shape: "
            f"{list(test_embedding.shape)}"
        )

    print()
    print(
        "[OK] PyTorch model output shapes"
    )

    # --------------------------------------------------------------
    # Export + validation
    # --------------------------------------------------------------

    try:

        export_onnx(
            model=model,
            dummy_input=dummy_input,
            output_path=temporary_path,
        )

        check_onnx_structure(
            onnx_path=temporary_path,
            num_classes=num_classes,
        )

        session = (
            check_onnx_runtime_structure(
                onnx_path=temporary_path,
                num_classes=num_classes,
            )
        )

        check_numerical_equivalence(
            pytorch_model=model,
            session=session,
            dummy_input=dummy_input,
        )

        # ----------------------------------------------------------
        # ALL checks passed
        # ----------------------------------------------------------

        if output_path.exists():
            output_path.unlink()

        temporary_path.replace(
            output_path
        )

    except Exception:

        # Do not leave an invalid ONNX model behind.
        if temporary_path.exists():
            temporary_path.unlink()

        raise

    # --------------------------------------------------------------
    # Final independent re-check after rename
    # --------------------------------------------------------------

    final_model = onnx.load(
        str(output_path)
    )

    onnx.checker.check_model(
        final_model
    )

    final_opsets = [
        int(opset.version)
        for opset in final_model.opset_import
        if opset.domain in (
            "",
            "ai.onnx",
        )
    ]

    if final_opsets != [
        ONNX_OPSET
    ]:
        raise RuntimeError(
            "Final ONNX file failed the "
            "post-save Opset 13 verification."
        )

    # --------------------------------------------------------------
    # Success
    # --------------------------------------------------------------

    print()
    print("==============================")
    print("ONNX EXPORT SUCCESSFUL")
    print("==============================")
    print()

    print(
        f"File:      {output_path}"
    )

    print(
        f"Opset:     {ONNX_OPSET}  [VERIFIED]"
    )

    print(
        f"Input:     [1, 3, 224, 224]"
    )

    print(
        f"Logits:    [1, {num_classes}]"
    )

    print(
        f"Embedding: [1, {EMBEDDING_DIM}]"
    )

    print()

    print(
        "ONNX checker:       PASS"
    )

    print(
        "ONNX Runtime:       PASS"
    )

    print(
        "Numerical logits:   PASS"
    )

    print(
        "Numerical embedding: PASS"
    )

    print(
        "Opset 13:           PASS"
    )

    print()


if __name__ == "__main__":
    main()