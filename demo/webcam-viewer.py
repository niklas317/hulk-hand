#!/usr/bin/env python3
"""Run live four-class gesture inference from a camera or video source."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "resnet18-hagrid" / "preprocessing"))
from gesture_preprocessing import PreprocessConfig, preprocess_bgr_frame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dinov2" / "preprocessing"))
from dinov2_preprocessing import (  # noqa: E402
    PreprocessConfig as DinoPreprocessConfig,
    preprocess_bgr_frame as preprocess_dinov2_frame,
)

_IN1K_PATH = Path(__file__).resolve().parents[1] / "resnet18-in1k" / "preprocessing" / "in1k_preprocessing.py"
_IN1K_SPEC = importlib.util.spec_from_file_location("resnet18_in1k_preprocessing", _IN1K_PATH)
if _IN1K_SPEC is None or _IN1K_SPEC.loader is None:
    raise RuntimeError(f"Could not load ResNet18-IN1K preprocessing: {_IN1K_PATH}")
in1k_preprocessing = importlib.util.module_from_spec(_IN1K_SPEC)
sys.modules[_IN1K_SPEC.name] = in1k_preprocessing
_IN1K_SPEC.loader.exec_module(in1k_preprocessing)


"""
Class mapping:
0: one
1: two
2: stop
3: no_gesture
"""


CLASS_NAMES = [
    "one",
    "two",
    "stop",
    "no_gesture",
]


def load_class_names(class_names_arg: str | None, class_file: str | None) -> List[str]:
    """Resolve labels from CLI input, a file, or the model's standard order."""
    if class_file:
        path = Path(class_file)
        if not path.exists():
            raise FileNotFoundError(f"Class file not found: {path}")
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    if class_names_arg:
        return [item.strip() for item in class_names_arg.split(",") if item.strip()]

    return CLASS_NAMES


def load_session(model_path: str | Path) -> ort.InferenceSession:
    """Create a CPU ONNX Runtime session for the selected model."""
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def preprocess_frame(
    frame: np.ndarray,
    preprocessing: str,
    image_size: int = 224,
) -> np.ndarray:
    """Apply the selected model's preprocessing to one OpenCV frame."""
    if preprocessing in (
        "resnet18",
        "resnet18-fractaldb",
        "resnet34-fractaldb",
    ):
        return preprocess_bgr_frame(
            frame,
            PreprocessConfig(image_size=image_size),
        )
    if preprocessing == "resnet18-in1k":
        return in1k_preprocessing.preprocess_bgr_frame(
            frame,
            in1k_preprocessing.PreprocessConfig(image_size=image_size),
        )
    return preprocess_dinov2_frame(
        frame,
        DinoPreprocessConfig(image_size=image_size),
    )


def annotate_frame(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw the predicted label over a camera frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (380, 56), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(
        frame,
        text,
        (14, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    """Open the source, classify frames, and display results until exit."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run live hulk-hand inference with an exported ONNX model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the ONNX model downloaded from a release.",
    )
    parser.add_argument("--camera", default="/dev/video0", help="Camera device or video file path")
    parser.add_argument(
        "--preprocessing",
        required=True,
        choices=(
            "resnet18",
            "resnet18-fractaldb",
            "resnet34-fractaldb",
            "resnet18-in1k",
            "dinov2",
        ),
        help="Preprocessing pipeline matching the selected model.",
    )
    parser.add_argument("--class-names", default=None, help="Comma-separated label list for model outputs")
    parser.add_argument("--class-file", default=None, help="Text file with one class name per line")
    args = parser.parse_args()

    model_path = Path(args.model)
    camera_path = args.camera

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    session = load_session(model_path)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    input_shape = list(session.get_inputs()[0].shape)
    if len(input_shape) != 4 or input_shape[1:] != [3, 224, 224]:
        raise RuntimeError(
            "Expected an ONNX input shaped [batch, 3, 224, 224], "
            f"got {input_shape}."
        )

    if "logits" in output_names:
        logits_output_name = "logits"
    elif output_names:
        logits_output_name = output_names[0]
    else:
        raise RuntimeError("ONNX model has no outputs.")

    class_names = load_class_names(args.class_names, args.class_file)

    capture_backend = cv2.CAP_V4L2 if str(camera_path).startswith("/dev/") else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_path, capture_backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {camera_path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            input_tensor = preprocess_frame(
                frame,
                preprocessing=args.preprocessing,
            )
            logits = session.run(
                [logits_output_name],
                {input_name: input_tensor},
            )[0]
            # Softmax makes the displayed decision numerically stable without changing argmax.
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            idx = int(np.argmax(probs, axis=1)[0])
            label = class_names[idx] if idx < len(class_names) else str(idx)

            annotated = annotate_frame(frame, label)
            cv2.imshow("hulk-hand", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
