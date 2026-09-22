#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort

from gesture_preprocessing import PreprocessConfig, preprocess_bgr_frame


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
    if class_file:
        path = Path(class_file)
        if not path.exists():
            raise FileNotFoundError(f"Class file not found: {path}")
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    if class_names_arg:
        return [item.strip() for item in class_names_arg.split(",") if item.strip()]

    return CLASS_NAMES


def load_session(model_path: str | Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def preprocess_frame(frame: np.ndarray, image_size: int = 224) -> np.ndarray:
    return preprocess_bgr_frame(frame, PreprocessConfig(image_size=image_size))


def annotate_frame(frame: np.ndarray, text: str) -> np.ndarray:
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
    import argparse

    parser = argparse.ArgumentParser(description="Run webcam inference with the exported ResNet18 ONNX model")
    parser.add_argument("--model", default=str(Path(__file__).with_name("ResNet18.onnx")), help="Path to the ONNX model")
    parser.add_argument("--camera", default="/dev/video0", help="Camera device or video file path")
    parser.add_argument("--class-names", default=None, help="Comma-separated label list for model outputs")
    parser.add_argument("--class-file", default=None, help="Text file with one class name per line")
    args = parser.parse_args()

    model_path = Path(args.model)
    camera_path = args.camera

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    session = load_session(model_path)
    input_name = session.get_inputs()[0].name
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

            input_tensor = preprocess_frame(frame)
            logits = session.run(None, {input_name: input_tensor})[0]
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            idx = int(np.argmax(probs, axis=1)[0])
            label = class_names[idx] if idx < len(class_names) else str(idx)

            annotated = annotate_frame(frame, label)
            cv2.imshow("HaGRID ResNet18", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
