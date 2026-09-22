#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from gesture_preprocessing import PreprocessConfig, iter_image_files, preprocess_image_file


def preprocess_path(input_path: Path, output_path: Path, image_size: int) -> None:
    config = PreprocessConfig(image_size=image_size)

    if input_path.is_file():
        if output_path.is_dir():
            output_path = output_path / input_path.name
        preprocess_image_file(input_path, output_path, config)
        print(f"{input_path} -> {output_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    for image_path in iter_image_files(input_path):
        relative = image_path.relative_to(input_path)
        target = output_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        preprocess_image_file(image_path, target, config)
        print(f"{image_path} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess gesture images into square model input")
    parser.add_argument("--input", required=True, help="Input image file or directory")
    parser.add_argument("--output", required=True, help="Output file or directory")
    parser.add_argument("--image-size", type=int, default=224, help="Target square size")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    preprocess_path(input_path, output_path, args.image_size)


if __name__ == "__main__":
    main()
