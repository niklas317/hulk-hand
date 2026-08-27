#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def open_rgb(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)

    # Handle images with transparency.
    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        image = image.convert("RGBA")

        background = Image.new(
            "RGBA",
            image.size,
            (255, 255, 255, 255),
        )

        background.alpha_composite(image)
        image = background.convert("RGB")

    else:
        image = image.convert("RGB")

    return image


def resize_image(
    image: Image.Image,
    image_size: int,
    mode: str,
) -> Image.Image:

    target = (image_size, image_size)
    resample = Image.Resampling.LANCZOS

    if mode == "fit-pad":
        return ImageOps.pad(
            image,
            target,
            method=resample,
            color=(0, 0, 0),
            centering=(0.5, 0.5),
        )

    if mode == "center-crop":
        return ImageOps.fit(
            image,
            target,
            method=resample,
            centering=(0.5, 0.5),
        )

    if mode == "stretch":
        return image.resize(
            target,
            resample=resample,
        )

    raise ValueError(f"Unknown resize mode: {mode}")


def collect_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def preprocess(
    input_dir: Path,
    output_dir: Path,
    image_size: int,
    mode: str,
    quality: int,
) -> None:

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError(
            "Output directory must not be inside the input directory."
        )

    images = collect_images(input_dir)

    if not images:
        raise RuntimeError(
            f"No supported images found in: {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp identifying this preprocessing run.
    run_timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    processed = 0
    errors = 0

    print(f"Input:      {input_dir}")
    print(f"Output:     {output_dir}")
    print(f"Images:     {len(images)}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Mode:       {mode}")
    print()

    for index, src in enumerate(images, start=1):

        relative_path = src.relative_to(input_dir)
        relative_parent = relative_path.parent

        target_dir = output_dir / relative_parent
        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            f"image_{run_timestamp}_{index:06d}.jpg"
        )

        dst = target_dir / filename

        try:
            with open_rgb(src) as image:
                image = resize_image(
                    image,
                    image_size=image_size,
                    mode=mode,
                )

                image.save(
                    dst,
                    format="JPEG",
                    quality=quality,
                )

            processed += 1

        except (
            OSError,
            UnidentifiedImageError,
            ValueError,
        ) as exc:

            errors += 1
            print(
                f"[ERROR] {src}: {exc}"
            )

        if (
            index == 1
            or index % 100 == 0
            or index == len(images)
        ):
            print(
                f"{index}/{len(images)} "
                f"processed={processed} "
                f"errors={errors}"
            )

    print()
    print("Preprocessing finished.")
    print(f"Processed: {processed}")
    print(f"Errors:    {errors}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess images recursively for "
            "hulk-hand training."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the source images.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the preprocessed images.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Output image size. Default: 224",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "fit-pad",
            "center-crop",
            "stretch",
        ],
        default="fit-pad",
        help="Resize strategy. Default: fit-pad",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="JPEG quality. Default: 90",
    )

    args = parser.parse_args()

    preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        mode=args.mode,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()
