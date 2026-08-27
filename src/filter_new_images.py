#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class QualityThresholds:
    blur_variance: float
    dark_mean: float
    bright_mean: float
    min_contrast: float


def collect_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_preserving_structure(
    source: Path,
    destination_root: Path,
    relative_path: Path,
    overwrite: bool,
) -> Path:
    destination = destination_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        raise RuntimeError(f"Destination already exists: {destination}")

    shutil.copy2(source, destination)
    return destination


def laplacian_variance(grayscale: np.ndarray) -> float:
    padded = np.pad(grayscale, 1, mode="edge")

    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )

    return float(laplacian.var())


def score_image(
    path: Path,
    thresholds: QualityThresholds,
) -> dict[str, object]:
    metrics: dict[str, float] = {}
    reasons: list[str] = []

    try:
        with Image.open(path) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
            grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return {
            "decision": "reject",
            "reasons": [f"unreadable: {exc.__class__.__name__}"],
            "metrics": metrics,
            "quality_score": 0.0,
        }

    blur_variance = laplacian_variance(grayscale)
    mean_intensity = float(grayscale.mean())
    contrast = float(grayscale.std())

    metrics["blur_variance"] = blur_variance
    metrics["mean_intensity"] = mean_intensity
    metrics["contrast"] = contrast

    passed_checks = 0
    total_checks = 3

    if blur_variance < thresholds.blur_variance:
        reasons.append(
            f"blur<({blur_variance:.2f} < {thresholds.blur_variance:.2f})"
        )
    else:
        passed_checks += 1

    if mean_intensity < thresholds.dark_mean:
        reasons.append(
            f"too_dark<({mean_intensity:.2f} < {thresholds.dark_mean:.2f})"
        )
    elif mean_intensity > thresholds.bright_mean:
        reasons.append(
            f"too_bright>({mean_intensity:.2f} > {thresholds.bright_mean:.2f})"
        )
    else:
        passed_checks += 1

    if contrast < thresholds.min_contrast:
        reasons.append(
            f"low_contrast<({contrast:.2f} < {thresholds.min_contrast:.2f})"
        )
    else:
        passed_checks += 1

    decision = "keep" if not reasons else "reject"

    return {
        "decision": decision,
        "reasons": reasons,
        "metrics": metrics,
        "quality_score": passed_checks / total_checks,
    }


def process_old_samples(
    class_dir: Path,
    class_name: str,
    input_root: Path,
    clean_root: Path,
    manifest_file,
    overwrite: bool,
    dry_run: bool,
    summary: dict[str, object],
) -> None:
    old_dir = class_dir / "old"

    if not old_dir.is_dir():
        return

    for image_path in collect_images(old_dir):
        relative_path = image_path.relative_to(input_root)
        destination = clean_root / relative_path

        if not dry_run:
            copy_preserving_structure(
                source=image_path,
                destination_root=clean_root,
                relative_path=relative_path,
                overwrite=overwrite,
            )

        record = {
            "source_path": str(image_path),
            "relative_path": str(relative_path),
            "destination_path": str(destination),
            "class_name": class_name,
            "source_type": "old",
            "decision": "keep",
            "reasons": [],
            "quality_score": None,
            "metrics": {},
        }

        manifest_file.write(json.dumps(record, ensure_ascii=True) + "\n")

        summary["copied_old"] += 1
        summary["per_class"][class_name]["old"] += 1


def process_new_samples(
    class_dir: Path,
    class_name: str,
    input_root: Path,
    clean_root: Path,
    rejected_root: Path,
    manifest_file,
    thresholds: QualityThresholds,
    overwrite: bool,
    dry_run: bool,
    summary: dict[str, object],
) -> None:
    session_dirs = sorted(
        child
        for child in class_dir.iterdir()
        if child.is_dir() and child.name.startswith("session_")
    )

    for session_dir in session_dirs:
        session_name = session_dir.name
        images = collect_images(session_dir)

        for image_path in tqdm(
            images,
            desc=f"{class_name}/{session_name}",
            leave=False,
        ):
            relative_path = image_path.relative_to(input_root)
            analysis = score_image(image_path, thresholds)
            decision = str(analysis["decision"])
            reasons = list(analysis["reasons"])
            metrics = dict(analysis["metrics"])
            quality_score = float(analysis["quality_score"])

            if decision == "keep":
                destination_root = clean_root
                summary["kept_new"] += 1
                summary["per_class"][class_name]["new_kept"] += 1
                summary["per_session"][session_name]["kept"] += 1
            else:
                destination_root = rejected_root
                summary["rejected_new"] += 1
                summary["per_class"][class_name]["new_rejected"] += 1
                summary["per_session"][session_name]["rejected"] += 1
                for reason in reasons:
                    summary["reject_reasons"][reason] += 1

            destination = destination_root / relative_path

            if not dry_run:
                copy_preserving_structure(
                    source=image_path,
                    destination_root=destination_root,
                    relative_path=relative_path,
                    overwrite=overwrite,
                )

            record = {
                "source_path": str(image_path),
                "relative_path": str(relative_path),
                "destination_path": str(destination),
                "class_name": class_name,
                "session": session_name,
                "source_type": "new",
                "decision": decision,
                "reasons": reasons,
                "quality_score": quality_score,
                "metrics": metrics,
            }

            manifest_file.write(json.dumps(record, ensure_ascii=True) + "\n")


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise RuntimeError(
                f"Output directory already exists: {output_root}"
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy OLD samples to clean/ and split NEW samples into "
            "clean/ vs rejected/ based on image quality checks."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Original dataset root.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for clean/ and rejected/ outputs.",
    )

    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=60.0,
        help="Minimum Laplacian variance to keep an image. Default: 60.0",
    )

    parser.add_argument(
        "--dark-threshold",
        type=float,
        default=25.0,
        help="Minimum grayscale mean to avoid rejecting dark images. Default: 25.0",
    )

    parser.add_argument(
        "--bright-threshold",
        type=float,
        default=230.0,
        help="Maximum grayscale mean to avoid rejecting bright images. Default: 230.0",
    )

    parser.add_argument(
        "--contrast-threshold",
        type=float,
        default=18.0,
        help="Minimum grayscale standard deviation to keep an image. Default: 18.0",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and report only; do not copy files.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before writing.",
    )

    args = parser.parse_args()

    input_root = args.input_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    if output_root == input_root or input_root in output_root.parents:
        raise ValueError(
            "Output directory must not be inside the input directory."
        )

    prepare_output_root(output_root, overwrite=args.overwrite)

    clean_root = output_root / "clean"
    rejected_root = output_root / "rejected"

    if not args.dry_run:
        clean_root.mkdir(parents=True, exist_ok=True)
        rejected_root.mkdir(parents=True, exist_ok=True)

    thresholds = QualityThresholds(
        blur_variance=args.blur_threshold,
        dark_mean=args.dark_threshold,
        bright_mean=args.bright_threshold,
        min_contrast=args.contrast_threshold,
    )

    manifest_path = output_root / "filter_manifest.jsonl"
    summary_path = output_root / "filter_summary.json"

    summary: dict[str, object] = {
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "dry_run": args.dry_run,
        "thresholds": {
            "blur_variance": thresholds.blur_variance,
            "dark_mean": thresholds.dark_mean,
            "bright_mean": thresholds.bright_mean,
            "min_contrast": thresholds.min_contrast,
        },
        "copied_old": 0,
        "kept_new": 0,
        "rejected_new": 0,
        "reject_reasons": Counter(),
        "per_class": defaultdict(
            lambda: {
                "old": 0,
                "new_kept": 0,
                "new_rejected": 0,
            }
        ),
        "per_session": defaultdict(
            lambda: {
                "kept": 0,
                "rejected": 0,
            }
        ),
    }

    class_dirs = sorted(
        child
        for child in input_root.iterdir()
        if child.is_dir()
    )

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for class_dir in tqdm(class_dirs, desc="classes"):
            class_name = class_dir.name

            process_old_samples(
                class_dir=class_dir,
                class_name=class_name,
                input_root=input_root,
                clean_root=clean_root,
                manifest_file=manifest_file,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                summary=summary,
            )

            process_new_samples(
                class_dir=class_dir,
                class_name=class_name,
                input_root=input_root,
                clean_root=clean_root,
                rejected_root=rejected_root,
                manifest_file=manifest_file,
                thresholds=thresholds,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                summary=summary,
            )

    summary_output = {
        "input_dir": summary["input_dir"],
        "output_dir": summary["output_dir"],
        "dry_run": summary["dry_run"],
        "thresholds": summary["thresholds"],
        "copied_old": summary["copied_old"],
        "kept_new": summary["kept_new"],
        "rejected_new": summary["rejected_new"],
        "reject_reasons": dict(summary["reject_reasons"]),
        "per_class": dict(summary["per_class"]),
        "per_session": dict(summary["per_session"]),
    }

    summary_path.write_text(
        json.dumps(summary_output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print("Filtering finished.")
    print(f"OLD copied:   {summary_output['copied_old']}")
    print(f"NEW kept:     {summary_output['kept_new']}")
    print(f"NEW rejected: {summary_output['rejected_new']}")

    if summary_output["reject_reasons"]:
        print("Reject reasons:")
        for reason, count in sorted(
            summary_output["reject_reasons"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
