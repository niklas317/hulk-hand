#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

CLASS_NAMES = ["one", "two", "stop", "no_gesture"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Job:
    src: str
    dst: str
    class_name: str
    label: int
    image_size: int
    quality: int
    mode: str
    overwrite: bool
    optimize_jpeg: bool


def _has_class_dirs(path: Path, class_names: Sequence[str]) -> bool:
    return path.is_dir() and all((path / class_name).is_dir() for class_name in class_names)


def _find_dataset_roots(search_root: Path, class_names: Sequence[str], max_depth: int) -> list[Path]:
    """Find folders below search_root that directly contain all class folders.

    This scans directories only, not all image files, so it is much cheaper than
    a full dataset index. It is useful when the user passes /media instead of
    /media/<user>/<drive>/<dataset>.
    """
    matches: list[Path] = []
    queue: list[tuple[Path, int]] = [(search_root, 0)]
    seen: set[Path] = set()

    while queue:
        current, depth = queue.pop(0)
        try:
            resolved = current.resolve()
        except OSError:
            resolved = current
        if resolved in seen:
            continue
        seen.add(resolved)

        if _has_class_dirs(current, class_names):
            matches.append(current)
            # Do not descend into a matching dataset root; nested matches are
            # usually the class folders themselves or duplicates.
            continue

        if depth >= max_depth:
            continue

        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    # Skip hidden/system-ish folders unless the user explicitly
                    # points to them as input_root.
                    if entry.name.startswith("."):
                        continue
                    queue.append((Path(entry.path), depth + 1))
        except (PermissionError, FileNotFoundError, OSError):
            continue

    return sorted(matches)


def resolve_input_root(input_root: Path, class_names: Sequence[str], auto_detect_depth: int) -> Path:
    input_root = input_root.expanduser().resolve()
    if _has_class_dirs(input_root, class_names):
        return input_root

    if auto_detect_depth <= 0:
        missing = [name for name in class_names if not (input_root / name).is_dir()]
        raise FileNotFoundError(
            f"Input root does not directly contain all class folders. Missing under {input_root}: {missing}"
        )

    print(f"Input root does not directly contain class folders. Searching below: {input_root}", flush=True)
    candidates = _find_dataset_roots(input_root, class_names, max_depth=auto_detect_depth)

    if len(candidates) == 1:
        print(f"Auto-detected dataset root: {candidates[0]}", flush=True)
        return candidates[0].resolve()

    if len(candidates) > 1:
        print("Multiple possible dataset roots found:", flush=True)
        for candidate in candidates[:20]:
            print(f"  {candidate}", flush=True)
        raise RuntimeError(
            "Please rerun with the exact --input-root path to the dataset folder that contains "
            "one/, two/, stop/, and no_gesture/."
        )

    raise FileNotFoundError(
        f"No dataset root with class folders {list(class_names)} found below {input_root} "
        f"within depth {auto_detect_depth}. Please pass the exact --input-root."
    )


def _safe_output_path(src: Path, class_dir: Path, output_class_dir: Path) -> Path:
    """Mirror the source folder structure but add a short hash to avoid collisions."""
    rel = src.relative_to(class_dir)
    digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
    filename = f"{rel.stem}__{digest}.jpg"
    return output_class_dir / rel.parent / filename


def _collect_jobs(
    input_root: Path,
    output_root: Path,
    class_names: Sequence[str],
    image_size: int,
    quality: int,
    mode: str,
    overwrite: bool,
    optimize_jpeg: bool,
    max_per_class: int | None = None,
) -> list[Job]:
    jobs: list[Job] = []
    for label, class_name in enumerate(class_names):
        class_dir = input_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        output_class_dir = output_root / class_name
        print(f"Indexing class '{class_name}': {class_dir}", flush=True)
        count_before = len(jobs)
        class_count = 0
        for src in sorted(class_dir.rglob("*")):
            if not src.is_file() or src.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if max_per_class is not None and class_count >= max_per_class:
                break
            dst = _safe_output_path(src, class_dir, output_class_dir)
            jobs.append(
                Job(
                    src=str(src),
                    dst=str(dst),
                    class_name=class_name,
                    label=label,
                    image_size=image_size,
                    quality=quality,
                    mode=mode,
                    overwrite=overwrite,
                    optimize_jpeg=optimize_jpeg,
                )
            )
            class_count += 1
        print(f"  found {len(jobs) - count_before:,} images", flush=True)
    return jobs


def _open_rgb(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)

    # Convert transparent images onto a white background instead of black.
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        image = background.convert("RGB")
    else:
        image = image.convert("RGB")
    return image


def _resize(image: Image.Image, image_size: int, mode: str) -> Image.Image:
    resample = Image.Resampling.LANCZOS
    target = (image_size, image_size)

    if mode == "fit-pad":
        # Preserve the full image, resize to fit inside image_size, then pad to square.
        return ImageOps.pad(image, target, method=resample, color=(0, 0, 0), centering=(0.5, 0.5))
    if mode == "center-crop":
        # Fill the square and crop borders. Faster at training time, but can cut off edges.
        return ImageOps.fit(image, target, method=resample, centering=(0.5, 0.5))
    if mode == "stretch":
        # Fast and exact, but changes aspect ratio. Usually not recommended for gestures.
        return image.resize(target, resample=resample)

    raise ValueError(f"Unknown resize mode: {mode}")


def _process_one(job: Job) -> tuple[bool, str, int, str, str]:
    src = Path(job.src)
    dst = Path(job.dst)

    try:
        if dst.exists() and not job.overwrite:
            return True, str(dst), job.label, job.class_name, "skipped"

        dst.parent.mkdir(parents=True, exist_ok=True)
        with _open_rgb(src) as image:
            image = _resize(image, job.image_size, job.mode)
            image.save(dst, format="JPEG", quality=job.quality, optimize=job.optimize_jpeg)
        return True, str(dst), job.label, job.class_name, "processed"
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return False, str(src), job.label, job.class_name, f"error: {exc}"


def _write_manifest(manifest_path: Path, rows: list[tuple[str, int, str]], output_root: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "class_name"])
        output_root_resolved = output_root.resolve()
        for path, label, class_name in rows:
            rel_path = Path(path).resolve().relative_to(output_root_resolved)
            writer.writerow([rel_path.as_posix(), label, class_name])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a local 224x224 JPEG dataset plus manifest for faster PyTorch training."
    )
    parser.add_argument(
        "--input-root",
        default="/media/niklas/X/dataset",
        help="Original dataset root with one/, two/, stop/, no_gesture/. Default: /media. If /media is too broad, the script auto-detects below it.",
    )
    parser.add_argument(
        "--output-root",
        default="~/Downloads/gestures_224",
        help="Local output root for the optimized dataset. Default: ~/Downloads/gestures_224",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality, 85-92 is usually a good range")
    parser.add_argument("--mode", choices=["fit-pad", "center-crop", "stretch"], default="fit-pad")
    parser.add_argument("--num-workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional smoke-test limit per class")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--class-names", nargs="+", default=CLASS_NAMES)
    parser.add_argument("--manifest-name", default="manifest.csv")
    parser.add_argument(
        "--auto-detect-depth",
        type=int,
        default=6,
        help="When --input-root does not directly contain the class folders, search this many directory levels below it. Use 0 to disable.",
    )
    parser.add_argument(
        "--optimize-jpeg",
        action="store_true",
        help="Use JPEG optimize=True. This can make files slightly smaller, but is usually slower. Off by default for speed.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=32,
        help="Chunk size for multiprocessing. Larger can be faster; smaller gives smoother progress updates.",
    )
    args = parser.parse_args()

    raw_input_root = Path(args.input_root).expanduser()
    if not raw_input_root.exists():
        raise FileNotFoundError(f"Input root not found: {raw_input_root}")

    output_root = Path(args.output_root).expanduser().resolve()
    manifest_path = output_root / args.manifest_name

    print("Start preprocessing", flush=True)
    print(f"requested_input_root={raw_input_root}", flush=True)
    print(f"output_root={output_root}", flush=True)
    print(f"image_size={args.image_size} mode={args.mode} quality={args.quality}", flush=True)
    print(f"num_workers={args.num_workers} chunksize={args.chunksize}", flush=True)
    print(f"optimize_jpeg={args.optimize_jpeg}", flush=True)

    input_root = resolve_input_root(raw_input_root, args.class_names, args.auto_detect_depth)
    print(f"using_input_root={input_root}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)

    jobs = _collect_jobs(
        input_root=input_root,
        output_root=output_root,
        class_names=args.class_names,
        image_size=args.image_size,
        quality=args.quality,
        mode=args.mode,
        overwrite=args.overwrite,
        optimize_jpeg=args.optimize_jpeg,
        max_per_class=args.max_per_class,
    )
    if not jobs:
        raise ValueError(f"No images found under {input_root}")

    print(f"Total images indexed: {len(jobs):,}", flush=True)
    print("Converting images...", flush=True)

    start = time.perf_counter()
    rows: list[tuple[str, int, str]] = []
    errors: list[tuple[str, str]] = []
    processed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        iterator = executor.map(_process_one, jobs, chunksize=max(1, args.chunksize))
        for done, result in enumerate(iterator, start=1):
            ok, path, label, class_name, status = result
            if ok:
                rows.append((path, label, class_name))
                if status == "skipped":
                    skipped += 1
                else:
                    processed += 1
            else:
                errors.append((path, status))

            if done == 1 or done % max(1, args.log_interval) == 0 or done == len(jobs):
                elapsed = max(1e-9, time.perf_counter() - start)
                imgs_per_sec = done / elapsed
                remaining = (len(jobs) - done) / max(1e-9, imgs_per_sec)
                print(
                    f"progress {done:,}/{len(jobs):,} ({100.0 * done / len(jobs):5.1f}%) "
                    f"speed={imgs_per_sec:6.1f} img/s eta={remaining/60:7.1f} min "
                    f"processed={processed:,} skipped={skipped:,} errors={len(errors):,}",
                    flush=True,
                )

    rows.sort(key=lambda row: row[0])
    _write_manifest(manifest_path, rows, output_root)
    (output_root / "class_names.txt").write_text("\n".join(args.class_names) + "\n", encoding="utf-8")

    if errors:
        error_path = output_root / "preprocessing_errors.csv"
        with error_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "error"])
            writer.writerows(errors)
        print(f"Errors written to: {error_path}", flush=True)

    elapsed = time.perf_counter() - start
    print("Preprocessing finished", flush=True)
    print(f"usable_images={len(rows):,} processed={processed:,} skipped={skipped:,} errors={len(errors):,}", flush=True)
    print(f"manifest={manifest_path}", flush=True)
    print(f"elapsed_minutes={elapsed/60:.1f}", flush=True)


if __name__ == "__main__":
    main()
