#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2


def open_camera(device: str) -> cv2.VideoCapture:
    """
    Open a V4L2 camera device such as /dev/video0.
    """

    capture = cv2.VideoCapture(
        device,
        cv2.CAP_V4L2,
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera device: {device}"
        )

    return capture


def draw_status(
    frame,
    text: str,
):
    """
    Draw status information on the preview frame.
    """

    preview = cv2.flip(
        frame,
        1,
    )

    cv2.putText(
        preview,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return preview


def capture_burst(
    device: str,
    output_dir: Path,
    fps: float,
    num_images: int,
    delay: float,
) -> None:

    if fps <= 0:
        raise ValueError(
            "FPS must be greater than zero."
        )

    if num_images <= 0:
        raise ValueError(
            "Number of images must be greater than zero."
        )

    output_dir = (
        output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture = open_camera(
        device
    )

    frame_interval = (
        1.0 / fps
    )

    window_name = "hulk-hand burst capture"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    try:

        # ----------------------------------------------------------
        # Initial preview / delay
        # ----------------------------------------------------------

        start_time = time.monotonic()

        while True:

            success, frame = capture.read()

            if not success:
                raise RuntimeError(
                    "Failed to read frame from camera."
                )

            elapsed = (
                time.monotonic()
                - start_time
            )

            remaining = max(
                0.0,
                delay - elapsed,
            )

            preview = draw_status(
                frame,
                f"Starting in {remaining:.1f}s",
            )

            cv2.imshow(
                window_name,
                preview,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Capture aborted.")
                return

            if elapsed >= delay:
                break

        # ----------------------------------------------------------
        # Burst
        # ----------------------------------------------------------

        print()
        print("Recording burst")
        print("----------------")
        print(f"Device:  {device}")
        print(f"FPS:     {fps}")
        print(f"Images:  {num_images}")
        print(f"Output:  {output_dir}")
        print()

        next_capture_time = (
            time.monotonic()
        )

        captured = 0

        while captured < num_images:

            success, frame = capture.read()

            if not success:
                raise RuntimeError(
                    "Failed to read frame from camera."
                )

            now = time.monotonic()

            # ------------------------------------------------------
            # Save frame when the next FPS interval is reached
            # ------------------------------------------------------

            if now >= next_capture_time:

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                filename = (
                    f"image_{timestamp}.jpg"
                )

                output_path = (
                    output_dir
                    / filename
                )

                success_write = cv2.imwrite(
                    str(output_path),
                    frame,
                )

                if not success_write:
                    raise RuntimeError(
                        f"Failed to write image: "
                        f"{output_path}"
                    )

                captured += 1

                next_capture_time += (
                    frame_interval
                )

                print(
                    f"\rCaptured "
                    f"{captured}/{num_images}",
                    end="",
                    flush=True,
                )

            # ------------------------------------------------------
            # Preview
            # ------------------------------------------------------

            preview = draw_status(
                frame,
                (
                    f"RECORDING "
                    f"{captured}/{num_images}"
                ),
            )

            cv2.imshow(
                window_name,
                preview,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print()
                print("Capture aborted.")
                return

        print()
        print()
        print("Burst finished.")
        print(
            f"Saved {captured} images "
            f"to {output_dir}"
        )

        # Briefly show completion.
        completion_start = (
            time.monotonic()
        )

        while (
            time.monotonic()
            - completion_start
            < 0.5
        ):

            success, frame = capture.read()

            if not success:
                break

            preview = draw_status(
                frame,
                "DONE",
            )

            cv2.imshow(
                window_name,
                preview,
            )

            cv2.waitKey(1)

    finally:

        capture.release()
        cv2.destroyAllWindows()


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Capture a fixed-size image burst "
            "from a V4L2 camera."
        )
    )

    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help=(
            "V4L2 camera device, "
            "for example /dev/video0."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory where captured images "
            "will be stored."
        ),
    )

    parser.add_argument(
        "--fps",
        type=float,
        required=True,
        help=(
            "Capture rate in frames per second."
        ),
    )

    parser.add_argument(
        "--num-images",
        type=int,
        required=True,
        help=(
            "Number of images to capture."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help=(
            "Delay before recording starts "
            "in seconds. Default: 1.0"
        ),
    )

    args = parser.parse_args()

    capture_burst(
        device=args.device,
        output_dir=args.output_dir,
        fps=args.fps,
        num_images=args.num_images,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()