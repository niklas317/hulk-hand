from pathlib import Path
import shutil
import argparse

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

NUM_IMAGES = 103


def get_unique_path(target_dir, filename):
    """Verhindert Überschreiben bei gleichen Dateinamen."""
    target_path = target_dir / filename

    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    counter = 1

    while True:
        new_path = target_dir / f"{stem}_{counter}{suffix}"

        if not new_path.exists():
            return new_path

        counter += 1


def main(source_dir, target_dir):
    source = Path(source_dir).resolve()
    target = Path(target_dir).resolve()

    target.mkdir(parents=True, exist_ok=True)

    for folder in source.rglob("*"):
        if not folder.is_dir():
            continue

        # Zielordner überspringen, falls er im Quellordner liegt
        if target == folder or target in folder.parents:
            continue

        images = sorted(
            file for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            continue

        selected = images[:NUM_IMAGES]

        for image in selected:
            destination = get_unique_path(target, image.name)
            shutil.copy2(image, destination)

        print(
            f"{folder}: "
            f"{len(selected)} von {len(images)} Bildern kopiert"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kopiert 103 Bilder aus jedem Unterverzeichnis direkt ins Ziel."
    )

    parser.add_argument("source", help="Quellverzeichnis")
    parser.add_argument("target", help="Zielverzeichnis")

    args = parser.parse_args()

    main(args.source, args.target)