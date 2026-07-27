"""Convert HEIC paper photos, audit quality and create disjoint split folders."""

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from tamer.university.image_io import read_grayscale


PAPER_TYPES = ("a4", "ruled", "grid")


def allocate(count, train_ratio=0.70, validation_ratio=0.15):
    train = max(1, int(round(count * train_ratio)))
    validation = max(1, int(round(count * validation_ratio)))
    test = count - train - validation
    if test < 1:
        train -= 1
        test = 1
    return train, validation, test


def convert(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(str(source)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        # Preserve paper grain and fine rules; chroma subsampling would soften them.
        image.save(str(destination), "JPEG", quality=95, subsampling=0, optimize=True)


def audit(path: Path):
    image = read_grayscale(path)
    if image is None:
        raise IOError("Cannot read {}".format(path))
    height, width = image.shape
    thumbnail_scale = min(1.0, 1400.0 / max(height, width))
    if thumbnail_scale < 1.0:
        image = cv2.resize(image, None, fx=thumbnail_scale, fy=thumbnail_scale, interpolation=cv2.INTER_AREA)
    brightness = float(image.mean())
    contrast = float(image.std())
    sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
    clipped_dark = float((image <= 5).mean())
    clipped_bright = float((image >= 250).mean())
    flags = []
    if brightness < 65:
        flags.append("very_dark")
    if brightness > 247:
        flags.append("overexposed")
    if sharpness < 18:
        flags.append("possibly_blurry")
    if clipped_bright > 0.65:
        flags.append("large_highlight_area")
    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "sharpness": round(sharpness, 3),
        "clipped_dark": round(clipped_dark, 6),
        "clipped_bright": round(clipped_bright, 6),
        "quality_flags": ";".join(flags),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/paper_backgrounds")
    parser.add_argument("--output", default="data/paper_backgrounds_processed")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    register_heif_opener()
    source_root = Path(args.source).resolve()
    output_root = Path(args.output).resolve()
    if output_root.exists():
        shutil.rmtree(str(output_root))
    output_root.mkdir(parents=True)
    rng = random.Random(args.seed)
    report = []
    summary = {}
    for paper_type in PAPER_TYPES:
        source_dir = source_root / paper_type
        files = sorted(
            path for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in (".heic", ".heif", ".jpg", ".jpeg", ".png")
        )
        if not files:
            raise RuntimeError("No background images found for {}".format(paper_type))
        rng.shuffle(files)
        train_count, validation_count, test_count = allocate(len(files))
        split_names = (
            ["train"] * train_count
            + ["validation"] * validation_count
            + ["test"] * test_count
        )
        summary[paper_type] = {
            "total": len(files),
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
        }
        for index, (source, split) in enumerate(zip(files, split_names), start=1):
            destination = output_root / split / paper_type / (paper_type + "_{:03d}.jpg".format(index))
            if source.suffix.lower() in (".heic", ".heif"):
                convert(source, destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(str(source)) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image.save(str(destination), "JPEG", quality=95, subsampling=0, optimize=True)
            metrics = audit(destination)
            report.append(
                {
                    "source": source.relative_to(source_root).as_posix(),
                    "output": destination.relative_to(output_root).as_posix(),
                    "paper_type": paper_type,
                    "split": split,
                    **metrics,
                }
            )
    fields = list(report[0].keys())
    with (output_root / "background_audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report)
    print("Prepared paper backgrounds: {}".format(summary))
    flagged = [row for row in report if row["quality_flags"]]
    print("Quality warnings: {}/{} (see background_audit.csv)".format(len(flagged), len(report)))


if __name__ == "__main__":
    main()
