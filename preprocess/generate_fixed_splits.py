"""Generate validation/test paper images once with reproducible per-sample seeds."""

import argparse
import csv
import hashlib
import random
from pathlib import Path

import cv2

from tamer.university.augmentation import DynamicPaperAugmentation
from tamer.university.image_io import read_grayscale, write_image


def stable_seed(global_seed: int, split: str, sample_id: str) -> int:
    digest = hashlib.sha256("{}:{}:{}".format(global_seed, split, sample_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def severity_schedule(split: str, count: int, seed: int):
    distribution = {
        "validation": (0.60, 0.30, 0.10),
        "test": (0.30, 0.40, 0.30),
    }[split]
    mild = int(round(count * distribution[0]))
    medium = int(round(count * distribution[1]))
    hard = count - mild - medium
    values = ["mild"] * mild + ["medium"] * medium + ["hard"] * hard
    random.Random(seed + (1 if split == "validation" else 2)).shuffle(values)
    return values


def generate(data_root: Path, split: str, augmenter: DynamicPaperAugmentation, seed: int) -> None:
    input_manifest = data_root / "splits" / (split + "_clean.csv")
    output_manifest = data_root / "splits" / (split + "_fixed.csv")
    output_dir = data_root / "fixed" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    severities = severity_schedule(split, len(rows), seed)
    for index, (row, severity) in enumerate(zip(rows, severities), start=1):
        clean = read_grayscale(data_root / row["image_path"])
        if clean is None:
            raise FileNotFoundError(row["image_path"])
        output = augmenter(
            clean,
            seed=stable_seed(seed, split, row["sample_id"]),
            severity=severity,
        )
        image_path = output_dir / (row["sample_id"] + ".png")
        if not write_image(image_path, output):
            raise IOError(str(image_path))
        row["image_path"] = image_path.relative_to(data_root).as_posix()
        row["severity"] = severity
        if index % 250 == 0:
            print("Generated {}/{} fixed {} images".format(index, len(rows), split))
    with output_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/university")
    parser.add_argument("--backgrounds", default=None)
    parser.add_argument("--validation-backgrounds", default=None)
    parser.add_argument("--test-backgrounds", default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    validation_backgrounds = args.validation_backgrounds or args.backgrounds
    test_backgrounds = args.test_backgrounds or args.backgrounds
    validation_augmenter = DynamicPaperAugmentation(background_dir=validation_backgrounds)
    test_augmenter = DynamicPaperAugmentation(background_dir=test_backgrounds)
    generate(root, "validation", validation_augmenter, args.seed)
    generate(root, "test", test_augmenter, args.seed)
    print("Fixed validation/test generated. These files must never be dynamically augmented.")


if __name__ == "__main__":
    main()
