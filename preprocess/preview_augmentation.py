"""Render deterministic augmentation examples before launching a training run."""

import argparse
from pathlib import Path

import cv2

from tamer.university.augmentation import DynamicPaperAugmentation
from tamer.university.image_io import read_grayscale, write_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="A clean rasterized MathWriting PNG")
    parser.add_argument("--output", default="outputs/augmentation_preview")
    parser.add_argument("--backgrounds", default="data/paper_backgrounds")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    clean = read_grayscale(args.image)
    if clean is None:
        raise FileNotFoundError(args.image)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    augmenter = DynamicPaperAugmentation(args.backgrounds)
    write_image(output / "clean.png", clean)
    for index in range(args.count):
        image = augmenter(clean, seed=args.seed + index)
        write_image(output / "variant_{:02d}.png".format(index + 1), image)
    print("Wrote {} deterministic variants to {}".format(args.count, output))


if __name__ == "__main__":
    main()
