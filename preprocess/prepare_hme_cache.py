"""Create disjoint HME100K replay and retention-validation caches.

Both cache splits come from HME100K train. The official HME100K test split is
never opened here and remains untouched for one final evaluation.
"""

import argparse
import csv
import pickle
import random
from pathlib import Path

from tamer.university.image_io import write_image


def read_captions(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            parts = line.strip().split()
            if parts:
                rows.append((parts[0], " ".join(parts[1:])))
    return rows


def export_selection(images, output: Path, cache_name: str, selected) -> None:
    image_dir = output / "images" / cache_name
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for index, (sample_id, label) in enumerate(selected, start=1):
        image_path = image_dir / (Path(sample_id).stem + ".png")
        if not write_image(image_path, images[sample_id]):
            raise IOError(str(image_path))
        manifest_rows.append(
            {
                "sample_id": "hme_" + Path(sample_id).stem,
                "image_path": image_path.relative_to(output).as_posix(),
                "label": label,
                "category": "hme100k",
                "source": "hme100k_train",
                "token_count": len(label.split()),
            }
        )
        if index % 2500 == 0:
            print("Exported {}/{} {} images".format(index, len(selected), cache_name))

    manifest_path = output / ("replay.csv" if cache_name == "replay" else "validation.csv")
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hme100k", required=True)
    parser.add_argument("--output", default="data/hme_cache")
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--validation-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    source = Path(args.hme100k).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    captions = read_captions(source / "train" / "caption.txt")
    requested = args.replay_size + args.validation_size
    if requested > len(captions):
        raise ValueError(
            "Requested {} samples from an HME train split containing {}".format(
                requested, len(captions)
            )
        )

    rng = random.Random(args.seed)
    rng.shuffle(captions)
    replay = captions[: args.replay_size]
    validation = captions[args.replay_size : requested]
    replay_ids = {sample_id for sample_id, _ in replay}
    validation_ids = {sample_id for sample_id, _ in validation}
    overlap = replay_ids & validation_ids
    if overlap:
        raise RuntimeError("Replay/validation overlap: {}".format(len(overlap)))

    pickle_path = source / "train" / "images.pkl"
    print("Loading {} once...".format(pickle_path))
    with pickle_path.open("rb") as stream:
        images = pickle.load(stream)
    export_selection(images, output, "replay", replay)
    export_selection(images, output, "validation", validation)
    del images

    print(
        "HME cache ready: replay={} validation={} overlap=0 source=train seed={}.".format(
            len(replay), len(validation), args.seed
        )
    )
    print("The official HME100K test split was not opened or modified.")


if __name__ == "__main__":
    main()
