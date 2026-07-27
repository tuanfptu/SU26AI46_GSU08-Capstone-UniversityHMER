"""Select, split and rasterize a MathWriting university subset.

Only MathWriting's original ``train`` split is scanned. Groups are assigned by
canonical label, so no expression label can leak between train/validation/test.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tamer.university.latex import (
    canonical_label,
    categorize_formula,
    load_vocabulary,
    normalize_and_tokenize,
)


DEFAULT_QUOTAS = {
    "train": {
        "integral": 3000,
        "derivative": 3500,
        "limit": 600,
        "sum_series": 1200,
        "trig_log_exp": 800,
        "mixed": 900,
    },
    "validation": {
        "integral": 300,
        "derivative": 350,
        "limit": 60,
        "sum_series": 120,
        "trig_log_exp": 80,
        "mixed": 90,
    },
    "test": {
        "integral": 300,
        "derivative": 350,
        "limit": 60,
        "sum_series": 120,
        "trig_log_exp": 80,
        "mixed": 90,
    },
}


def _label_from_root(root) -> str:
    namespace = {"ink": "http://www.w3.org/2003/InkML"}
    for wanted_type in ("normalizedLabel", "label"):
        for annotation in root.findall("ink:annotation", namespace):
            if annotation.attrib.get("type") == wanted_type and annotation.text:
                return annotation.text
    raise ValueError("missing label")


def parse_label(path: Path) -> str:
    return _label_from_root(ET.parse(str(path)).getroot())


def parse_inkml(path: Path) -> Tuple[str, List["np.ndarray"]]:
    import numpy as np

    root = ET.parse(str(path)).getroot()
    namespace = {"ink": "http://www.w3.org/2003/InkML"}
    label = _label_from_root(root)
    traces = []
    for trace in root.findall("ink:trace", namespace):
        points = []
        for raw_point in (trace.text or "").split(","):
            values = raw_point.strip().split()
            if len(values) >= 2:
                points.append((float(values[0]), float(values[1])))
        if points:
            traces.append(np.asarray(points, dtype=np.float32))
    if not label or not traces:
        raise ValueError("missing label or traces")
    return label, traces


def rasterize(traces: Sequence["np.ndarray"], target_height: int = 96, max_width: int = 1024) -> "np.ndarray":
    import cv2
    import numpy as np

    all_points = np.concatenate(traces, axis=0)
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    source_h = max(float(y_max - y_min), 1.0)
    source_w = max(float(x_max - x_min), 1.0)
    margin = 10
    scale = min((target_height - 2 * margin) / source_h, (max_width - 2 * margin) / source_w)
    width = max(2 * margin + 1, int(round(source_w * scale)) + 2 * margin)
    height = max(2 * margin + 1, int(round(source_h * scale)) + 2 * margin)
    canvas = np.full((height, width), 255, dtype=np.uint8)
    thickness = max(1, int(round(target_height / 48.0)))
    for trace in traces:
        points = trace.copy()
        points[:, 0] = (points[:, 0] - x_min) * scale + margin
        points[:, 1] = (points[:, 1] - y_min) * scale + margin
        points = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
        if len(points) == 1:
            cv2.circle(canvas, tuple(points[0, 0]), thickness, 0, -1, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [points], False, 0, thickness, lineType=cv2.LINE_AA)
    return canvas


def scan_candidates(train_dir: Path, vocabulary: set, max_tokens: int) -> Tuple[List[dict], Counter]:
    accepted = []
    rejected = Counter()
    files = sorted(train_dir.rglob("*.inkml"))
    for number, path in enumerate(files, start=1):
        try:
            label = parse_label(path)
        except Exception:
            rejected["invalid_inkml"] += 1
            continue
        category = categorize_formula(label)
        if category is None:
            rejected["not_target_university_math"] += 1
            continue
        tokens, reason = normalize_and_tokenize(label, vocabulary, max_tokens=max_tokens)
        if reason:
            rejected[reason.split(":", 1)[0]] += 1
            continue
        accepted.append(
            {
                "sample_id": path.stem,
                "inkml_path": str(path.resolve()),
                "label": canonical_label(tokens),
                "category": category,
                "source": "mathwriting",
                "token_count": len(tokens),
            }
        )
        if number % 25000 == 0:
            print("Scanned {}/{} InkML files".format(number, len(files)))
    return accepted, rejected


def grouped_split(candidates: Sequence[dict], quotas: Dict[str, Dict[str, int]], seed: int) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for sample in candidates:
        groups[(sample["category"], sample["label"])].append(sample)
    by_category = defaultdict(list)
    for (category, _), samples in groups.items():
        rng.shuffle(samples)
        by_category[category].append(samples)
    for category_groups in by_category.values():
        rng.shuffle(category_groups)
        # Held-out sets consume small label groups first. This preserves common
        # multi-writer expressions for the larger train quota without leakage.
        category_groups.sort(key=len)

    result = {split: [] for split in quotas}
    # Protect the held-out sets first. A label group is consumed by one split only.
    split_order = ["test", "validation", "train"]
    for category in quotas["train"]:
        available = by_category[category]
        cursor = 0
        for split in split_order:
            target = quotas[split][category]
            selected = []
            while len(selected) < target and cursor < len(available):
                group = available[cursor]
                cursor += 1
                remaining = target - len(selected)
                # Multiple writers of the same expression may stay in one split.
                selected.extend(group[:remaining])
            if len(selected) != target:
                raise RuntimeError(
                    "Not enough compatible samples for {}:{} (wanted {}, got {}). "
                    "Reduce the quota or relax normalization explicitly.".format(split, category, target, len(selected))
                )
            for sample in selected:
                sample = dict(sample)
                sample["split"] = split
                result[split].append(sample)
    for samples in result.values():
        rng.shuffle(samples)
    return result


def write_manifest(path: Path, rows: Iterable[dict]) -> None:
    fields = ["sample_id", "image_path", "label", "category", "source", "token_count"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def materialize(splits: Dict[str, List[dict]], output_dir: Path) -> None:
    from tamer.university.image_io import write_image

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, samples in splits.items():
        image_dir = output_dir / "clean" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for index, sample in enumerate(samples, start=1):
            _, traces = parse_inkml(Path(sample["inkml_path"]))
            image = rasterize(traces)
            image_path = image_dir / (sample["sample_id"] + ".png")
            if not write_image(image_path, image):
                raise IOError("Could not write {}".format(image_path))
            row = dict(sample)
            row["image_path"] = image_path.relative_to(output_dir).as_posix()
            written.append(row)
            if index % 1000 == 0:
                print("Rasterized {}/{} {} samples".format(index, len(samples), split))
        write_manifest(split_dir / (split + "_clean.csv"), written)


def verify_no_leakage(splits: Dict[str, List[dict]]) -> None:
    labels = {name: {row["label"] for row in rows} for name, rows in splits.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = labels[left] & labels[right]
        if overlap:
            raise AssertionError("Label leakage between {} and {}: {} labels".format(left, right, len(overlap)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mathwriting", required=True, help="Path containing MathWriting train/")
    parser.add_argument("--dictionary", required=True, help="HME100K dictionary.txt")
    parser.add_argument("--output", default="data/university")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary = load_vocabulary(args.dictionary)
    candidates, rejected = scan_candidates(Path(args.mathwriting) / "train", vocabulary, args.max_tokens)
    summary = {
        "accepted": len(candidates),
        "accepted_by_category": Counter(row["category"] for row in candidates),
        "unique_labels_by_category": {
            category: len({row["label"] for row in candidates if row["category"] == category})
            for category in DEFAULT_QUOTAS["train"]
        },
        "rejected": rejected,
        "seed": args.seed,
        "quotas": DEFAULT_QUOTAS,
    }
    with (output_dir / "selection_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    splits = grouped_split(candidates, DEFAULT_QUOTAS, args.seed)
    verify_no_leakage(splits)
    print(
        "Verified split sizes: {}".format(
            {name: len(rows) for name, rows in splits.items()}
        )
    )
    if args.scan_only:
        return
    materialize(splits, output_dir)
    print("Prepared {} samples with zero canonical-label leakage".format(sum(map(len, splits.values()))))


if __name__ == "__main__":
    main()
