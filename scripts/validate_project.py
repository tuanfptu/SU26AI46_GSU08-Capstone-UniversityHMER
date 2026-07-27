"""Fail-fast audit before spending GPU hours on a training run."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/university")
    parser.add_argument("--hme-cache", default="data/hme_cache")
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    manifests = {
        "train": root / "splits" / "train_clean.csv",
        "validation": root / "splits" / "validation_fixed.csv",
        "test": root / "splits" / "test_fixed.csv",
    }
    expected = {"train": 10000, "validation": 1000, "test": 1000}
    vocabulary = {line.strip() for line in Path(args.dictionary).read_text(encoding="utf-8").splitlines() if line.strip()}
    rows = {}
    for split, path in manifests.items():
        rows[split] = read_csv(path)
        assert len(rows[split]) == expected[split], "{} has {} rows, expected {}".format(
            split, len(rows[split]), expected[split]
        )
        for row in rows[split]:
            missing = set(row["label"].split()) - vocabulary
            assert not missing, "OOV {} in {}".format(sorted(missing), row["sample_id"])
            assert (root / row["image_path"]).is_file(), "Missing image {}".format(row["image_path"])
    labels = {split: {row["label"] for row in split_rows} for split, split_rows in rows.items()}
    leakage = {
        "train_validation": len(labels["train"] & labels["validation"]),
        "train_test": len(labels["train"] & labels["test"]),
        "validation_test": len(labels["validation"] & labels["test"]),
    }
    assert not any(leakage.values()), "Canonical-label leakage: {}".format(leakage)
    severity_expected = {
        "validation": {"mild": 600, "medium": 300, "hard": 100},
        "test": {"mild": 300, "medium": 400, "hard": 300},
    }
    severity_counts = {}
    for split in ("validation", "test"):
        counts = dict(Counter(row.get("severity", "missing") for row in rows[split]))
        assert counts == severity_expected[split], "Unexpected {} severity counts: {}".format(split, counts)
        severity_counts[split] = counts
    replay = read_csv(Path(args.hme_cache) / "replay.csv")
    hme_validation = read_csv(Path(args.hme_cache) / "validation.csv")
    assert len(replay) == 20000, "HME replay has {} rows, expected 20000".format(len(replay))
    assert len(hme_validation) == 3000, "HME validation has {} rows, expected 3000".format(
        len(hme_validation)
    )
    replay_ids = {row["sample_id"] for row in replay}
    hme_validation_ids = {row["sample_id"] for row in hme_validation}
    hme_overlap = replay_ids & hme_validation_ids
    assert not hme_overlap, "HME replay/validation overlap: {}".format(len(hme_overlap))
    for split_name, split_rows in (("replay", replay), ("validation", hme_validation)):
        for row in split_rows:
            assert row["sample_id"].startswith("hme_train_"), "{} contains non-train sample {}".format(
                split_name, row["sample_id"]
            )
            assert row["source"] == "hme100k_train", "{} has source {}".format(
                row["sample_id"], row["source"]
            )
            assert (Path(args.hme_cache) / row["image_path"]).is_file(), "Missing HME cache image {}".format(
                row["image_path"]
            )
            missing = set(row["label"].split()) - vocabulary
            assert not missing, "HME OOV {} in {}".format(sorted(missing), row["sample_id"])
    checkpoint = Path(args.checkpoint)
    assert checkpoint.is_file(), "Missing checkpoint {}".format(checkpoint)
    report = {
        "status": "PASS",
        "split_sizes": {name: len(value) for name, value in rows.items()},
        "unique_labels": {name: len(value) for name, value in labels.items()},
        "label_leakage": leakage,
        "severity_counts": severity_counts,
        "hme_replay_size": len(replay),
        "hme_validation_size": len(hme_validation),
        "hme_replay_validation_overlap": len(hme_overlap),
        "hme_cache_source": "HME100K train only; official test untouched",
        "vocabulary_size_without_special_tokens": len(vocabulary),
        "checkpoint": str(checkpoint.resolve()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
