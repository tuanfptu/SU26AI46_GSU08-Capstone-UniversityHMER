"""Evaluation metrics used consistently across every baseline experiment."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import editdistance

from .latex import latex_is_syntactically_valid


def compute_metrics(records: Sequence[dict]) -> Dict[str, float]:
    total = len(records)
    if total == 0:
        raise ValueError("Cannot compute metrics for an empty prediction set")
    exact = within_one = within_two = valid = edit_sum = token_sum = 0
    for record in records:
        prediction = record["pred_tokens"]
        truth = record["gt_tokens"]
        distance = int(editdistance.eval(prediction, truth))
        record["edit_distance"] = distance
        exact += int(distance == 0)
        within_one += int(distance <= 1)
        within_two += int(distance <= 2)
        edit_sum += distance
        token_sum += len(truth)
        valid += int(latex_is_syntactically_valid(prediction))
    return {
        "count": total,
        "ExpRate": exact / total,
        "ExpRate_le_1": within_one / total,
        "ExpRate_le_2": within_two / total,
        "TokenErrorRate": edit_sum / max(token_sum, 1),
        "ValidLaTeX": valid / total,
        "total_edit_distance": edit_sum,
        "total_gt_tokens": token_sum,
    }


def compute_group_metrics(records: Sequence[dict], key: str) -> Dict[str, Dict[str, float]]:
    groups = defaultdict(list)
    for record in records:
        groups[record.get(key) or "unknown"].append(record)
    return {name: compute_metrics(rows) for name, rows in sorted(groups.items())}


def write_metric_report(records: List[dict], output_dir: str, extra: dict = None) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    overall = compute_metrics(records)
    categories = compute_group_metrics(records, "category")
    severities = compute_group_metrics(records, "severity")
    report = {"overall": overall, "by_category": categories, "by_severity": severities}
    if extra:
        report.update(extra)
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    serializable = []
    for record in records:
        row = dict(record)
        row["pred"] = " ".join(row.pop("pred_tokens"))
        row["gt"] = " ".join(row.pop("gt_tokens"))
        serializable.append(row)
    with (output / "predictions.json").open("w", encoding="utf-8") as stream:
        json.dump(serializable, stream, ensure_ascii=False, indent=2)
    with (output / "category_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["category", "count", "ExpRate", "ExpRate_le_1", "ExpRate_le_2", "TokenErrorRate", "ValidLaTeX"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for category, metrics in categories.items():
            writer.writerow({"category": category, **{field: metrics[field] for field in fields[1:]}})
    with (output / "severity_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["severity", "count", "ExpRate", "ExpRate_le_1", "ExpRate_le_2", "TokenErrorRate", "ValidLaTeX"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for severity, metrics in severities.items():
            writer.writerow({"severity": severity, **{field: metrics[field] for field in fields[1:]}})
    return report
