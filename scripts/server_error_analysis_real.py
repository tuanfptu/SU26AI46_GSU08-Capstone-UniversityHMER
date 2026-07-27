#!/usr/bin/env python3
import argparse, csv, json, shutil
from collections import Counter, defaultdict
from pathlib import Path

CONFUSIONS = {
    "b_to_6": [("b", "6"), ("6", "b")],
    "gamma_to_y": [(r"\gamma", "y"), ("y", r"\gamma")],
    "alpha_to_a": [(r"\alpha", "a"), ("a", r"\alpha")],
    "theta_to_0_o": [(r"\theta", "0"), (r"\theta", "o"), ("0", r"\theta"), ("o", r"\theta")],
    "k_to_h": [("k", "h"), ("h", "k")],
    "cdot_to_dot": [(r"\cdot", "."), (".", r"\cdot")],
    "ln_split": [(r"\ln", "l n"), ("l n", r"\ln")],
    "log_split": [(r"\log", "l o g"), ("l o g", r"\log")],
}

def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def read_manifest(p):
    with open(p, encoding="utf-8", newline="") as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}

def write_csv(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def dist(a, b):
    aa, bb = a.split(), b.split()
    dp = list(range(len(bb) + 1))
    for i, x in enumerate(aa, 1):
        ndp = [i]
        for j, y in enumerate(bb, 1):
            ndp.append(min(dp[j] + 1, ndp[j-1] + 1, dp[j-1] + (x != y)))
        dp = ndp
    return dp[-1]

def load_preds(model_dirs):
    out = {}
    for name, d in model_dirs.items():
        p = Path(d) / "predictions.json"
        if not p.exists():
            print("WARNING missing", name, p)
            continue
        out[name] = {r["sample_id"]: r for r in read_json(p)}
    return out

def merge(manifest, preds):
    rows = []
    for sid, meta in manifest.items():
        row = {
            "sample_id": sid,
            "image_path": meta.get("image_path", ""),
            "category": meta.get("category", ""),
            "severity": meta.get("severity", ""),
            "token_count": meta.get("token_count", ""),
            "gt": meta.get("label", ""),
        }
        for model, by_id in preds.items():
            pr = by_id.get(sid, {})
            pred = pr.get("pred", pr.get("raw_prediction", ""))
            gt = pr.get("gt", row["gt"])
            ed = pr.get("edit_distance")
            if ed is None and pred:
                ed = dist(pred, gt)
            row[f"{model}_pred"] = pred
            row[f"{model}_edit_distance"] = ed if ed is not None else ""
            row[f"{model}_exact"] = int(ed == 0) if ed != "" else ""
        rows.append(row)
    return rows

def pairwise(rows, left, right):
    counts = Counter()
    cases = []
    for r in rows:
        ld = int(r[f"{left}_edit_distance"])
        rd = int(r[f"{right}_edit_distance"])
        if ld == 0 and rd == 0:
            tag = "both_correct"
        elif ld == 0:
            tag = f"{left}_only_correct"
        elif rd == 0:
            tag = f"{right}_only_correct"
        elif ld < rd:
            tag = f"{left}_closer"
        elif rd < ld:
            tag = f"{right}_closer"
        else:
            tag = "same_wrong_distance"
        counts[tag] += 1
        row = dict(r)
        row["pairwise_tag"] = tag
        row["distance_gap_abs"] = abs(ld - rd)
        cases.append(row)
    return counts, cases

def group_summary(rows, models):
    out = []
    for m in models:
        for key in ["category", "severity"]:
            groups = defaultdict(list)
            for r in rows:
                groups[r.get(key) or "unknown"].append(r)
            for g, subset in sorted(groups.items()):
                total = len(subset)
                exact = sum(int(r[f"{m}_edit_distance"]) == 0 for r in subset)
                le1 = sum(int(r[f"{m}_edit_distance"]) <= 1 for r in subset)
                le2 = sum(int(r[f"{m}_edit_distance"]) <= 2 for r in subset)
                edit = sum(int(r[f"{m}_edit_distance"]) for r in subset)
                toks = sum(len(r["gt"].split()) for r in subset)
                out.append({
                    "model": m, "group_type": key, "group": g, "count": total,
                    "ExpRate": exact / total, "ExpRate_le_1": le1 / total,
                    "ExpRate_le_2": le2 / total, "TokenErrorRate": edit / max(toks, 1),
                })
    return out

def conf_count(gt, pred, pairs):
    c = 0
    gt = " ".join(gt.split())
    pred = " ".join(pred.split())
    for a, b in pairs:
        if a in gt and b in pred:
            c += min(gt.count(a), pred.count(b))
    return c

def symbol_confusions(rows, models):
    out = []
    for m in models:
        for name, pairs in CONFUSIONS.items():
            out.append({
                "model": m,
                "confusion": name,
                "count": sum(conf_count(r["gt"], r[f"{m}_pred"], pairs) for r in rows),
            })
    return out

def reps(pair_cases):
    by = defaultdict(list)
    for r in pair_cases:
        by[r["pairwise_tag"]].append(r)
    selected = []
    for tag in [
        "a3_realft_only_correct",
        "unimumer_lora_only_correct",
        "a3_realft_closer",
        "unimumer_lora_closer",
        "same_wrong_distance",
    ]:
        selected += sorted(by.get(tag, []), key=lambda r: (-int(r["distance_gap_abs"]), -int(r.get("token_count") or 0)))[:5]
    return selected

def copy_images(rows, data_root, out):
    img_dir = Path(out) / "case_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, r in enumerate(rows, 1):
        src = Path(data_root) / r["image_path"]
        if not src.exists():
            continue
        dst = img_dir / f"{i:02d}_{r['pairwise_tag']}_{r['sample_id']}{src.suffix}"
        shutil.copy2(src, dst)
        rr = dict(r)
        rr["case_index"] = i
        rr["copied_image"] = str(dst)
        meta.append(rr)
    write_csv(img_dir / "case_metadata.csv", meta)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--left", default="a3_realft")
    ap.add_argument("--right", default="unimumer_lora")
    args = ap.parse_args()

    model_dirs = dict(x.split("=", 1) for x in args.model)
    manifest = read_manifest(args.manifest)
    preds = load_preds(model_dirs)
    rows = merge(manifest, preds)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    write_csv(out / "per_sample_comparison.csv", rows)
    write_csv(out / "summary_by_group.csv", group_summary(rows, list(preds)))
    write_csv(out / "symbol_confusion_counts.csv", symbol_confusions(rows, list(preds)))

    counts, pair_cases = pairwise(rows, args.left, args.right)
    with open(out / f"{args.left}_vs_{args.right}_counts.json", "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)

    write_csv(out / f"{args.left}_vs_{args.right}_cases.csv", pair_cases)
    selected = reps(pair_cases)
    write_csv(out / "representative_cases.csv", selected)
    copy_images(selected, args.data_root, out)

    print("Wrote", out)
    print(json.dumps(counts, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
