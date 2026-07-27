"""Summarize best University-validation epochs for the phase-1 runs."""

import csv
from pathlib import Path


RUNS = {
    "A0 control": Path("outputs/phase1_a0_control_seed7"),
    "A1 encoder": Path("outputs/phase1_a1_encoder_seed7"),
    "A2 decoder": Path("outputs/phase1_a2_decoder_seed7"),
    "A3 dual": Path("outputs/phase1_a3_dual_seed7"),
}


def latest_metrics(root: Path) -> Path:
    files = list((root / "logs").glob("version_*/metrics.csv"))
    if not files:
        raise FileNotFoundError("No metrics.csv under {}".format(root))
    return max(files, key=lambda path: int(path.parent.name.split("_")[-1]))


def best_row(path: Path):
    by_epoch = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            epoch = row.get("epoch", "")
            if epoch == "":
                continue
            merged = by_epoch.setdefault(epoch, {})
            for key, value in row.items():
                if value not in (None, ""):
                    merged[key] = value
    candidates = []
    for row in by_epoch.values():
        university = row.get("val_university_ExpRate/dataloader_idx_0", row.get("val_university_ExpRate", ""))
        if university != "":
            row["_university"] = float(university)
            candidates.append(row)
    if not candidates:
        raise RuntimeError("No University validation metric in {}".format(path))
    return max(candidates, key=lambda row: row["_university"])


def checkpoint_for_epoch(root: Path, epoch: int) -> str:
    matches = sorted((root / "checkpoints").glob("epoch={:02d}-*.ckpt".format(epoch)))
    return str(matches[-1]) if matches else "CHECK_FILENAME"


def main() -> None:
    print("| Run | Best epoch | University val | HME val | Encoder gate | Decoder gate | Checkpoint |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for name, root in RUNS.items():
        row = best_row(latest_metrics(root))
        epoch = int(float(row["epoch"]))
        hme = row.get("val_hme_ExpRate/dataloader_idx_1", row.get("val_hme_ExpRate", ""))
        encoder_gate = row.get("adapter/encoder_gate", "")
        decoder_gate = row.get("adapter/decoder_gate", "")
        print(
            "| {} | {} | {:.4f} | {} | {} | {} | {} |".format(
                name,
                epoch,
                row["_university"],
                "{:.4f}".format(float(hme)) if hme else "-",
                "{:.4f}".format(float(encoder_gate)) if encoder_gate else "-",
                "{:.4f}".format(float(decoder_gate)) if decoder_gate else "-",
                checkpoint_for_epoch(root, epoch),
            )
        )


if __name__ == "__main__":
    main()
