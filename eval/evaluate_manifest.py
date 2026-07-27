"""Evaluate a fixed manifest with the agreed ExpRate/TER/validity metrics."""

import argparse
import json
import time
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything

from tamer.datamodule.university_datamodule import ManifestTestDataModule
from tamer.lit_university import LitUniversityTAMER


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    seed_everything(7, workers=True)
    data = ManifestTestDataModule(
        dictionary=args.dictionary,
        manifest=args.manifest,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = LitUniversityTAMER.load_from_checkpoint(
        args.checkpoint, prediction_output_dir=args.output, strict=True, use_fusion=True
    )
    trainer = Trainer(logger=False, gpus=args.gpus)
    start = time.perf_counter()
    result = trainer.test(model, datamodule=data)
    elapsed = time.perf_counter() - start
    metrics_path = Path(args.output) / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    count = report["overall"]["count"]
    report["runtime"] = {
        "total_seconds": elapsed,
        "average_seconds_per_image": elapsed / max(count, 1),
        "note": "Includes dataloader and metric overhead; use only under identical hardware/settings.",
    }
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
