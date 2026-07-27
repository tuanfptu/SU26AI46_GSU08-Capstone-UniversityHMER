"""Evaluate either version_3 or a fine-tuned checkpoint on full HME100K test."""

import argparse
import json
import time
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything

from tamer.datamodule import HMEDatamodule
from tamer.lit_university import LitUniversityTAMER


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hme100k", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    seed_everything(7, workers=True)
    data = HMEDatamodule(
        folder=args.hme100k,
        test_folder="test",
        max_size=480000,
        scale_to_limit=False,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = LitUniversityTAMER.load_from_checkpoint(
        args.checkpoint, prediction_output_dir=args.output, strict=True, use_fusion=True
    )
    trainer = Trainer(logger=False, gpus=args.gpus)
    start = time.perf_counter()
    trainer.test(model, datamodule=data)
    elapsed = time.perf_counter() - start
    metrics_path = Path(args.output) / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    count = report["overall"]["count"]
    report["runtime"] = {
        "total_seconds": elapsed,
        "average_seconds_per_image": elapsed / max(count, 1),
        "note": "Includes dataloader and metric overhead; compare only on identical hardware.",
    }
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
