"""Evaluate Uni-MuMER on a TAMER-style manifest.

This script keeps the VLM branch separate from TAMER while reusing the same
manifest format and metric writer for fair comparison.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor

from tamer.university.latex import tokenize_latex
from tamer.university.metrics import write_metric_report


DEFAULT_PROMPT = (
    "<|im_start|>system\n"
    "You are a mathematical OCR system. Return only LaTeX, without explanation."
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "Convert the mathematical formula in this image to LaTeX format. "
    "Return only the LaTeX expression."
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty manifest: {}".format(path))
    required = {"sample_id", "image_path", "label", "category", "source"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError("Manifest {} is missing {}".format(path, sorted(missing)))
    return rows


def resolve_image(data_root: str, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return Path(data_root) / path


def load_model(model_name: str, lora_path: Optional[str], dtype_name: str):
    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    kwargs = {"device_map": "auto", "trust_remote_code": True}
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    model = AutoModelForMultimodalLM.from_pretrained(model_name, **kwargs)
    if lora_path:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("LoRA evaluation needs peft: python -m pip install peft") from error
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return processor, model


def first_model_device(model) -> torch.device:
    return next(model.parameters()).device


def generate_latex(processor, model, image: Image.Image, prompt: str, generation: dict) -> str:
    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    device = first_model_device(model)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(generation.get("max_new_tokens", 512)),
            do_sample=bool(generation.get("do_sample", False)),
            num_beams=int(generation.get("num_beams", 1)),
            repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
        )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def manifest_for_split(config: dict, split: str) -> str:
    data = config["data"]
    keys = {
        "validation": "validation_manifest",
        "val": "validation_manifest",
        "test": "test_manifest",
        "blind": "test_manifest",
        "train": "train_manifest",
    }
    if split not in keys:
        raise ValueError("Unknown split '{}'. Use validation, test, or train.".format(split))
    key = keys[split]
    if key not in data:
        raise KeyError("Config data section is missing {}".format(key))
    return data[key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "val", "test", "blind", "train"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    manifest = manifest_for_split(config, args.split)
    data_root = config["data"]["data_root"]
    output_dir = args.output or str(Path(config["output_dir"]) / args.split)
    prompt = config.get("prompt") or DEFAULT_PROMPT
    generation = config.get("generation", {})
    lora_path = args.lora_path or config.get("lora_path")

    rows = read_manifest(manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    processor, model = load_model(config["model_name"], lora_path, args.dtype)
    records = []
    started = time.perf_counter()

    for row in tqdm(rows, desc="Uni-MuMER {}".format(args.split)):
        image_path = resolve_image(data_root, row["image_path"])
        image = Image.open(image_path).convert("RGB")
        raw_prediction = generate_latex(processor, model, image, prompt, generation)
        pred_tokens = tokenize_latex(raw_prediction)
        records.append(
            {
                "sample_id": row["sample_id"],
                "pred_tokens": pred_tokens,
                "gt_tokens": row["label"].split(),
                "category": row.get("category", "unknown"),
                "source": row.get("source", "unknown"),
                "severity": row.get("severity", "unknown"),
                "raw_prediction": raw_prediction,
            }
        )

    elapsed = time.perf_counter() - started
    report = write_metric_report(
        records,
        output_dir,
        extra={
            "model": config["model_name"],
            "split": args.split,
            "manifest": manifest,
            "lora_path": lora_path,
            "generation": generation,
            "runtime": {
                "total_seconds": elapsed,
                "average_seconds_per_image": elapsed / max(len(records), 1),
            },
        },
    )
    with (Path(output_dir) / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
