"""LoRA fine-tuning for Uni-MuMER on the real classroom train split."""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor, Trainer, TrainerCallback, TrainingArguments

from tamer.university.latex import tokenize_latex
from tamer.university.metrics import compute_metrics, write_metric_report


def read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty manifest: {}".format(path))
    required = {"sample_id", "image_path", "label"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError("Manifest {} is missing {}".format(path, sorted(missing)))
    return rows


class FormulaRows(Dataset):
    def __init__(self, rows: List[Dict[str, str]], data_root: str) -> None:
        self.rows = rows
        self.data_root = Path(data_root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = self.data_root / path
        return {
            "sample_id": row["sample_id"],
            "image": Image.open(path).convert("RGB"),
            "label": row["label"],
        }


class MultimodalCompletionCollator:
    def __init__(self, processor, prompt: str, max_length: int = 2048) -> None:
        self.processor = processor
        self.prompt = prompt
        self.max_length = max_length
        tokenizer = getattr(processor, "tokenizer", processor)
        self.eos_token = tokenizer.eos_token or ""

    def _prompt_length(self, image: Image.Image) -> int:
        encoded = self.processor(
            text=[self.prompt],
            images=[image],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return int(encoded["input_ids"].shape[1])

    def __call__(self, examples: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        images = [example["image"] for example in examples]
        texts = [self.prompt + str(example["label"]).strip() + self.eos_token for example in examples]
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        for row_index, image in enumerate(images):
            prompt_len = min(self._prompt_length(image), labels.shape[1])
            labels[row_index, :prompt_len] = -100
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def resolve_image(data_root: str, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return Path(data_root) / path


def first_model_device(model) -> torch.device:
    return next(model.parameters()).device


def generate_latex(processor, model, image: Image.Image, prompt: str, generation: dict) -> str:
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(first_model_device(model))
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(generation.get("max_new_tokens", 512)),
            do_sample=bool(generation.get("do_sample", False)),
            num_beams=int(generation.get("num_beams", 1)),
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


class RealValidationCallback(TrainerCallback):
    def __init__(
        self,
        processor,
        rows: List[Dict[str, str]],
        data_root: str,
        prompt: str,
        generation: dict,
        output_dir: str,
        monitor: str,
        mode: str,
        patience: int,
        min_delta: float,
        eval_limit: Optional[int] = None,
        use_wandb: bool = False,
    ) -> None:
        self.processor = processor
        self.rows = rows[:eval_limit] if eval_limit else rows
        self.data_root = data_root
        self.prompt = prompt
        self.generation = generation
        self.output_dir = Path(output_dir)
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.use_wandb = use_wandb
        self.best_value = math.inf if mode == "min" else -math.inf
        self.best_epoch = None
        self.bad_epochs = 0
        self.history = []

    def _improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control

        epoch = int(round(state.epoch or 0))
        model.eval()
        started = time.perf_counter()
        records = []

        for row in tqdm(self.rows, desc="Real validation epoch {}".format(epoch)):
            image = Image.open(resolve_image(self.data_root, row["image_path"])).convert("RGB")
            raw_prediction = generate_latex(self.processor, model, image, self.prompt, self.generation)
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "pred_tokens": tokenize_latex(raw_prediction),
                    "gt_tokens": row["label"].split(),
                    "category": row.get("category", "unknown"),
                    "source": row.get("source", "unknown"),
                    "severity": row.get("severity", "unknown"),
                    "raw_prediction": raw_prediction,
                }
            )

        elapsed = time.perf_counter() - started
        metrics = compute_metrics(records)
        if self.monitor not in metrics:
            raise KeyError("Validation metric '{}' not found".format(self.monitor))
        value = float(metrics[self.monitor])

        epoch_dir = self.output_dir / "validation_epochs" / "epoch_{:03d}".format(epoch)
        report = write_metric_report(
            records,
            str(epoch_dir),
            extra={
                "split": "validation",
                "epoch": epoch,
                "generation": self.generation,
                "runtime": {
                    "total_seconds": elapsed,
                    "average_seconds_per_image": elapsed / max(len(records), 1),
                },
            },
        )

        if self._improved(value):
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0
            best_dir = self.output_dir / "best_adapter"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            self.processor.save_pretrained(best_dir)
            with (best_dir / "best_metrics.json").open("w", encoding="utf-8") as stream:
                json.dump(report["overall"], stream, ensure_ascii=False, indent=2)
        else:
            self.bad_epochs += 1

        summary = {
            "epoch": epoch,
            "monitor": self.monitor,
            "monitor_value": value,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            **{"validation_" + key: val for key, val in metrics.items()},
        }
        self.history.append(summary)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "validation_history.json").open("w", encoding="utf-8") as stream:
            json.dump(self.history, stream, ensure_ascii=False, indent=2)

        if self.use_wandb:
            try:
                import wandb
                wandb.log({"real_validation/" + key: val for key, val in metrics.items()}, step=state.global_step)
                wandb.log(
                    {
                        "real_validation/monitor_value": value,
                        "real_validation/best_value": self.best_value,
                        "real_validation/bad_epochs": self.bad_epochs,
                    },
                    step=state.global_step,
                )
            except Exception as error:
                print("W&B validation logging failed: {}".format(error))

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if self.bad_epochs >= self.patience:
            print("Early stopping: {} did not improve for {} validation epochs.".format(self.monitor, self.bad_epochs))
            control.should_training_stop = True

        model.train()
        return control


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:
        raise RuntimeError("LoRA training needs peft: python -m pip install peft") from error

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    model_name = config["model_name"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    lora_config = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_config.get("r", 8)),
            lora_alpha=int(lora_config.get("alpha", 16)),
            lora_dropout=float(lora_config.get("dropout", 0.05)),
            target_modules=list(lora_config["target_modules"]),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model.print_trainable_parameters()

    rows = read_manifest(config["data"]["train_manifest"])
    if args.limit is not None:
        rows = rows[: args.limit]
    dataset = FormulaRows(rows, config["data"]["data_root"])
    train_config = config["train"]
    output_dir = config["output_dir"]
    validation_config = config.get("validation", {})
    wandb_config = config.get("wandb", {})
    use_wandb = bool(wandb_config.get("enabled", False))

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=wandb_config.get("run_name"),
        num_train_epochs=float(train_config.get("epochs", 8)),
        per_device_train_batch_size=int(train_config.get("batch_size", 1)),
        gradient_accumulation_steps=int(train_config.get("grad_accum_steps", 8)),
        learning_rate=float(train_config.get("learning_rate", 5e-5)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.05)),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        logging_steps=int(train_config.get("logging_steps", 10)),
        save_strategy=train_config.get("save_strategy", "epoch"),
        save_total_limit=int(train_config.get("save_total_limit", 3)),
        fp16=bool(train_config.get("fp16", False)),
        bf16=bool(train_config.get("bf16", args.dtype == "bfloat16")),
        gradient_checkpointing=bool(train_config.get("gradient_checkpointing", False)),
        remove_unused_columns=False,
        report_to=["wandb"] if use_wandb else [],
    )

    callbacks = []
    if bool(validation_config.get("enabled", True)):
        validation_rows = read_manifest(config["data"]["validation_manifest"])
        callbacks.append(
            RealValidationCallback(
                processor=processor,
                rows=validation_rows,
                data_root=config["data"]["data_root"],
                prompt=config["prompt"],
                generation=validation_config.get("generation", config.get("generation", {})),
                output_dir=output_dir,
                monitor=validation_config.get("metric", "TokenErrorRate"),
                mode=validation_config.get("mode", "min"),
                patience=int(validation_config.get("patience", 2)),
                min_delta=float(validation_config.get("min_delta", 0.001)),
                eval_limit=validation_config.get("limit"),
                use_wandb=use_wandb,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=MultimodalCompletionCollator(
            processor,
            config["prompt"],
            max_length=int(train_config.get("cutoff_len", 2048)),
        ),
        callbacks=callbacks,
    )
    trainer.train()

    adapter_dir = Path(output_dir) / "final_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print("Saved final LoRA adapter to {}".format(adapter_dir))


if __name__ == "__main__":
    main()
