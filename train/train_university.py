"""Fine-tune the version_3 fusion checkpoint on university mathematics."""

import argparse
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from tamer.datamodule.university_datamodule import UniversityDataModule
from tamer.lit_university import LitUniversityTAMER


def adapter_enabled(model_overrides: dict) -> bool:
    return bool(
        model_overrides.get("use_encoder_adapter", False)
        or model_overrides.get("use_decoder_adapter", False)
    )


def validate_adapter_checkpoint_load(model, checkpoint_path: str) -> None:
    """Ensure a non-strict pretrained load skipped only new adapter weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_keys = set(checkpoint["state_dict"])
    model_keys = set(model.state_dict())
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    invalid_missing = [key for key in missing if "adapter" not in key]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Unsafe pretrained load. Non-adapter missing keys: {} | unexpected keys: {}".format(
                invalid_missing, unexpected
            )
        )
    if not missing:
        raise RuntimeError("Adapter run expected new adapter weights, but none were missing")
    print("Initialized {} new adapter tensors; pretrained backbone loaded exactly.".format(len(missing)))


def configure_cuda(performance_config: dict) -> None:
    """Apply optional Ampere optimizations and print the actual training GPU."""
    allow_tf32 = bool(performance_config.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(
            performance_config.get("float32_matmul_precision", "high")
        )
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        memory_gib = properties.total_memory / (1024 ** 3)
        print(
            "CUDA device: {} | VRAM: {:.1f} GiB | capability: {}.{} | TF32: {}".format(
                properties.name,
                memory_gib,
                properties.major,
                properties.minor,
                allow_tf32,
            )
        )


def build_loggers(config: dict, output_dir: str):
    loggers = [CSVLogger(save_dir=output_dir, name="logs")]
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return loggers[0]

    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError as error:
        raise RuntimeError(
            "W&B logging is enabled; install it with: python -m pip install wandb"
        ) from error

    wandb_logger = WandbLogger(
        project=wandb_config.get("project", "University-TAMER"),
        entity=wandb_config.get("entity") or None,
        name=wandb_config.get("name", "v3-fusion-baseline-seed7"),
        id=wandb_config.get("id", "university-tamer-v3-fusion-seed7"),
        resume=wandb_config.get("resume", "allow"),
        save_dir=wandb_config.get("save_dir", output_dir),
        log_model=wandb_config.get("log_model", False),
    )
    wandb_logger.log_hyperparams(config)
    loggers.append(wandb_logger)
    return loggers


def enforce_train_safety(model: LitUniversityTAMER, safety_config: dict) -> dict:
    """Print and validate the number of trainable parameters before training."""
    summary = model.print_trainable_parameter_summary()
    trainable = int(summary["trainable"])
    ratio = float(summary["trainable_ratio"])

    min_trainable_params = safety_config.get("min_trainable_params")
    max_trainable_params = safety_config.get("max_trainable_params")
    max_trainable_ratio = safety_config.get("max_trainable_ratio")

    if min_trainable_params is not None and trainable < int(min_trainable_params):
        raise RuntimeError(
            "Train safety failed: trainable parameters {:,} < min_trainable_params {:,}".format(
                trainable, int(min_trainable_params)
            )
        )
    if max_trainable_params is not None and trainable > int(max_trainable_params):
        raise RuntimeError(
            "Train safety failed: trainable parameters {:,} > max_trainable_params {:,}".format(
                trainable, int(max_trainable_params)
            )
        )
    if max_trainable_ratio is not None and ratio > float(max_trainable_ratio):
        raise RuntimeError(
            "Train safety failed: trainable ratio {:.4f}% > max_trainable_ratio {:.4f}%".format(
                100.0 * ratio, 100.0 * float(max_trainable_ratio)
            )
        )
    print("Train safety: OK")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/university_baseline.yaml")
    parser.add_argument("--resume", default=None, help="Resume an interrupted fine-tune checkpoint")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load checkpoint, apply train_policy, print trainable parameters, then exit",
    )
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    seed_everything(config.get("seed", 7), workers=True)
    configure_cuda(config.get("performance", {}))
    data = UniversityDataModule(**config["data"])
    checkpoint = config["pretrained_checkpoint"]
    if not Path(checkpoint).is_file():
        raise FileNotFoundError("version_3 checkpoint not found: {}".format(checkpoint))
    model_overrides = dict(config.get("model_overrides", {}))
    selection_config = config.get("selection", {})
    monitor = selection_config.get("monitor", "val_retention_score")
    mode = selection_config.get("mode", "max")
    model_overrides["optimization_monitor"] = monitor
    if args.resume:
        if not Path(args.resume).is_file():
            raise FileNotFoundError("Resume checkpoint not found: {}".format(args.resume))
        # Preserve the calibrated HME baseline stored in last.ckpt. Trainer then
        # restores the optimizer, scheduler, epoch and callback states below.
        model = LitUniversityTAMER.load_from_checkpoint(args.resume, strict=True)
        if model_overrides.get("train_policy"):
            model.train_policy = dict(model_overrides["train_policy"])
            model.apply_train_policy()
    else:
        use_adapter = adapter_enabled(model_overrides)
        adapter_checkpoint = bool(config.get("pretrained_checkpoint_is_adapter_checkpoint", False))
        model = LitUniversityTAMER.load_from_checkpoint(
            checkpoint,
            strict=(not use_adapter) or adapter_checkpoint,
            **model_overrides,
        )
        if use_adapter and not adapter_checkpoint:
            validate_adapter_checkpoint_load(model, checkpoint)
        elif use_adapter and adapter_checkpoint:
            print("Loaded adapter checkpoint strictly: {}".format(checkpoint))
    output_dir = config.get("output_dir", "outputs/university_baseline")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    enforce_train_safety(model, config.get("train_safety", {}))
    if args.dry_run:
        print("Dry run completed; no training was started.")
        return
    trainer_config = dict(config["trainer"])
    micro_batch = int(config["data"].get("train_batch_size", 1))
    accumulation = int(trainer_config.get("accumulate_grad_batches", 1))
    print(
        "Training batch: micro={} | accumulation={} | effective={} (single GPU)".format(
            micro_batch, accumulation, micro_batch * accumulation
        )
    )
    if args.resume:
        trainer_config["resume_from_checkpoint"] = args.resume
    early_stopping_patience = trainer_config.pop("early_stopping_patience", 4)
    early_stopping_min_delta = trainer_config.pop("early_stopping_min_delta", 0.0)
    if config.get("auto_calibrate_hme_baseline", True) and not args.resume:
        data.setup("fit")
        calibration_trainer = Trainer(
            logger=False,
            enable_checkpointing=False,
            gpus=trainer_config.get("gpus", 1),
            precision=trainer_config.get("precision", 32),
            deterministic=trainer_config.get("deterministic", True),
            num_sanity_val_steps=0,
        )
        validation_results = calibration_trainer.validate(model, datamodule=data, verbose=False)
        baseline = None
        for result in validation_results:
            for key, value in result.items():
                if key.startswith("val_hme_ExpRate"):
                    baseline = float(value)
        if baseline is None:
            raise RuntimeError("Could not calibrate version_3 on the fixed HME validation cache")
        model.hme_baseline_exprate = baseline
        model.hparams.hme_baseline_exprate = baseline
        print("Calibrated version_3 HME validation ExpRate: {:.6f}".format(baseline))
        del calibration_trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    filename = "{epoch:02d}-{%s:.4f}" % monitor
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path(output_dir) / "checkpoints"),
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=int(selection_config.get("save_top_k", 3)),
        save_last=True,
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
        ),
    ]
    trainer = Trainer(
        callbacks=callbacks,
        logger=build_loggers(config, output_dir),
        **trainer_config,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.fit(model, datamodule=data)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    adapter_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "adapter" in name
    )
    print(
        "Parameters: total={:,} | adapters={:,} ({:.4f}%)".format(
            total_parameters,
            adapter_parameters,
            100.0 * adapter_parameters / max(total_parameters, 1),
        )
    )
    print("Best checkpoint by {}: {}".format(monitor, checkpoint_callback.best_model_path))
    if torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(
            "Peak CUDA memory: allocated={:.2f} GiB | reserved={:.2f} GiB".format(
                allocated, reserved
            )
        )


if __name__ == "__main__":
    main()
