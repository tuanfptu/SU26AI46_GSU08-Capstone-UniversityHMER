"""TAMER fine-tuning module with replay-aware validation and fixed metrics."""

from pathlib import Path
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.optim as optim

from tamer.datamodule import Batch, vocab
from tamer.lit_tamer import LitTAMER
from tamer.university.metrics import write_metric_report
from tamer.utils.utils import ExpRateRecorder, ce_loss, to_bi_tgt_out, to_struct_output


class LitUniversityTAMER(LitTAMER):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        learning_rate: float = 5e-5,
        patience: int = 4,
        milestones: List[int] = (10, 16),
        vocab_size: int = 248,
        weight_decay: float = 1e-4,
        freeze_encoder_epochs: int = 2,
        hme_baseline_exprate: float = 0.6954,
        max_hme_drop: float = 0.02,
        retention_penalty: float = 10.0,
        prediction_output_dir: str = "outputs/predictions",
        use_fusion: bool = False,
        use_encoder_adapter: bool = False,
        use_decoder_adapter: bool = False,
        adapter_bottleneck_dim: int = 64,
        adapter_dropout: float = 0.1,
        adapter_gate_init_bias: float = -2.0,
        optimization_monitor: str = "val_retention_score",
        train_policy: Optional[Dict[str, bool]] = None,
    ) -> None:
        super().__init__(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            beam_size=beam_size,
            max_len=max_len,
            alpha=alpha,
            early_stopping=early_stopping,
            temperature=temperature,
            learning_rate=learning_rate,
            patience=patience,
            milestones=list(milestones),
            vocab_size=vocab_size,
            use_fusion=use_fusion,
            use_encoder_adapter=use_encoder_adapter,
            use_decoder_adapter=use_decoder_adapter,
            adapter_bottleneck_dim=adapter_bottleneck_dim,
            adapter_dropout=adapter_dropout,
            adapter_gate_init_bias=adapter_gate_init_bias,
        )
        self.save_hyperparameters()
        self.weight_decay = weight_decay
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.hme_baseline_exprate = hme_baseline_exprate
        self.max_hme_drop = max_hme_drop
        self.retention_penalty = retention_penalty
        self.prediction_output_dir = prediction_output_dir
        self.optimization_monitor = optimization_monitor
        self.train_policy = dict(train_policy or {})
        self.university_exprate = ExpRateRecorder()
        self.hme_exprate = ExpRateRecorder()
        if self.train_policy:
            self.apply_train_policy()

    @staticmethod
    def _set_module_trainable(module, trainable: bool) -> None:
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def _unfreeze_if_requested(self, policy_key: str, module) -> None:
        if self.train_policy.get(policy_key, False):
            self._set_module_trainable(module, True)

    def apply_train_policy(self) -> None:
        """Apply an explicit parameter-efficient fine-tuning policy.

        When ``train_policy`` is not provided, training keeps the original behavior
        used by the previous phase1/RealFT experiments. When it is provided, all
        parameters are frozen first and only the requested modules are unfrozen.
        """
        if not self.train_policy:
            return

        for parameter in self.parameters():
            parameter.requires_grad = False

        decoder = self.tamer_model.decoder
        self._unfreeze_if_requested("train_encoder", self.tamer_model.encoder)
        self._unfreeze_if_requested("train_encoder_adapter", self.tamer_model.encoder_adapter)
        self._unfreeze_if_requested("train_decoder_transformer", decoder.model)
        self._unfreeze_if_requested("train_word_embedding", decoder.word_embed)
        self._unfreeze_if_requested("train_positional_encoding", decoder.pos_enc)
        self._unfreeze_if_requested("train_decoder_norm", decoder.norm)
        self._unfreeze_if_requested("train_structsim", decoder.struct_sim)
        self._unfreeze_if_requested("train_decoder_adapter", decoder.decoder_adapter)
        self._unfreeze_if_requested("train_fusion", getattr(decoder, "fusion", None))

        if self.train_policy.get("train_output_projection", False):
            self._set_module_trainable(getattr(decoder, "exp_proj", None), True)
            self._set_module_trainable(getattr(decoder, "fusion_proj", None), True)
            self._set_module_trainable(getattr(decoder, "proj", None), True)

    def trainable_parameter_summary(self) -> Dict[str, object]:
        total = 0
        trainable = 0
        trainable_tensors = []
        frozen_tensors = []
        for name, parameter in self.named_parameters():
            count = parameter.numel()
            total += count
            if parameter.requires_grad:
                trainable += count
                trainable_tensors.append((name, count))
            else:
                frozen_tensors.append((name, count))
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_ratio": trainable / max(total, 1),
            "trainable_tensors": trainable_tensors,
            "frozen_tensors": frozen_tensors,
        }

    def print_trainable_parameter_summary(self, max_lines: int = 80) -> Dict[str, object]:
        summary = self.trainable_parameter_summary()
        print("")
        print("Train policy summary")
        print("--------------------")
        print("Explicit train_policy: {}".format(bool(self.train_policy)))
        if self.train_policy:
            for key in sorted(self.train_policy):
                print("  {}: {}".format(key, self.train_policy[key]))
        print("Total parameters:      {:,}".format(summary["total"]))
        print("Trainable parameters:  {:,}".format(summary["trainable"]))
        print("Frozen parameters:     {:,}".format(summary["frozen"]))
        print("Trainable ratio:       {:.4f}%".format(100.0 * summary["trainable_ratio"]))
        print("")
        print("Trainable tensors:")
        for name, count in summary["trainable_tensors"][:max_lines]:
            print("  {:>10,}  {}".format(count, name))
        remaining = len(summary["trainable_tensors"]) - max_lines
        if remaining > 0:
            print("  ... {} more trainable tensors".format(remaining))
        if not summary["trainable_tensors"]:
            print("  NONE")
        print("")
        return summary

    def on_train_epoch_start(self) -> None:
        if self.train_policy:
            self.apply_train_policy()
            self.log("encoder_frozen", float(not self.train_policy.get("train_encoder", False)), prog_bar=False, on_step=False, on_epoch=True)
            return
        frozen = self.current_epoch < self.freeze_encoder_epochs
        for parameter in self.tamer_model.encoder.parameters():
            parameter.requires_grad = not frozen
        if frozen:
            self.tamer_model.encoder.eval()
        self.log("encoder_frozen", float(frozen), prog_bar=False, on_step=False, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        encoder_adapter = self.tamer_model.encoder_adapter
        decoder_adapter = self.tamer_model.decoder.decoder_adapter
        if encoder_adapter is not None:
            self.log("adapter/encoder_gate", encoder_adapter.gate, on_step=False, on_epoch=True)
        if decoder_adapter is not None:
            self.log("adapter/decoder_gate", decoder_adapter.gate, on_step=False, on_epoch=True)

    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        struct_out, _ = to_struct_output(batch.indices, self.device)
        model_outputs = self(batch.imgs, batch.mask, tgt)
        if self.hparams.use_fusion:
            exp_hat, fusion_hat, sim = model_outputs
            sequence_loss = ce_loss(exp_hat, out) + ce_loss(fusion_hat, out)
        else:
            out_hat, sim = model_outputs
            sequence_loss = ce_loss(out_hat, out)
        loss = sequence_loss + ce_loss(sim, struct_out, ignore_idx=-1)
        prefix = "university" if dataloader_idx == 0 else "hme"
        self.log(
            "val_{}_loss".format(prefix),
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        recorder = self.university_exprate if dataloader_idx == 0 else self.hme_exprate
        recorder([hyp.seq for hyp in hyps], batch.indices)
        self.log(
            "val_{}_ExpRate".format(prefix),
            recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def validation_epoch_end(self, outputs) -> None:
        university = self.university_exprate.compute()
        has_hme_validation = bool(self.hme_exprate.total_line.detach().item() > 0)
        if has_hme_validation:
            hme = self.hme_exprate.compute()
            threshold = self.hme_baseline_exprate - self.max_hme_drop
            violation = torch.relu(torch.as_tensor(threshold, device=self.device) - hme)
            score = university - self.retention_penalty * violation
        else:
            score = university
        self.log("val_ExpRate", university, prog_bar=False, sync_dist=True)
        self.log("val_retention_score", score, prog_bar=True, sync_dist=True)
        if has_hme_validation:
            self.log("val_hme_drop", self.hme_baseline_exprate - hme, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Batch, batch_idx: int):
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        records = []
        categories = batch.categories or ["unknown"] * len(batch)
        sources = batch.sources or ["unknown"] * len(batch)
        severities = batch.severities or ["unknown"] * len(batch)
        for name, hyp, truth, category, source, severity in zip(
            batch.img_bases, hyps, batch.indices, categories, sources, severities
        ):
            records.append(
                {
                    "sample_id": name,
                    "pred_tokens": vocab.indices2words(hyp.seq),
                    "gt_tokens": vocab.indices2words(truth),
                    "category": category,
                    "source": source,
                    "severity": severity,
                }
            )
        return records

    def test_epoch_end(self, outputs) -> None:
        records = [record for batch_records in outputs for record in batch_records]
        report = write_metric_report(records, self.prediction_output_dir)
        overall = report["overall"]
        for name in ("ExpRate", "ExpRate_le_1", "ExpRate_le_2", "TokenErrorRate", "ValidLaTeX"):
            self.log("test_" + name, float(overall[name]))

    def configure_optimizers(self):
        trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters. Check model_overrides.train_policy.")
        optimizer = optim.AdamW(
            trainable_parameters, lr=self.hparams.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(1, self.hparams.patience // 2),
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.optimization_monitor,
                "interval": "epoch",
                "frequency": 1,
            },
        }
