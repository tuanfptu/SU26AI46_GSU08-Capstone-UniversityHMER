"""Balanced MathWriting/HME100K datamodule used by the baseline fine-tune."""

import csv
import hashlib
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import pytorch_lightning as pl
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from tamer.university.augmentation import DynamicPaperAugmentation
from tamer.university.image_io import read_grayscale

from .datamodule import Batch
from .vocab import vocab


def _read_manifest(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty manifest: {}".format(path))
    required = {"sample_id", "image_path", "label", "category", "source"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError("Manifest {} is missing {}".format(path, sorted(missing)))
    return rows


def _resolve_image(data_root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(data_root, path)


def _augmentation_seed(global_seed: int, epoch: int, sample_id: str) -> int:
    value = "{}:{}:{}".format(global_seed, epoch, sample_id).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


class FormulaManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        data_root: Optional[str] = None,
        dynamic_augmentation: bool = False,
        background_dir: Optional[str] = None,
        seed: int = 7,
    ) -> None:
        self.manifest = os.path.abspath(manifest)
        self.data_root = os.path.abspath(data_root or os.path.dirname(os.path.dirname(self.manifest)))
        self.rows = _read_manifest(self.manifest)
        self.seed = seed
        self.augmenter = DynamicPaperAugmentation(background_dir) if dynamic_augmentation else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            index, epoch = item
        else:
            index, epoch = item, 0
        row = self.rows[index]
        path = _resolve_image(self.data_root, row["image_path"])
        image = read_grayscale(path)
        if image is None:
            raise FileNotFoundError(path)
        if self.augmenter is not None:
            image = self.augmenter(image, seed=_augmentation_seed(self.seed, epoch, row["sample_id"]))
        tensor = to_tensor(image)
        return (
            row["sample_id"],
            tensor,
            row["label"].split(),
            row["category"],
            row["source"],
            row.get("severity", "dynamic" if self.augmenter is not None else "unknown"),
        )


class CombinedDataset(Dataset):
    def __init__(self, university: FormulaManifestDataset, replay: FormulaManifestDataset) -> None:
        self.university = university
        self.replay = replay
        self.offset = len(university)

    def __len__(self) -> int:
        return len(self.university) + len(self.replay)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            index, epoch = item
        else:
            index, epoch = item, 0
        if index < self.offset:
            return self.university[(index, epoch)]
        return self.replay[(index - self.offset, epoch)]


class BalancedReplayBatchSampler(BatchSampler):
    """Visit every university sample and maintain the requested replay ratio.

    Fractional replay counts use an accumulator. For batch_size=8 and ratio=.4,
    the HME counts repeat 3, 3, 4, 3, 3, which is exactly 40% over five batches.
    """

    def __init__(
        self,
        university_size: int,
        replay_size: int,
        batch_size: int,
        replay_ratio: float,
        seed: int,
    ) -> None:
        if not 0.0 < replay_ratio < 1.0:
            raise ValueError("replay_ratio must be between 0 and 1")
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2")
        self.university_size = university_size
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.replay_ratio = replay_ratio
        self.seed = seed
        self.epoch = 0
        average_university = batch_size * (1.0 - replay_ratio)
        self.num_batches = int(math.ceil(university_size / average_university))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        university = list(range(self.university_size))
        replay = list(range(self.replay_size))
        rng.shuffle(university)
        rng.shuffle(replay)
        university_cursor = 0
        replay_cursor = 0
        replay_accumulator = 0.0
        current_epoch = self.epoch
        self.epoch += 1
        for _ in range(self.num_batches):
            replay_accumulator += self.batch_size * self.replay_ratio
            replay_count = int(replay_accumulator)
            replay_accumulator -= replay_count
            replay_count = max(1, min(self.batch_size - 1, replay_count))
            university_count = self.batch_size - replay_count
            batch = []
            for _ in range(university_count):
                if university_cursor >= len(university):
                    rng.shuffle(university)
                    university_cursor = 0
                batch.append((university[university_cursor], current_epoch))
                university_cursor += 1
            for _ in range(replay_count):
                if replay_cursor >= len(replay):
                    rng.shuffle(replay)
                    replay_cursor = 0
                batch.append((self.university_size + replay[replay_cursor], current_epoch))
                replay_cursor += 1
            rng.shuffle(batch)
            yield batch


def collate_formula_samples(samples: Sequence[Tuple]) -> Batch:
    names = [sample[0] for sample in samples]
    images = [sample[1] for sample in samples]
    labels = [vocab.words2indices(sample[2]) for sample in samples]
    categories = [sample[3] for sample in samples]
    sources = [sample[4] for sample in samples]
    severities = [sample[5] for sample in samples]
    heights = [image.size(1) for image in images]
    widths = [image.size(2) for image in images]
    batch = torch.zeros(len(images), 1, max(heights), max(widths))
    mask = torch.ones(len(images), max(heights), max(widths), dtype=torch.bool)
    for index, image in enumerate(images):
        batch[index, :, : heights[index], : widths[index]] = image
        mask[index, : heights[index], : widths[index]] = 0
    return Batch(
        names,
        batch,
        mask,
        labels,
        categories=categories,
        sources=sources,
        severities=severities,
    )


class UniversityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dictionary: str,
        university_train_manifest: str,
        university_val_manifest: str,
        hme_replay_manifest: str,
        hme_val_manifest: str,
        university_data_root: Optional[str] = None,
        hme_cache_root: Optional[str] = None,
        background_dir: Optional[str] = None,
        train_batch_size: int = 8,
        eval_batch_size: int = 2,
        replay_ratio: float = 0.40,
        dynamic_augmentation: bool = True,
        enable_hme_validation: bool = True,
        num_workers: int = 4,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.dictionary = dictionary
        self.university_train_manifest = university_train_manifest
        self.university_val_manifest = university_val_manifest
        self.hme_replay_manifest = hme_replay_manifest
        self.hme_val_manifest = hme_val_manifest
        self.university_data_root = university_data_root
        self.hme_cache_root = hme_cache_root
        self.background_dir = background_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.replay_ratio = replay_ratio
        self.dynamic_augmentation = dynamic_augmentation
        self.enable_hme_validation = enable_hme_validation
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        vocab.init(dictionary)

    def _worker_options(self) -> dict:
        """Options that are only valid when worker processes are enabled."""
        if self.num_workers <= 0:
            return {}
        return {
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
        }

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            university = FormulaManifestDataset(
                self.university_train_manifest,
                self.university_data_root,
                dynamic_augmentation=self.dynamic_augmentation,
                background_dir=self.background_dir,
                seed=self.seed,
            )
            if self.replay_ratio > 0:
                replay = FormulaManifestDataset(
                    self.hme_replay_manifest, self.hme_cache_root, dynamic_augmentation=False, seed=self.seed
                )
                self.train_dataset = CombinedDataset(university, replay)
                self.train_sampler = BalancedReplayBatchSampler(
                    len(university), len(replay), self.train_batch_size, self.replay_ratio, self.seed
                )
            else:
                self.train_dataset = university
                self.train_sampler = None
            self.university_val = FormulaManifestDataset(
                self.university_val_manifest, self.university_data_root, dynamic_augmentation=False
            )
            if self.enable_hme_validation:
                self.hme_val = FormulaManifestDataset(
                    self.hme_val_manifest, self.hme_cache_root, dynamic_augmentation=False
                )

    def train_dataloader(self):
        if self.train_sampler is None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_formula_samples,
                pin_memory=True,
                **self._worker_options(),
            )
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=collate_formula_samples,
            pin_memory=True,
            **self._worker_options(),
        )

    def val_dataloader(self):
        common = dict(
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_formula_samples,
            pin_memory=True,
            **self._worker_options(),
        )
        university_loader = DataLoader(self.university_val, **common)
        if not self.enable_hme_validation:
            return university_loader
        return [university_loader, DataLoader(self.hme_val, **common)]


class ManifestTestDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dictionary: str,
        manifest: str,
        data_root: Optional[str] = None,
        batch_size: int = 2,
        num_workers: int = 2,
    ) -> None:
        super().__init__()
        self.dictionary = dictionary
        self.manifest = manifest
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        vocab.init(dictionary)

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = FormulaManifestDataset(self.manifest, self.data_root, dynamic_augmentation=False)

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_formula_samples,
            pin_memory=True,
        )
