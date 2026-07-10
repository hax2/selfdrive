from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        processed_root: Path,
        split: str,
        image_size: tuple[int, int],
        max_samples: int | None = None,
        augment: bool = False,
        seed: int = 1337,
    ) -> None:
        self.processed_root = processed_root
        self.split = split
        self.width, self.height = image_size
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        manifest = json.loads((processed_root / "manifest.json").read_text())
        self.samples = [item for item in manifest if item["split"] == split]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + index + self.epoch * max(len(self.samples), 1))

    def _apply_train_augmentations(
        self,
        image: Image.Image,
        mask: Image.Image,
        index: int,
    ) -> tuple[Image.Image, Image.Image]:
        rng = self._rng(index)

        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        brightness = float(rng.uniform(0.9, 1.1))
        contrast = float(rng.uniform(0.9, 1.1))
        color = float(rng.uniform(0.9, 1.1))
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(color)
        if rng.random() < 0.25:
            image = ImageEnhance.Sharpness(image).enhance(float(rng.uniform(0.7, 1.3)))
        if rng.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 1.1))))

        return image, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        with Image.open(self.processed_root / sample["image_path"]) as image_file:
            image = image_file.convert("RGB")
        with Image.open(self.processed_root / sample["mask_path"]) as mask_file:
            mask = mask_file.copy()

        image = image.resize((self.width, self.height), resample=Image.Resampling.BILINEAR)
        mask = mask.resize((self.width, self.height), resample=Image.Resampling.NEAREST)
        if self.augment:
            image, mask = self._apply_train_augmentations(image, mask, index)

        image_array = np.array(image).astype(np.float32) / 255.0
        mask_array = np.array(mask).astype(np.int64)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(mask_array)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": sample["image_path"],
            "mask_path": sample["mask_path"],
        }


def compute_class_weights(processed_root: Path) -> torch.Tensor:
    manifest = json.loads((processed_root / "manifest.json").read_text())
    train_samples = [item for item in manifest if item["split"] == "train"]
    counts = np.zeros(2, dtype=np.int64)
    for sample in train_samples:
        mask = np.array(Image.open(processed_root / sample["mask_path"]))
        values, value_counts = np.unique(mask, return_counts=True)
        for value, count in zip(values.tolist(), value_counts.tolist()):
            if value in (0, 1):
                counts[value] += count
    counts = np.maximum(counts, 1)
    inv_freq = counts.sum() / counts
    weights = inv_freq / inv_freq.sum() * len(inv_freq)
    return torch.tensor(weights, dtype=torch.float32)
