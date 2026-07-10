from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .utils import ensure_dir


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask == 0] = np.array([215, 48, 39], dtype=np.uint8)
    out[mask == 1] = np.array([49, 163, 84], dtype=np.uint8)
    return out


def blend(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = mask_to_color(mask)
    return np.clip(image * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def save_triptych(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, destination: Path) -> None:
    ensure_dir(destination.parent)
    canvas = np.concatenate([image, mask_to_color(gt), mask_to_color(pred)], axis=1)
    Image.fromarray(canvas).save(destination)


def save_mask(mask: np.ndarray, destination: Path) -> None:
    ensure_dir(destination.parent)
    Image.fromarray(mask_to_color(mask)).save(destination)


def save_overlay(image: np.ndarray, mask: np.ndarray, destination: Path) -> None:
    ensure_dir(destination.parent)
    Image.fromarray(blend(image, mask)).save(destination)
