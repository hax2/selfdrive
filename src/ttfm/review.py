from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import SegmentationDataset
from .metrics import predict_from_logits
from .model import build_model
from .utils import ensure_dir, save_json
from .visualize import save_mask, save_overlay, save_triptych


def _per_image_rates(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    untrav = gt == 0
    trav = gt == 1
    false_safe_pixels = int(np.logical_and(untrav, pred == 1).sum())
    false_block_pixels = int(np.logical_and(trav, pred == 0).sum())
    untrav_pixels = int(untrav.sum())
    trav_pixels = int(trav.sum())
    return {
        "false_safe_pixels": false_safe_pixels,
        "false_block_pixels": false_block_pixels,
        "untraversable_pixels": untrav_pixels,
        "traversable_pixels": trav_pixels,
        "false_safe_rate": false_safe_pixels / max(untrav_pixels, 1),
        "false_block_rate": false_block_pixels / max(trav_pixels, 1),
    }


def run_review(config: dict[str, Any], split: str = "test", top_k: int = 10, checkpoint_name: str = "best.pt") -> dict[str, Any]:
    processed_root = Path(config["processed_root"])
    output_root = ensure_dir(Path(config["outputs_dir"]) / config["experiment_name"])
    review_root = output_root / f"{split}_review"
    if review_root.exists():
        shutil.rmtree(review_root)

    all_triptychs = ensure_dir(review_root / "all_triptychs")
    all_overlays = ensure_dir(review_root / "all_overlays")
    all_masks = ensure_dir(review_root / "all_masks")
    worst_safe_dir = ensure_dir(review_root / "worst_false_safe")
    worst_block_dir = ensure_dir(review_root / "worst_false_block")

    checkpoint = torch.load(output_root / "checkpoints" / checkpoint_name, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=2,
        model_name=config["training"].get("model_name", "pidnet-s"),
        model_config=config["training"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = SegmentationDataset(
        processed_root,
        split,
        tuple(config["training"]["input_size"]),
        max_samples=config["training"].get(f"max_{split}_samples"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    traversable_threshold = float(config.get("postprocessing", {}).get("traversable_threshold", 0.5))

    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images)
            pred = predict_from_logits(logits, traversable_threshold=traversable_threshold)[0].cpu().numpy().astype(np.uint8)
            gt = batch["mask"][0].cpu().numpy().astype(np.uint8)
            image = (batch["image"][0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            stem = Path(batch["image_path"][0]).stem

            save_triptych(image, gt, pred, all_triptychs / f"{stem}.png")
            save_overlay(image, pred, all_overlays / f"{stem}.png")
            save_mask(pred, all_masks / f"{stem}.png")

            rates = _per_image_rates(pred, gt)
            results.append(
                {
                    "name": stem,
                    "image_path": batch["image_path"][0],
                    "mask_path": batch["mask_path"][0],
                    **rates,
                }
            )

    worst_false_safe = sorted(results, key=lambda item: item["false_safe_rate"], reverse=True)[:top_k]
    worst_false_block = sorted(results, key=lambda item: item["false_block_rate"], reverse=True)[:top_k]

    for group, directory in [(worst_false_safe, worst_safe_dir), (worst_false_block, worst_block_dir)]:
        for item in group:
            stem = item["name"]
            shutil.copy2(all_triptychs / f"{stem}.png", directory / f"{stem}.png")

    payload = {
        "split": split,
        "checkpoint": str(output_root / "checkpoints" / checkpoint_name),
        "review_root": str(review_root),
        "num_samples": len(results),
        "top_k": top_k,
        "traversable_threshold": traversable_threshold,
        "worst_false_safe": worst_false_safe,
        "worst_false_block": worst_false_block,
    }
    save_json(review_root / "summary.json", payload)

    lines = [
        f"# {split.title()} Review",
        "",
        f"- Samples reviewed: `{len(results)}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        "",
        "## Worst False-Safe",
        "",
    ]
    for item in worst_false_safe:
        lines.append(
            f"- `{item['name']}` false_safe_rate=`{item['false_safe_rate']:.4f}` false_safe_pixels=`{item['false_safe_pixels']}`"
        )
    lines.extend(["", "## Worst False-Block", ""])
    for item in worst_false_block:
        lines.append(
            f"- `{item['name']}` false_block_rate=`{item['false_block_rate']:.4f}` false_block_pixels=`{item['false_block_pixels']}`"
        )
    (review_root / "summary.md").write_text("\n".join(lines) + "\n")
    return payload
