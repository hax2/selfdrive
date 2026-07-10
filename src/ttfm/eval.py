from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import SegmentationDataset
from .metrics import metrics_from_confusion, predict_from_logits, update_confusion
from .model import build_model
from .utils import ensure_dir, save_json
from .visualize import save_triptych


def run_evaluation(config: dict[str, Any], split: str = "test", checkpoint_name: str = "best.pt") -> dict[str, Any]:
    processed_root = Path(config["processed_root"])
    output_root = ensure_dir(Path(config["outputs_dir"]) / config["experiment_name"])
    checkpoint_path = output_root / "checkpoints" / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=2,
        model_name=config["training"].get("model_name", "pidnet-s"),
        model_config=config["training"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    max_key = f"max_{split}_samples"
    dataset = SegmentationDataset(
        processed_root,
        split,
        tuple(config["training"]["input_size"]),
        max_samples=config["training"].get(max_key),
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    ce_loss = nn.CrossEntropyLoss(weight=checkpoint["class_weights"].to(device))
    running_loss = 0.0
    examples_written = 0
    visuals_dir = ensure_dir(output_root / f"{split}_visuals")
    traversable_threshold = float(config.get("postprocessing", {}).get("traversable_threshold", 0.5))

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            running_loss += ce_loss(logits, masks).item() * images.shape[0]
            confusion = update_confusion(confusion, logits.cpu(), masks.cpu(), traversable_threshold=traversable_threshold)
            preds = predict_from_logits(logits.cpu(), traversable_threshold=traversable_threshold)
            for index in range(min(2, images.shape[0])):
                if examples_written >= 8:
                    break
                save_triptych(
                    (images[index].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8"),
                    masks[index].cpu().numpy().astype("uint8"),
                    preds[index].numpy().astype("uint8"),
                    visuals_dir / f"{Path(batch['image_path'][index]).stem}.png",
                )
                examples_written += 1

    metrics = metrics_from_confusion(confusion)
    payload = {
        "split": split,
        "loss": running_loss / max(len(dataset), 1),
        "metrics": metrics,
        "checkpoint": str(checkpoint_path),
        "traversable_threshold": traversable_threshold,
    }
    save_json(output_root / f"{split}_metrics.json", payload)
    return payload
