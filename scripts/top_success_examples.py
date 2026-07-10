from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ttfm.data import SegmentationDataset
from ttfm.metrics import predict_from_logits
from ttfm.model import build_model
from ttfm.utils import ensure_dir, load_yaml, save_json
from ttfm.visualize import save_triptych


def per_image_stats(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    untrav = gt == 0
    trav = gt == 1
    false_safe_pixels = int(np.logical_and(untrav, pred == 1).sum())
    false_block_pixels = int(np.logical_and(trav, pred == 0).sum())
    untrav_pixels = int(untrav.sum())
    trav_pixels = int(trav.sum())
    false_safe_rate = false_safe_pixels / max(untrav_pixels, 1)
    false_block_rate = false_block_pixels / max(trav_pixels, 1)
    return {
        "false_safe_pixels": false_safe_pixels,
        "false_block_pixels": false_block_pixels,
        "untraversable_pixels": untrav_pixels,
        "traversable_pixels": trav_pixels,
        "false_safe_rate": false_safe_rate,
        "false_block_rate": false_block_rate,
        "combined_error": false_safe_rate + false_block_rate,
        "max_error": max(false_safe_rate, false_block_rate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Save the top successful review examples for a trained checkpoint.")
    parser.add_argument("--config", default="configs/fourth_run.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--checkpoint", default="best.pt")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    output_root = Path(config["outputs_dir"]) / config["experiment_name"]
    checkpoint_path = output_root / "checkpoints" / args.checkpoint
    success_root = output_root / f"{args.split}_top_success"
    if success_root.exists():
        shutil.rmtree(success_root)
    triptychs_dir = ensure_dir(success_root / "triptychs")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=2,
        model_name=config["training"].get("model_name", "pidnet-s"),
        model_config=config["training"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = SegmentationDataset(
        Path(config["processed_root"]),
        args.split,
        tuple(config["training"]["input_size"]),
        max_samples=config["training"].get(f"max_{args.split}_samples"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    traversable_threshold = float(config.get("postprocessing", {}).get("traversable_threshold", 0.5))

    results: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images)
            pred = predict_from_logits(logits, traversable_threshold=traversable_threshold)[0].cpu().numpy().astype(np.uint8)
            gt = batch["mask"][0].cpu().numpy().astype(np.uint8)
            image = (batch["image"][0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            stem = Path(batch["image_path"][0]).stem
            save_triptych(image, gt, pred, triptychs_dir / f"{stem}.png")
            stats = per_image_stats(pred, gt)
            results.append({"name": stem, "image_path": batch["image_path"][0], **stats})

    best = sorted(
        results,
        key=lambda item: (
            float(item["combined_error"]),
            float(item["max_error"]),
            int(item["false_safe_pixels"]) + int(item["false_block_pixels"]),
        ),
    )[: args.top_k]

    best_dir = ensure_dir(success_root / "best_triptychs")
    for item in best:
        shutil.copy2(triptychs_dir / f"{item['name']}.png", best_dir / f"{item['name']}.png")

    payload = {
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "traversable_threshold": traversable_threshold,
        "num_samples": len(results),
        "top_k": args.top_k,
        "best_examples": best,
    }
    save_json(success_root / "summary.json", payload)
    lines = [
        f"# Top Successful {args.split.title()} Examples",
        "",
        f"- Samples reviewed: `{len(results)}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Traversable threshold: `{traversable_threshold}`",
        "",
        "## Best Examples",
        "",
    ]
    for item in best:
        lines.append(
            f"- `{item['name']}` combined_error=`{float(item['combined_error']):.4f}` "
            f"false_safe_rate=`{float(item['false_safe_rate']):.4f}` "
            f"false_block_rate=`{float(item['false_block_rate']):.4f}`"
        )
    (success_root / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
