from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ttfm.data import SegmentationDataset
from ttfm.metrics import update_confusion, metrics_from_confusion
from ttfm.model import build_model
from ttfm.utils import load_yaml


def run_threshold_sweep(
    config_path: Path,
    split: str,
    thresholds: list[float],
    checkpoint_name: str = "best.pt",
) -> dict[str, Any]:
    config = load_yaml(config_path)
    processed_root = Path(config["processed_root"])
    output_root = Path(config["outputs_dir"]) / config["experiment_name"]
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

    dataset = SegmentationDataset(
        processed_root,
        split,
        tuple(config["training"]["input_size"]),
        max_samples=config["training"].get(f"max_{split}_samples"),
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    confusions = {threshold: torch.zeros((2, 2), dtype=torch.int64) for threshold in thresholds}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"]
            logits = model(images).cpu()
            for threshold, confusion in confusions.items():
                update_confusion(confusion, logits, masks, traversable_threshold=threshold)

    rows = []
    for threshold in thresholds:
        metrics = metrics_from_confusion(confusions[threshold])
        rows.append(
            {
                "threshold": threshold,
                "mIoU": metrics["mIoU"],
                "FSR": metrics["false_safe_rate"],
                "FBR": metrics["false_block_rate"],
                "confusion_counts_traversable": metrics["confusion_counts_traversable"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )

    best_miou = max(rows, key=lambda item: item["mIoU"])
    lowest_fsr = min(rows, key=lambda item: item["FSR"])
    payload = {
        "config": str(config_path),
        "experiment": config["experiment_name"],
        "split": split,
        "checkpoint": str(checkpoint_path),
        "rows": rows,
        "best_mIoU": best_miou,
        "lowest_FSR": lowest_fsr,
    }
    output_path = output_root / f"{split}_threshold_sweep.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(payload, output_root / f"{split}_threshold_sweep.md")
    return payload


def write_markdown(payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        f"# {payload['split'].title()} Threshold Sweep",
        "",
        f"- Experiment: `{payload['experiment']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        "- Definitions: FSR = FP/(FP+TN), FBR = FN/(FN+TP), traversable is positive.",
        "",
        "| Threshold | mIoU | FSR | FBR | TP | FP | TN | FN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        counts = row["confusion_counts_traversable"]
        lines.append(
            f"| {row['threshold']:.2f} | {row['mIoU']:.4f} | {row['FSR']:.4f} | {row['FBR']:.4f} | "
            f"{counts['tp']} | {counts['fp']} | {counts['tn']} | {counts['fn']} |"
        )
    best = payload["best_mIoU"]
    lines.extend(
        [
            "",
            f"Best mIoU in sweep: threshold `{best['threshold']:.2f}`, mIoU `{best['mIoU']:.4f}`, "
            f"FSR `{best['FSR']:.4f}`, FBR `{best['FBR']:.4f}`.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def _parse_thresholds(value: str) -> list[float]:
    if ":" in value:
        start, stop, step = [float(part) for part in value.split(":")]
        thresholds = []
        current = start
        while current <= stop + 1e-9:
            thresholds.append(round(current, 4))
            current += step
        return thresholds
    return [float(part) for part in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep traversable probability thresholds for a trained checkpoint.")
    parser.add_argument("--config", default="configs/blue_green_second_run.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--thresholds", default="0.35:0.65:0.05")
    args = parser.parse_args()

    payload = run_threshold_sweep(Path(args.config), args.split, _parse_thresholds(args.thresholds))
    print(json.dumps({"experiment": payload["experiment"], "rows": len(payload["rows"]), "best_mIoU": payload["best_mIoU"]}, indent=2))


if __name__ == "__main__":
    main()
