from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ttfm.utils import load_yaml


DEFAULT_RUNS = [
    ("default", Path("configs/default.yaml"), "5ep baseline"),
    ("conservative_weights", Path("configs/conservative.yaml"), "class-weight bias, 15 epochs"),
    ("resolution_safety_loss", Path("configs/third_run.yaml"), "720x448, false-safe penalty, threshold 0.58"),
    ("resolution_no_safety_loss", Path("configs/fourth_run.yaml"), "720x448, no false-safe penalty, threshold 0.53"),
    ("blue_green_best", Path("configs/blue_green_second_run.yaml"), "blue+green traversable labels, class-weight bias"),
]


def _load_metrics(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    output_root = Path(config["outputs_dir"]) / config["experiment_name"]
    metrics_path = output_root / "test_metrics.json"
    summary_path = output_root / "train_summary.json"
    metrics_payload = json.loads(metrics_path.read_text())
    summary_payload = json.loads(summary_path.read_text())
    metrics = metrics_payload["metrics"]
    counts = metrics.get("confusion_counts_traversable") or _counts_from_matrix(metrics["confusion_matrix"])
    return {
        "experiment": config["experiment_name"],
        "config": str(config_path),
        "input_size": config["training"]["input_size"],
        "epochs": config["training"]["epochs"],
        "augment": config["training"].get("augment", True),
        "class_weights": summary_payload.get("class_weights"),
        "ce_loss_weight": summary_payload.get("ce_loss_weight", config["training"].get("ce_loss_weight", 1.0)),
        "dice_loss_weight": summary_payload.get("dice_loss_weight", config["training"].get("dice_loss_weight", 1.0)),
        "false_safe_penalty_weight": summary_payload.get(
            "false_safe_penalty_weight", config["training"].get("false_safe_penalty_weight", 0.0)
        ),
        "threshold": metrics_payload.get("traversable_threshold", 0.5),
        "mIoU": metrics["mIoU"],
        "FSR": metrics["false_safe_rate"],
        "FBR": metrics["false_block_rate"],
        "confusion_counts_traversable": counts,
        "confusion_matrix": metrics["confusion_matrix"],
    }


def _counts_from_matrix(matrix: list[list[int]]) -> dict[str, int]:
    return {
        "tp": int(matrix[1][1]),
        "fp": int(matrix[0][1]),
        "tn": int(matrix[0][0]),
        "fn": int(matrix[1][0]),
    }


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def write_report(rows: list[dict[str, Any]], output_path: Path) -> None:
    best = max(rows, key=lambda item: item["mIoU"])
    threshold_sweep = _load_threshold_sweep(best)
    lines = [
        "# Supervised Baseline Ablation Report",
        "",
        "Definitions: FSR = FP / (FP + TN) for untraversable ground truth predicted traversable. "
        "FBR = FN / (FN + TP) for traversable ground truth predicted blocked. "
        "Confusion counts treat traversable as the positive class.",
        "",
        "| Run | Controlled change | mIoU | FSR | FBR | TP | FP | TN | FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        counts = row["confusion_counts_traversable"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    row["change"],
                    _format_float(row["mIoU"]),
                    _format_float(row["FSR"]),
                    _format_float(row["FBR"]),
                    str(counts["tp"]),
                    str(counts["fp"]),
                    str(counts["tn"]),
                    str(counts["fn"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            f"Best saved supervised run by test mIoU: `{best['name']}` at `{best['mIoU']:.4f}` mIoU, "
            f"FSR `{best['FSR']:.4f}`, FBR `{best['FBR']:.4f}`.",
        ]
    )
    if threshold_sweep is not None:
        sweep_best = threshold_sweep["best_mIoU"]
        lines.extend(
            [
                "",
                "## Threshold Sweep",
                "",
                f"For `{best['name']}`, the best test threshold in the saved sweep is `{sweep_best['threshold']:.2f}`: "
                f"mIoU `{sweep_best['mIoU']:.4f}`, FSR `{sweep_best['FSR']:.4f}`, FBR `{sweep_best['FBR']:.4f}`.",
                f"Full sweep: `{threshold_sweep['path']}`",
            ]
        )
    if len({row["augment"] for row in rows}) == 1:
        lines.extend(
            [
                "",
                "Augmentation setting was held constant in the saved full-data runs. The training loader now varies "
                "augmentation randomness by epoch for future controlled augmentation ablations.",
            ]
        )
    lines.extend(
        [
            "",
            "## Run Settings",
            "",
        ]
    )
    for row in rows:
        lines.append(
            f"- `{row['name']}` size=`{row['input_size']}` epochs=`{row['epochs']}` "
            f"augment=`{row['augment']}` CE=`{row['ce_loss_weight']}` Dice=`{row['dice_loss_weight']}` "
            f"false_safe_penalty=`{row['false_safe_penalty_weight']}` threshold=`{row['threshold']}` "
            f"class_weights=`{row['class_weights']}`"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _load_threshold_sweep(row: dict[str, Any]) -> dict[str, Any] | None:
    config = load_yaml(Path(row["config"]))
    sweep_path = Path(config["outputs_dir"]) / config["experiment_name"] / "test_threshold_sweep.json"
    if not sweep_path.exists():
        return None
    payload = json.loads(sweep_path.read_text())
    payload["path"] = str(sweep_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize saved supervised ablation runs.")
    parser.add_argument("--output", default="reports/supervised_ablation_report.md")
    args = parser.parse_args()

    rows = []
    for name, config_path, change in DEFAULT_RUNS:
        row = _load_metrics(config_path)
        row["name"] = name
        row["change"] = change
        rows.append(row)

    write_report(rows, Path(args.output))
    print(json.dumps({"output": args.output, "runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
