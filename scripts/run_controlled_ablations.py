from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ABLATIONS = {
    "baseline": {
        "description": "Blue+green baseline; all controlled variants change one setting relative to this.",
        "training": {},
    },
    "loss_false_safe": {
        "description": "Loss ablation: add false-safe penalty only.",
        "training": {"false_safe_penalty_weight": 0.20},
    },
    "augment_off": {
        "description": "Augmentation ablation: disable training augmentations only.",
        "training": {"augment": False},
    },
    "resolution_720x448": {
        "description": "Resolution ablation: increase input resolution only.",
        "training": {"input_size": [720, 448]},
    },
    "class_weights_neutral": {
        "description": "Class-weight ablation: use neutral class weights only.",
        "training": {"class_weights": [1.0, 1.0]},
    },
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def build_config(
    base_config: dict[str, Any],
    run_name: str,
    run_spec: dict[str, Any],
    experiment_prefix: str,
    epochs: int | None,
    batch_size: int | None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["experiment_name"] = f"{experiment_prefix}_{run_name}"
    training = config.setdefault("training", {})
    training.update(run_spec["training"])
    if epochs is not None:
        training["epochs"] = epochs
    if batch_size is not None:
        training["batch_size"] = batch_size
    config.setdefault("postprocessing", {})["traversable_threshold"] = 0.5
    return config


def run_command(command: list[str], dry_run: bool, env: dict[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True, env=env)


def summarize(config_paths: list[Path], output_path: Path) -> None:
    rows = []
    for config_path in config_paths:
        config = load_yaml(config_path)
        metrics_path = Path(config["outputs_dir"]) / config["experiment_name"] / "test_metrics.json"
        if not metrics_path.exists():
            rows.append({"experiment": config["experiment_name"], "status": "missing_metrics"})
            continue
        payload = json.loads(metrics_path.read_text())
        metrics = payload["metrics"]
        rows.append(
            {
                "experiment": config["experiment_name"],
                "config": str(config_path),
                "mIoU": metrics["mIoU"],
                "FSR": metrics["false_safe_rate"],
                "FBR": metrics["false_block_rate"],
                "confusion_counts_traversable": metrics.get("confusion_counts_traversable"),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"runs": rows}, indent=2, sort_keys=True) + "\n")

    markdown_path = output_path.with_suffix(".md")
    lines = [
        "# Controlled Supervised Ablations",
        "",
        "All variants are generated from the same blue+green base config. Each variant changes one setting relative to the baseline unless explicitly noted.",
        "",
        "| Run | mIoU | FSR | FBR | TP | FP | TN | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("status"):
            lines.append(f"| `{row['experiment']}` | missing | missing | missing | - | - | - | - |")
            continue
        counts = row["confusion_counts_traversable"] or {}
        lines.append(
            f"| `{row['experiment']}` | {row['mIoU']:.4f} | {row['FSR']:.4f} | {row['FBR']:.4f} | "
            f"{counts.get('tp', '-')} | {counts.get('fp', '-')} | {counts.get('tn', '-')} | {counts.get('fn', '-')} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and run controlled supervised ablations.")
    parser.add_argument("--base-config", default="configs/blue_green_second_run.yaml")
    parser.add_argument("--config-dir", default="configs/ablations")
    parser.add_argument("--experiment-prefix", default="controlled_ablation")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all generated runs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for all generated runs.")
    parser.add_argument("--only", nargs="*", choices=sorted(ABLATIONS), default=list(ABLATIONS), help="Subset of ablations to run.")
    parser.add_argument("--generate-only", action="store_true", help="Write configs but do not train/evaluate.")
    parser.add_argument("--skip-train", action="store_true", help="Run eval/report only for existing checkpoints.")
    parser.add_argument("--summary", default="reports/controlled_ablation_summary.json")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    base_config = load_yaml(Path(args.base_config))
    config_paths = []
    for run_name in args.only:
        config = build_config(base_config, run_name, ABLATIONS[run_name], args.experiment_prefix, args.epochs, args.batch_size)
        config_path = Path(args.config_dir) / f"{args.experiment_prefix}_{run_name}.yaml"
        save_yaml(config_path, config)
        config_paths.append(config_path)
        print(f"Wrote {config_path}: {ABLATIONS[run_name]['description']}")

    if args.generate_only:
        return

    for config_path in config_paths:
        if not args.skip_train:
            run_command([sys.executable, "-m", "ttfm.cli", "--config", str(config_path), "train"], dry_run=False, env=env)
        run_command([sys.executable, "-m", "ttfm.cli", "--config", str(config_path), "eval", "--split", "test"], dry_run=False, env=env)

    summarize(config_paths, Path(args.summary))
    print(f"Wrote {args.summary} and {Path(args.summary).with_suffix('.md')}")


if __name__ == "__main__":
    main()
