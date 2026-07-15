#!/usr/bin/env python3
"""Run and summarize the matched ORFD ROD/FPN comparison."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


MODELS = {
    "ROD ViT-S": Path("configs/orfd_rod_vits.yaml"),
    "FPN/EfficientNet-B0": Path("configs/orfd_fpn_efficientnetb0.yaml"),
}


def _run(command: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env={**os.environ, "PYTHONPATH": "src"})


def _generated_config(base_path: Path, model_label: str, seed: int) -> tuple[Path, dict[str, Any]]:
    config = yaml.safe_load(base_path.read_text())
    slug = "rod_vits" if model_label.startswith("ROD") else "fpn_efficientnetb0"
    config["seed"] = seed
    config["experiment_name"] = f"orfd_{slug}_seed{seed}"
    path = Path("configs/generated/orfd") / f"{slug}_seed{seed}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path, config


def _mean_std(values: list[float]) -> str:
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} +/- {std:.4f}"


def _write_report(records: list[dict[str, Any]], benchmarks: dict[str, dict[str, Any]]) -> None:
    report_dir = Path("reports/orfd")
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset_summary = json.loads(Path("data_processed_orfd/summary.json").read_text())
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["model"], []).append(record)

    summary: dict[str, Any] = {
        "protocol": {
            "dataset": "ORFD training/validation/testing partition",
            "dataset_profile": dataset_summary["dataset_profile"],
            "split_counts": dataset_summary["split_counts"],
            "comparable_to_published_full_test": dataset_summary["comparable_to_published_full_test"],
            "positive_class": "traversable",
            "input_size": [640, 384],
            "early_stopping": "validation mIoU, min_epochs=15, patience=8, min_delta=0.0005",
            "maximum_epochs": 60,
            "comparison_type": "controlled recipe, not an exact reproduction of the paper's undisclosed epoch budget",
        },
        "runs": records,
        "benchmarks": benchmarks,
        "aggregates": {},
    }
    lines = [
        "# ORFD ROD vs FPN Comparison",
        "",
        f"Dataset profile: `{dataset_summary['dataset_profile']}`; split counts: {dataset_summary['split_counts']}.",
        "Traversable is the positive class. Both models use the same 640 x 384 input,",
        "AdamW/CE recipe, batch size, validation-based stopping rule, and evaluation implementation.",
        "",
        "| Model | Seeds | mIoU | Traversable IoU | F1 | FSR | FBR | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, model_records in grouped.items():
        metrics = [record["metrics"] for record in model_records]
        aggregate = {
            key: {
                "mean": statistics.fmean(float(item[key]) for item in metrics),
                "population_std": statistics.pstdev(float(item[key]) for item in metrics) if len(metrics) > 1 else 0.0,
            }
            for key in ["mIoU", "iou_traversable", "f1_traversable", "false_safe_rate", "false_block_rate"]
        }
        summary["aggregates"][model] = aggregate
        fps = benchmarks.get(model, {}).get("fps")
        lines.append(
            f"| {model} | {len(model_records)} | {_mean_std([m['mIoU'] for m in metrics])} | "
            f"{_mean_std([m['iou_traversable'] for m in metrics])} | "
            f"{_mean_std([m['f1_traversable'] for m in metrics])} | "
            f"{_mean_std([m['false_safe_rate'] for m in metrics])} | "
            f"{_mean_std([m['false_block_rate'] for m in metrics])} | "
            f"{fps:.2f}" + " |" if fps is not None else "-- |"
        )
    lines.extend(
        [
            "",
            "FPS is synchronized CUDA batch-1 forward-pass throughput over 128 test images; it excludes disk I/O and visualization.",
            "The ROD paper does not disclose its epoch budget, so this is a controlled cross-model experiment rather than an exact reproduction.",
            "The Academic Torrents profile is also missing samples from the published full split, so its absolute score must not be compared directly with the paper's full-test score.",
            "",
        ]
    )
    (report_dir / "comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (report_dir / "comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337, 2027, 4242])
    args = parser.parse_args()
    records: list[dict[str, Any]] = []
    benchmarks: dict[str, dict[str, Any]] = {}

    for model_label, base_config in MODELS.items():
        for seed_index, seed in enumerate(args.seeds):
            config_path, config = _generated_config(base_config, model_label, seed)
            output_root = Path(config["outputs_dir"]) / config["experiment_name"]
            metrics_path = output_root / "test_metrics.json"
            if not metrics_path.is_file():
                _run([sys.executable, "-m", "ttfm.cli", "--config", str(config_path), "train"])
                _run([sys.executable, "-m", "ttfm.cli", "--config", str(config_path), "eval", "--split", "test"])
            else:
                print(f"SKIP completed: {config['experiment_name']}")
            payload = json.loads(metrics_path.read_text())
            records.append(
                {
                    "model": model_label,
                    "seed": seed,
                    "config": str(config_path),
                    "epochs_completed": json.loads((output_root / "train_summary.json").read_text())["epochs_completed"],
                    "metrics": payload["metrics"],
                }
            )

            if seed_index == 0:
                benchmark_path = Path("reports/orfd") / f"benchmark_{config['experiment_name']}.json"
                if not benchmark_path.is_file():
                    _run(
                        [
                            sys.executable,
                            "scripts/gpu_inference_benchmark.py",
                            "--config",
                            str(config_path),
                            "--input-dir",
                            "data_processed_orfd/test/images",
                            "--device",
                            "cuda",
                            "--batch-size",
                            "1",
                            "--warmup",
                            "10",
                            "--repeats",
                            "10",
                            "--limit",
                            "128",
                            "--output",
                            str(benchmark_path),
                        ]
                    )
                benchmarks[model_label] = json.loads(benchmark_path.read_text())

    _write_report(records, benchmarks)


if __name__ == "__main__":
    main()
