#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1337, 2027, 4242)
BASE_CONFIGS = (
    Path("configs/smp_fpn_mobilenetv2.yaml"),
    Path("configs/smp_fpn_efficientnetb0.yaml"),
    Path("configs/smp_unet_mobilenetv2.yaml"),
    Path("configs/smp_unet_efficientnetb0.yaml"),
    Path("configs/segformer_b0_cat_blue.yaml"),
    Path("configs/ddrnet23_slim_cat_blue.yaml"),
    Path("configs/bisenetv2_cat_blue.yaml"),
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def generated_config(base_path: Path, seed: int, policy: str) -> tuple[Path, dict[str, Any]]:
    config = load_config(ROOT / base_path)
    if policy == "blue" and seed == 1337:
        return base_path, config

    if policy == "blue-green":
        config["processed_root"] = "data_processed_blue_green"
        config["preprocessing"]["traversable_palette_names"] = ["blue", "green"]
        config["preprocessing"]["mapping_filename"] = "discovered_mapping_blue_green.yaml"
        base_experiment = config["experiment_name"]
        if base_experiment.endswith("_blue"):
            base_experiment = base_experiment.removesuffix("_blue")
        config["experiment_name"] = f"{base_experiment}_blue_green"

    config["seed"] = seed
    if seed != 1337:
        config["experiment_name"] = f"{config['experiment_name']}_seed{seed}"
    generated_dir = "realtime" if policy == "blue" else "realtime_blue_green"
    output_path = Path("configs/generated") / generated_dir / f"{base_path.stem}_seed{seed}.yaml"
    absolute_output = ROOT / output_path
    absolute_output.parent.mkdir(parents=True, exist_ok=True)
    absolute_output.write_text(yaml.safe_dump(config, sort_keys=False))
    return output_path, config


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env={**os.environ, "PYTHONPATH": "src"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the three-seed real-time model comparison.")
    parser.add_argument("--no-benchmarks", action="store_true")
    parser.add_argument("--policy", choices=("blue", "blue-green"), default="blue")
    args = parser.parse_args()

    if args.policy == "blue":
        required = (ROOT / "data_processed_blue/summary.json", ROOT / "configs/discovered_mapping_blue.yaml")
    else:
        required = (
            ROOT / "data_processed_blue_green/summary.json",
            ROOT / "configs/discovered_mapping_blue_green.yaml",
        )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required blue-only data: {', '.join(missing)}")

    for base_path in BASE_CONFIGS:
        for seed in SEEDS:
            config_path, config = generated_config(base_path, seed, args.policy)
            experiment = config["experiment_name"]
            metrics_path = ROOT / config["outputs_dir"] / experiment / "test_metrics.json"
            review_path = ROOT / config["outputs_dir"] / experiment / "review_test" / "summary.json"
            if metrics_path.is_file() and review_path.is_file():
                print(f"Skipping completed experiment: {experiment}", flush=True)
                continue
            run(["bash", "scripts/run_train_eval_review.sh", str(config_path)])

        if args.no_benchmarks:
            continue
        base_config = load_config(ROOT / base_path)
        experiment = base_config["experiment_name"]
        run(
            [
                sys.executable,
                "scripts/gpu_inference_benchmark.py",
                "--config",
                str(base_path),
                "--input-dir",
                "data_processed_blue/test/images",
                "--device",
                "cuda",
                "--batch-size",
                "1",
                "--warmup",
                "5",
                "--repeats",
                "10",
                "--limit",
                "128",
                "--output",
                f"reports/benchmark_{experiment}.json",
            ]
        )

    print("Three-seed real-time comparison completed successfully.")


if __name__ == "__main__":
    main()
