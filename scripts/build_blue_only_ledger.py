#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1337, 2027, 4242)
PATH_ALIASES: list[tuple[Path, str]] = [(ROOT, ".")]

NEW_MODELS = (
    ("FPN/MobileNetV2", "mixed_binary_traversability_smp_fpn_mobilenetv2", "smp_fpn_mobilenetv2.yaml", 4_215_554),
    ("FPN/EfficientNet-B0", "mixed_binary_traversability_smp_fpn_efficientnetb0", "smp_fpn_efficientnetb0.yaml", 5_759_614),
    ("U-Net/MobileNetV2", "mixed_binary_traversability_smp_unet_mobilenetv2", "smp_unet_mobilenetv2.yaml", 6_629_090),
    ("U-Net/EfficientNet-B0", "mixed_binary_traversability_smp_unet_efficientnetb0", "smp_unet_efficientnetb0.yaml", 6_251_614),
    ("SegFormer-B0", "mixed_binary_traversability_segformer_b0_blue", "segformer_b0_cat_blue.yaml", 3_714_658),
    ("DDRNet-23-Slim", "mixed_binary_traversability_ddrnet23_slim_blue", "ddrnet23_slim_cat_blue.yaml", 5_694_882),
    ("BiSeNetV2", "mixed_binary_traversability_bisenetv2_blue", "bisenetv2_cat_blue.yaml", 3_341_202),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def display_path(path: Path) -> str:
    resolved = path.resolve()
    for root, label in PATH_ALIASES:
        try:
            relative = resolved.relative_to(root.resolve())
            return str(Path(label) / relative)
        except ValueError:
            continue
    return str(path)


def metric_integrity(metrics: dict[str, Any]) -> bool:
    counts = metrics["confusion_counts_traversable"]
    tn, fp, fn, tp = (counts[key] for key in ("tn", "fp", "fn", "tp"))
    expected = {
        "false_safe_rate": fp / (fp + tn),
        "false_block_rate": fn / (fn + tp),
        "precision_traversable": tp / (tp + fp),
        "recall_traversable": tp / (tp + fn),
        "f1_traversable": 2 * tp / (2 * tp + fp + fn),
        "iou_traversable": tp / (tp + fp + fn),
        "iou_untraversable": tn / (tn + fp + fn),
    }
    expected["mIoU"] = (expected["iou_traversable"] + expected["iou_untraversable"]) / 2
    return all(math.isclose(metrics[key], value, rel_tol=0, abs_tol=1e-12) for key, value in expected.items())


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    normalized.pop("seed", None)
    normalized.pop("experiment_name", None)
    normalized.pop("config_path", None)
    return normalized


def benchmark_fields(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"available": False, "path": None, "sha256": None, "fps": None, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    data = load_json(path)
    return {
        "available": True,
        "path": display_path(path),
        "sha256": sha256(path),
        "fps": data["fps"],
        "mean_ms": data["latency_ms_mean_per_image"],
        "p50_ms": data["latency_ms_p50_per_image"],
        "p95_ms": data["latency_ms_p95_per_image"],
    }


def make_run(
    model: str,
    seed: int,
    config_path: Path,
    metrics_path: Path,
    checkpoint_path: Path | None,
    recorded_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    payload = load_json(metrics_path)
    metrics = payload["metrics"]
    if int(config["seed"]) != seed:
        raise ValueError(f"Seed mismatch in {config_path}: expected {seed}, got {config['seed']}")
    if not metric_integrity(metrics):
        raise ValueError(f"Confusion-derived metric mismatch in {metrics_path}")
    checkpoint_exists = checkpoint_path is not None and checkpoint_path.is_file()
    return {
        "model": model,
        "seed": seed,
        "experiment_name": config["experiment_name"],
        "config_path": display_path(config_path),
        "config_sha256": sha256(config_path),
        "metrics_path": display_path(metrics_path),
        "metrics_sha256": sha256(metrics_path),
        "checkpoint_path": display_path(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_available": checkpoint_exists,
        "checkpoint_sha256": sha256(checkpoint_path) if checkpoint_exists else recorded_checkpoint_sha256,
        "checkpoint_hash_source": "local_file" if checkpoint_exists else ("server_manifest" if recorded_checkpoint_sha256 else None),
        "metrics_integrity": True,
        "metrics": metrics,
        "normalized_config": normalized_config(config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the verified blue-only results ledger.")
    parser.add_argument(
        "--realtime-results",
        type=Path,
        default=Path("/home/Shodan/WCS/realtime-comparison-results/realtime_comparison_results"),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    realtime = args.realtime_results.resolve()
    rod_suite = ROOT / "rod_suite_results"
    PATH_ALIASES.insert(0, (realtime, "realtime_comparison_results"))

    runs: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    legacy_hash_manifest = ROOT / "reports/blue_only_legacy_checkpoint_sha256.txt"
    legacy_hashes = {
        path: digest
        for digest, path in (line.split(maxsplit=1) for line in legacy_hash_manifest.read_text().splitlines() if line.strip())
    }

    for model, stem, base_config_name, parameters in NEW_MODELS:
        model_runs = []
        for seed in SEEDS:
            suffix = "" if seed == 1337 else f"_seed{seed}"
            config_path = (
                realtime / "configs" / base_config_name
                if seed == 1337
                else realtime / "configs/generated/realtime" / f"{Path(base_config_name).stem}_seed{seed}.yaml"
            )
            output_root = realtime / "outputs" / f"{stem}{suffix}"
            model_runs.append(make_run(model, seed, config_path, output_root / "test_metrics.json", output_root / "checkpoints/best.pt"))
        runs.extend(model_runs)
        benchmark_path = realtime / "reports" / f"benchmark_{stem}.json"
        metadata[model] = {
            "parameters": parameters,
            "trainable_parameters": parameters,
            "h100_eager": benchmark_fields(benchmark_path),
            "ryzen_eager": benchmark_fields(ROOT / "reports" / f"cpu_{stem}_ryzen5500_idle.json"),
            "ryzen_compiled": benchmark_fields(ROOT / "reports" / f"cpu_{stem}_ryzen5500_idle_compiled.json"),
        }

    legacy_specs = (
        (
            "PIDNet-S",
            "mixed_binary_traversability_pidnet_blue_hardware",
            "pidnet_cat_blue.yaml",
            "pidnet_blue_only_seed",
            7_623_522,
            7_623_522,
            rod_suite / "reports/rod_suite_benchmark_pidnet_blue_only.json",
            ROOT / "reports/cpu_pidnet_s_ryzen5500_torch213_idle.json",
            ROOT / "reports/cpu_pidnet_s_ryzen5500_torch213_idle_compiled.json",
        ),
        (
            "ROD ViT-S",
            "mixed_binary_traversability_rod_vits_blue",
            "rod_vits_cat_blue.yaml",
            "rod_blue_only_seed",
            29_106_050,
            6_752_386,
            rod_suite / "reports/rod_suite_benchmark_rod_blue_only.json",
            None,
            None,
        ),
    )
    for model, stem, base_config_name, generated_prefix, parameters, trainable, h100, cpu, compiled in legacy_specs:
        model_runs = []
        for seed in SEEDS:
            suffix = "" if seed == 1337 else f"_seed{seed}"
            config_path = (
                rod_suite / "configs" / base_config_name
                if seed == 1337
                else rod_suite / "configs/generated" / f"{generated_prefix}{seed}.yaml"
            )
            output_root = rod_suite / "outputs" / f"{stem}{suffix}"
            server_checkpoint_path = f"outputs/{stem}{suffix}/checkpoints/best.pt"
            model_runs.append(
                make_run(
                    model,
                    seed,
                    config_path,
                    output_root / "test_metrics.json",
                    output_root / "checkpoints/best.pt",
                    legacy_hashes.get(server_checkpoint_path),
                )
            )
        runs.extend(model_runs)
        metadata[model] = {
            "parameters": parameters,
            "trainable_parameters": trainable,
            "h100_eager": benchmark_fields(h100),
            "ryzen_eager": benchmark_fields(cpu),
            "ryzen_compiled": benchmark_fields(compiled),
        }

    aggregates = []
    for model in metadata:
        model_runs = [run for run in runs if run["model"] == model]
        if [run["seed"] for run in model_runs] != list(SEEDS):
            raise ValueError(f"Incomplete seeds for {model}")
        controlled = all(
            run["normalized_config"] == model_runs[0]["normalized_config"] for run in model_runs[1:]
        )
        aggregate: dict[str, Any] = {
            "model": model,
            "seeds": list(SEEDS),
            "seed_count": len(model_runs),
            "controlled_seed_configs": controlled,
            "all_metrics_integrity_checks_passed": all(run["metrics_integrity"] for run in model_runs),
            "all_checkpoints_available": all(run["checkpoint_available"] for run in model_runs),
            "all_checkpoint_hashes_recorded": all(run["checkpoint_sha256"] is not None for run in model_runs),
            **metadata[model],
        }
        for key in ("mIoU", "f1_traversable", "false_safe_rate", "false_block_rate"):
            values = [run["metrics"][key] for run in model_runs]
            aggregate[key] = {"mean": statistics.mean(values), "std_population": statistics.pstdev(values), "values": values}
        aggregates.append(aggregate)
    aggregates.sort(key=lambda item: item["mIoU"]["mean"], reverse=True)

    for run in runs:
        run.pop("normalized_config")

    ledger = {
        "policy": "blue_only",
        "positive_class": "CaT blue/sedan traversable; black, green, and red untraversable",
        "seeds": list(SEEDS),
        "test_images": 544,
        "realtime_suite_commit": (realtime / "REALTIME_SUITE_COMMIT.txt").read_text().strip(),
        "generator_script": "./scripts/build_blue_only_ledger.py",
        "generator_script_sha256": sha256(Path(__file__)),
        "audit": {
            "expected_models": 9,
            "expected_runs": 27,
            "actual_models": len(aggregates),
            "actual_runs": len(runs),
            "all_metric_integrity_checks_passed": all(run["metrics_integrity"] for run in runs),
            "all_seed_configs_controlled": all(item["controlled_seed_configs"] for item in aggregates),
            "missing_checkpoint_hashes": [
                {"model": run["model"], "seed": run["seed"], "expected_path": run["checkpoint_path"]}
                for run in runs
                if run["checkpoint_sha256"] is None
            ],
        },
        "aggregates": aggregates,
        "runs": runs,
    }
    if len(runs) != 27 or len(aggregates) != 9:
        raise ValueError("Expected 27 runs across 9 models")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "blue_only_verified_ledger.json"
    csv_path = args.output_dir / "blue_only_verified_summary.csv"
    md_path = args.output_dir / "blue_only_verified_summary.md"
    json_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["model", "mIoU_mean", "mIoU_std", "F1_mean", "FSR_mean", "FBR_mean", "H100_FPS", "Ryzen_eager_FPS", "Ryzen_compiled_FPS", "parameters"])
        for item in aggregates:
            writer.writerow([
                item["model"], item["mIoU"]["mean"], item["mIoU"]["std_population"],
                item["f1_traversable"]["mean"], item["false_safe_rate"]["mean"],
                item["false_block_rate"]["mean"], item["h100_eager"]["fps"],
                item["ryzen_eager"]["fps"], item["ryzen_compiled"]["fps"], item["parameters"],
            ])

    lines = [
        "# Verified Blue-Only Results Ledger",
        "",
        "All accuracy and safety values are population mean +/- population standard deviation over seeds 1337, 2027, and 4242.",
        "H100 and Ryzen throughput use batch 1 and 640 x 384 input. Throughput is forward-pass-only.",
        "",
        "| Model | mIoU | F1 | FSR | FBR | H100 FPS | Ryzen eager | Ryzen compiled | Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        def pm(key: str) -> str:
            return f"{item[key]['mean']:.4f} +/- {item[key]['std_population']:.4f}"
        def fps(key: str) -> str:
            value = item[key]["fps"]
            return f"{value:.2f}" if value is not None else "--"
        lines.append(
            f"| {item['model']} | {pm('mIoU')} | {pm('f1_traversable')} | {pm('false_safe_rate')} | "
            f"{pm('false_block_rate')} | {fps('h100_eager')} | {fps('ryzen_eager')} | "
            f"{fps('ryzen_compiled')} | {item['parameters']:,} |"
        )
    missing = ledger["audit"]["missing_checkpoint_hashes"]
    lines.extend([
        "",
        "## Audit status",
        "",
        f"- Runs present: {len(runs)}/27",
        f"- Metric integrity checks: {'PASS' if ledger['audit']['all_metric_integrity_checks_passed'] else 'FAIL'}",
        f"- Seed-only configuration checks: {'PASS' if ledger['audit']['all_seed_configs_controlled'] else 'FAIL'}",
        f"- Checkpoint hashes present: {len(runs) - len(missing)}/27",
        "- The six controlled PIDNet-S and ROD checkpoints were omitted from the earlier download bundle; their SHA-256 values were recovered directly from the retained server files and recorded in `blue_only_legacy_checkpoint_sha256.txt`.",
        "",
        f"Detailed per-run paths and SHA-256 values: `{json_path.name}`.",
    ])
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(ledger["audit"], indent=2))
    print(md_path)


if __name__ == "__main__":
    main()
