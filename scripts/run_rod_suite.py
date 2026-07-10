from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
GENERATED_CONFIGS = ROOT / "configs" / "generated"
SEEDS = (1337, 2027, 4242)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def output_root(config: dict[str, Any]) -> Path:
    return ROOT / config["outputs_dir"] / config["experiment_name"]


def run_cli(config_path: Path, command: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [sys.executable, "-m", "ttfm.cli", "--config", str(config_path.relative_to(ROOT)), command],
        cwd=ROOT,
        env=env,
        check=True,
    )


def run_experiment(config_path: Path) -> None:
    config = load_yaml(config_path)
    root = output_root(config)
    metrics_path = root / "test_metrics.json"
    if metrics_path.exists():
        print(f"SKIP completed: {config['experiment_name']}", flush=True)
        return
    print(f"RUN: {config['experiment_name']}", flush=True)
    run_cli(config_path, "train")
    run_cli(config_path, "eval")
    # Completed runs only need the validation-selected best checkpoint.
    (root / "checkpoints" / "last.pt").unlink(missing_ok=True)


def seeded_configs(base_path: Path, family: str, policy: str) -> list[Path]:
    base = load_yaml(base_path)
    paths = [base_path]
    for seed in SEEDS[1:]:
        config = json.loads(json.dumps(base))
        config["seed"] = seed
        config["experiment_name"] = f"{base['experiment_name']}_seed{seed}"
        path = GENERATED_CONFIGS / f"{family}_{policy}_seed{seed}.yaml"
        save_yaml(path, config)
        paths.append(path)
    return paths


def benchmark(name: str, config_path: Path, input_dir: str) -> Path:
    destination = REPORTS / f"rod_suite_benchmark_{name}.json"
    command = [
        sys.executable,
        "scripts/gpu_inference_benchmark.py",
        "--config",
        str(config_path.relative_to(ROOT)),
        "--input-dir",
        input_dir,
        "--device",
        "cuda",
        "--batch-size",
        "1",
        "--limit",
        "128",
        "--warmup",
        "5",
        "--repeats",
        "10",
        "--output",
        str(destination.relative_to(ROOT)),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    print(f"BENCHMARK: {name}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    return destination


def read_metrics(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    path = output_root(config) / "test_metrics.json"
    return {
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "config": str(config_path.relative_to(ROOT)),
        "metrics": json.loads(path.read_text())["metrics"],
    }


def seed_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("mIoU", "f1_traversable", "false_safe_rate", "false_block_rate")
    aggregate = {}
    for key in keys:
        values = [float(entry["metrics"][key]) for entry in entries]
        aggregate[key] = {
            "mean": statistics.mean(values),
            "std_population": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return {"runs": entries, "aggregate": aggregate}


def build_report(
    controlled: dict[str, list[Path]],
    paper_configs: dict[str, Path],
    pid_configs: dict[str, list[Path]],
    benchmarks: list[Path],
) -> tuple[Path, Path]:
    payload = {
        "controlled_rod_seeds": {
            policy: seed_summary([read_metrics(path) for path in paths]) for policy, paths in controlled.items()
        },
        "paper_recipe": {policy: read_metrics(path) for policy, path in paper_configs.items()},
        "controlled_pidnet_seeds": {
            policy: seed_summary([read_metrics(path) for path in paths]) for policy, paths in pid_configs.items()
        },
        "benchmarks": {path.stem: json.loads(path.read_text()) for path in benchmarks},
        "notes": [
            "Paper-recipe epochs are approximated at 15 because the paper does not disclose an epoch budget.",
            "Hardware benchmarks are forward-pass-only, batch 1, on the same server GPU.",
            "Controlled seed standard deviation is population standard deviation across seeds 1337, 2027, and 4242.",
        ],
    }
    json_path = REPORTS / "rod_suite_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = ["# ROD Experiment Suite", "", "## Controlled Seeds", ""]
    lines.append("| Model | Policy | mIoU mean +/- std | F1 mean +/- std | False-safe mean | False-block mean |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for model_name, group in (("ROD", payload["controlled_rod_seeds"]), ("PIDNet-S", payload["controlled_pidnet_seeds"])):
        for policy, summary in group.items():
            aggregate = summary["aggregate"]
            lines.append(
                f"| {model_name} | {policy} | {aggregate['mIoU']['mean']:.4f} +/- {aggregate['mIoU']['std_population']:.4f} "
                f"| {aggregate['f1_traversable']['mean']:.4f} +/- {aggregate['f1_traversable']['std_population']:.4f} "
                f"| {aggregate['false_safe_rate']['mean']:.4f} | {aggregate['false_block_rate']['mean']:.4f} |"
            )
    lines.extend(["", "## Paper Recipe", "", "| Policy | mIoU | F1 | False-safe | False-block |", "|---|---:|---:|---:|---:|"])
    for policy, entry in payload["paper_recipe"].items():
        metrics = entry["metrics"]
        lines.append(
            f"| {policy} | {metrics['mIoU']:.4f} | {metrics['f1_traversable']:.4f} "
            f"| {metrics['false_safe_rate']:.4f} | {metrics['false_block_rate']:.4f} |"
        )
    lines.extend(["", "## Same-GPU Batch-1 Benchmarks", "", "| Experiment | FPS | Mean latency (ms) | P95 latency (ms) |", "|---|---:|---:|---:|"])
    for name, result in payload["benchmarks"].items():
        lines.append(
            f"| {name} | {result['fps']:.2f} | {result['latency_ms_mean_per_image']:.2f} "
            f"| {result['latency_ms_p95_per_image']:.2f} |"
        )
    lines.extend(["", "## Notes", ""] + [f"- {note}" for note in payload["notes"]])
    markdown_path = REPORTS / "rod_suite_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    return json_path, markdown_path


def package_results(summary_paths: tuple[Path, Path], configs: list[Path], benchmarks: list[Path]) -> Path:
    archive_path = ROOT / "rod_suite_results.zip"
    candidates = [*summary_paths, *configs, *benchmarks]
    for config_path in configs:
        config = load_yaml(config_path)
        root = output_root(config)
        candidates.extend(root / name for name in ("test_metrics.json", "train_summary.json", "history.json"))
        if config["experiment_name"].endswith("_paper"):
            candidates.append(root / "checkpoints" / "best.pt")
    log_path = ROOT / "rod_suite.log"
    if log_path.exists():
        candidates.append(log_path)

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in dict.fromkeys(candidates):
            if path.exists():
                archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("RUN_COMMIT.txt", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True))
    archive_path.chmod(0o644)
    return archive_path


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    GENERATED_CONFIGS.mkdir(parents=True, exist_ok=True)
    required = [
        ROOT / "data_processed_blue_green" / "manifest.json",
        ROOT / "data_processed_blue" / "manifest.json",
        ROOT / "weights" / "efficient_sam_vits.pt",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing prepared inputs: {missing}. Run make prepare-rod and make prepare-rod-blue.")

    controlled = {
        "blue_green": seeded_configs(ROOT / "configs" / "rod_vits_cat.yaml", "rod", "blue_green"),
        "blue_only": seeded_configs(ROOT / "configs" / "rod_vits_cat_blue.yaml", "rod", "blue_only"),
    }
    paper_configs = {
        "blue_green": ROOT / "configs" / "rod_vits_cat_paper.yaml",
        "blue_only": ROOT / "configs" / "rod_vits_cat_blue_paper.yaml",
    }
    pid_configs = {
        "blue_green": seeded_configs(ROOT / "configs" / "pidnet_cat.yaml", "pidnet", "blue_green"),
        "blue_only": seeded_configs(ROOT / "configs" / "pidnet_cat_blue.yaml", "pidnet", "blue_only"),
    }
    all_configs = [path for paths in controlled.values() for path in paths]
    all_configs.extend(paper_configs.values())
    all_configs.extend(path for paths in pid_configs.values() for path in paths)

    for config_path in all_configs:
        run_experiment(config_path)

    benchmark_paths = [
        benchmark("pidnet_blue_green", pid_configs["blue_green"][0], "data_processed_blue_green/test/images"),
        benchmark("rod_blue_green", controlled["blue_green"][0], "data_processed_blue_green/test/images"),
        benchmark("pidnet_blue_only", pid_configs["blue_only"][0], "data_processed_blue/test/images"),
        benchmark("rod_blue_only", controlled["blue_only"][0], "data_processed_blue/test/images"),
    ]
    summaries = build_report(controlled, paper_configs, pid_configs, benchmark_paths)
    archive = package_results(summaries, all_configs, benchmark_paths)
    print(f"SUITE COMPLETE: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()
