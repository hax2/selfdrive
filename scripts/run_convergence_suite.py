#!/usr/bin/env python3
"""Generate and run the convergence-aware CaT architecture matrix.

The launcher is intentionally experiment-granular: completed training and
evaluation stages are detected from their saved artifacts, so relaunching the
same command skips work that already finished. Each active training process
writes ``training_progress.json`` in its experiment directory; this launcher
combines those records into a suite-level status and rough ETA range.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1337, 2027, 4242)
MODEL_CONFIGS = (
    ("pidnet_s", Path("configs/pidnet_cat_blue.yaml")),
    ("rod_vits", Path("configs/rod_vits_cat_blue.yaml")),
    ("fpn_mobilenetv2", Path("configs/smp_fpn_mobilenetv2.yaml")),
    ("fpn_efficientnet_b0", Path("configs/smp_fpn_efficientnetb0.yaml")),
    ("unet_mobilenetv2", Path("configs/smp_unet_mobilenetv2.yaml")),
    ("unet_efficientnet_b0", Path("configs/smp_unet_efficientnetb0.yaml")),
    ("segformer_b0", Path("configs/segformer_b0_cat_blue.yaml")),
    ("ddrnet23_slim", Path("configs/ddrnet23_slim_cat_blue.yaml")),
    ("bisenetv2", Path("configs/bisenetv2_cat_blue.yaml")),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(int(round(seconds)), 0)
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def policy_values(policy: str) -> tuple[str, list[str], str, str]:
    if policy == "blue-green":
        return (
            "data_processed_blue_green",
            ["blue", "green"],
            "discovered_mapping_blue_green.yaml",
            "blue_green",
        )
    return "data_processed_blue", ["blue"], "discovered_mapping_blue.yaml", "blue"


@dataclass
class Task:
    model: str
    policy: str
    seed: int
    experiment: str
    config_path: Path
    output_root: Path
    process: subprocess.Popen[str] | None = None
    stage: str | None = None
    log_handle: Any | None = None
    failed: bool = False
    failure: str | None = None

    @property
    def train_summary(self) -> Path:
        return self.output_root / "train_summary.json"

    @property
    def best_checkpoint(self) -> Path:
        return self.output_root / "checkpoints" / "best.pt"

    @property
    def test_metrics(self) -> Path:
        return self.output_root / "test_metrics.json"

    @property
    def review_summary(self) -> Path:
        return self.output_root / "test_review" / "summary.json"

    @property
    def progress(self) -> Path:
        return self.output_root / "training_progress.json"


def tuning_token(args: argparse.Namespace) -> str:
    token = (
        f"e{args.max_epochs}_c{args.cosine_decay_epochs}"
        f"_m{args.min_epochs}_p{args.patience}"
    )
    if not math.isclose(args.min_delta, 0.0001):
        delta = f"{args.min_delta:g}".replace("-", "m").replace(".", "p")
        token += f"_d{delta}"
    if args.disable_augmentation:
        token += "_noaug"
    return token


def suite_id(args: argparse.Namespace) -> str:
    policy = args.policy.replace("-", "_")
    return f"convergence_{policy}_{tuning_token(args)}"


def selected_models(raw: str) -> list[tuple[str, Path]]:
    if raw == "all":
        return list(MODEL_CONFIGS)
    requested = {item.strip() for item in raw.split(",") if item.strip()}
    available = {name for name, _ in MODEL_CONFIGS}
    unknown = requested - available
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(sorted(unknown))}. Available: {', '.join(sorted(available))}")
    return [(name, path) for name, path in MODEL_CONFIGS if name in requested]


def selected_policies(policy: str) -> list[str]:
    return ["blue-green", "blue"] if policy == "both" else [policy]


def selected_seeds(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one seed is required")
    return values


def generate_tasks(args: argparse.Namespace, generated_root: Path) -> list[Task]:
    tasks: list[Task] = []
    for policy in selected_policies(args.policy):
        processed_root, palette, mapping, policy_token = policy_values(policy)
        for model, base_path in selected_models(args.models):
            base = yaml.safe_load((ROOT / base_path).read_text())
            for seed in selected_seeds(args.seeds):
                config = copy.deepcopy(base)
                experiment = (
                    f"convergence_{policy_token}_{model}_seed{seed}"
                    f"_{tuning_token(args)}"
                )
                config["seed"] = seed
                config["processed_root"] = processed_root
                config["experiment_name"] = experiment
                config["preprocessing"]["traversable_palette_names"] = palette
                config["preprocessing"]["mapping_filename"] = mapping
                training = config["training"]
                training["epochs"] = args.max_epochs
                training["lr_scheduler"] = "cosine"
                training["cosine_decay_epochs"] = args.cosine_decay_epochs
                training["save_last_checkpoint"] = False
                training["save_optimizer_state"] = False
                training["early_stopping"] = {
                    "enabled": True,
                    "metric": "mIoU",
                    "mode": "max",
                    "patience": args.patience,
                    "min_epochs": args.min_epochs,
                    "min_delta": args.min_delta,
                }
                if args.disable_augmentation:
                    training["augment"] = False
                relative_config = generated_root / policy_token / f"{model}_seed{seed}.yaml"
                absolute_config = ROOT / relative_config
                absolute_config.parent.mkdir(parents=True, exist_ok=True)
                absolute_config.write_text(yaml.safe_dump(config, sort_keys=False))
                tasks.append(
                    Task(
                        model=model,
                        policy=policy,
                        seed=seed,
                        experiment=experiment,
                        config_path=relative_config,
                        output_root=ROOT / config["outputs_dir"] / experiment,
                    )
                )
    return tasks


def validate_inputs(tasks: list[Task]) -> None:
    required = {
        ROOT / "data_processed_blue" / "summary.json" if any(task.policy == "blue" for task in tasks) else None,
        ROOT / "configs" / "discovered_mapping_blue.yaml"
        if any(task.policy == "blue" for task in tasks)
        else None,
        ROOT / "data_processed_blue_green" / "summary.json"
        if any(task.policy == "blue-green" for task in tasks)
        else None,
        ROOT / "configs" / "discovered_mapping_blue_green.yaml"
        if any(task.policy == "blue-green" for task in tasks)
        else None,
    }
    missing = [str(path.relative_to(ROOT)) for path in required if path is not None and not path.is_file()]
    if any(task.model == "rod_vits" for task in tasks):
        rod_weight = ROOT / "weights" / "efficient_sam_vits.pt"
        if not rod_weight.is_file():
            missing.append(str(rod_weight.relative_to(ROOT)))
    if missing:
        raise FileNotFoundError("Missing required prepared inputs: " + ", ".join(sorted(missing)))


def next_stage(task: Task, with_review: bool) -> str | None:
    if not task.train_summary.is_file() or not task.best_checkpoint.is_file():
        return "train"
    if not task.test_metrics.is_file():
        return "eval"
    if with_review and not task.review_summary.is_file():
        return "review"
    return None


def stage_command(task: Task, stage: str, top_k: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ttfm.cli",
        "--config",
        str(task.config_path),
        stage,
    ]
    if stage == "review":
        command.extend(["--split", "test", "--top-k", str(top_k)])
    return command


def launch_stage(task: Task, stage: str, logs_root: Path, top_k: int) -> None:
    logs_root.mkdir(parents=True, exist_ok=True)
    log_path = logs_root / f"{task.experiment}.log"
    task.log_handle = log_path.open("a", buffering=1)
    task.log_handle.write(f"\n[{utc_now()}] START stage={stage}\n")
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src{os.pathsep}{existing_pythonpath}"
    task.stage = stage
    task.process = subprocess.Popen(
        stage_command(task, stage, top_k),
        cwd=ROOT,
        env=env,
        stdout=task.log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"[{utc_now()}] launched {task.experiment} stage={stage} log={log_path.relative_to(ROOT)}", flush=True)


def result_rows(tasks: list[Task]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        metrics_payload = read_json(task.test_metrics)
        if metrics_payload is None:
            continue
        summary = read_json(task.train_summary) or {}
        metrics = metrics_payload.get("metrics", {})
        rows.append(
            {
                "model": task.model,
                "policy": task.policy,
                "seed": task.seed,
                "experiment": task.experiment,
                "epochs_completed": summary.get("epochs_completed"),
                "best_epoch": summary.get("best_epoch"),
                "best_val_mIoU": summary.get("best_val_mIoU"),
                "training_seconds": summary.get("total_training_seconds"),
                "test_mIoU": metrics.get("mIoU"),
                "test_F1": metrics.get("f1_traversable"),
                "test_FSR": metrics.get("false_safe_rate"),
                "test_FBR": metrics.get("false_block_rate"),
            }
        )
    return rows


def write_results(tasks: list[Task], suite_root: Path) -> None:
    rows = result_rows(tasks)
    suite_root.mkdir(parents=True, exist_ok=True)
    (suite_root / "results.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    fieldnames = [
        "model",
        "policy",
        "seed",
        "experiment",
        "epochs_completed",
        "best_epoch",
        "best_val_mIoU",
        "training_seconds",
        "test_mIoU",
        "test_F1",
        "test_FSR",
        "test_FBR",
    ]
    with (suite_root / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["policy"]), []).append(row)
    metric_fields = ("test_mIoU", "test_F1", "test_FSR", "test_FBR")
    summary_rows: list[dict[str, Any]] = []
    for (model, policy), group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "model": model,
            "policy": policy,
            "completed_seeds": len(group),
        }
        for metric in metric_fields:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None
            ]
            summary[f"{metric}_mean"] = statistics.fmean(values) if values else None
            summary[f"{metric}_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            ) if values else None
        summary_rows.append(summary)

    (suite_root / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True)
    )
    summary_fields = [
        "model",
        "policy",
        "completed_seeds",
        *[
            field
            for metric in metric_fields
            for field in (f"{metric}_mean", f"{metric}_std")
        ],
    ]
    with (suite_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def task_status(task: Task, with_review: bool) -> dict[str, Any]:
    progress = read_json(task.progress) or {}
    stage = task.stage or next_stage(task, with_review)
    if task.process is not None:
        status = "running"
    elif task.failed:
        status = "failed"
    elif stage is None:
        status = "completed"
    else:
        status = "pending"
    return {
        "model": task.model,
        "policy": task.policy,
        "seed": task.seed,
        "experiment": task.experiment,
        "status": status,
        "stage": task.stage if task.process is not None else stage,
        "failure": task.failure,
        "progress": progress,
    }


def estimate_eta(tasks: list[Task], args: argparse.Namespace) -> dict[str, Any]:
    seconds_per_epoch: list[float] = []
    for task in tasks:
        progress = read_json(task.progress) or {}
        rolling = progress.get("rolling_epoch_seconds") or progress.get("average_epoch_seconds")
        if isinstance(rolling, (int, float)) and rolling > 0:
            seconds_per_epoch.append(float(rolling))
            continue
        summary = read_json(task.train_summary) or {}
        average = summary.get("average_epoch_seconds")
        if isinstance(average, (int, float)) and average > 0:
            seconds_per_epoch.append(float(average))
    if not seconds_per_epoch:
        return {
            "basis": "waiting for the first completed epoch",
            "median_seconds_per_epoch": None,
            "early_stop_planning_seconds": None,
            "maximum_ceiling_seconds": None,
        }

    median_epoch = statistics.median(seconds_per_epoch)
    expected_epoch_work = 0.0
    maximum_epoch_work = 0.0
    for task in tasks:
        if task.failed:
            continue
        if next_stage(task, args.with_review) is None:
            continue
        progress = read_json(task.progress) or {}
        current = int(progress.get("current_epoch", 0) or 0)
        bad_epochs = int(progress.get("epochs_without_improvement", 0) or 0)
        if task.process is not None and task.stage == "train":
            maximum_remaining = max(args.max_epochs - current, 0)
            expected_remaining = min(
                max(args.min_epochs - current, args.patience - bad_epochs, 0),
                maximum_remaining,
            )
        elif task.train_summary.is_file():
            expected_remaining = 0
            maximum_remaining = 0
        else:
            expected_remaining = min(args.max_epochs, args.min_epochs + args.patience)
            maximum_remaining = args.max_epochs
        expected_epoch_work += expected_remaining
        maximum_epoch_work += maximum_remaining

    divisor = max(args.jobs, 1)
    return {
        "basis": "median observed epoch time; evaluation/review and GPU contention are not modelled",
        "median_seconds_per_epoch": median_epoch,
        "early_stop_planning_seconds": median_epoch * expected_epoch_work / divisor,
        "maximum_ceiling_seconds": median_epoch * maximum_epoch_work / divisor,
    }


def write_status(tasks: list[Task], suite_root: Path, args: argparse.Namespace, started_at: str) -> dict[str, Any]:
    statuses = [task_status(task, args.with_review) for task in tasks]
    counts = {
        name: sum(item["status"] == name for item in statuses)
        for name in ("completed", "running", "pending", "failed")
    }
    eta = estimate_eta(tasks, args)
    payload = {
        "suite": suite_root.name,
        "started_at": started_at,
        "updated_at": utc_now(),
        "configuration": {
            "policy": args.policy,
            "models": args.models,
            "seeds": selected_seeds(args.seeds),
            "jobs": args.jobs,
            "max_epochs": args.max_epochs,
            "cosine_decay_epochs": args.cosine_decay_epochs,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "disable_augmentation": args.disable_augmentation,
            "with_review": args.with_review,
        },
        "counts": counts,
        "eta": eta,
        "tasks": statuses,
    }
    suite_root.mkdir(parents=True, exist_ok=True)
    (suite_root / "status.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_results(tasks, suite_root)
    print(
        f"[{payload['updated_at']}] suite={suite_root.name} "
        f"complete={counts['completed']}/{len(tasks)} running={counts['running']} "
        f"pending={counts['pending']} failed={counts['failed']} "
        f"eta_plan={format_duration(eta['early_stop_planning_seconds'])} "
        f"eta_cap={format_duration(eta['maximum_ceiling_seconds'])}",
        flush=True,
    )
    for item in statuses:
        if item["status"] != "running":
            continue
        progress = item["progress"]
        print(
            f"  {item['experiment']} stage={item['stage']} "
            f"epoch={progress.get('current_epoch', '?')}/{progress.get('maximum_epochs', '?')} "
            f"best={progress.get('best_val_mIoU', '?')}@{progress.get('best_epoch', '?')} "
            f"bad={progress.get('epochs_without_improvement', '?')} "
            f"job_eta_stop={format_duration(progress.get('eta_to_early_stop_seconds'))}",
            flush=True,
        )
    return payload


def print_saved_status(suite_root: Path) -> int:
    payload = read_json(suite_root / "status.json")
    if payload is None:
        print(f"No status found at {suite_root / 'status.json'}", file=sys.stderr)
        return 1
    counts = payload["counts"]
    eta = payload["eta"]
    print(
        json.dumps(
            {
                "suite": payload["suite"],
                "updated_at": payload["updated_at"],
                "counts": counts,
                "eta": eta,
            },
            indent=2,
        )
    )
    print("\nRunning tasks:")
    for task in payload["tasks"]:
        if task["status"] == "running":
            progress = task.get("progress", {})
            print(
                f"- {task['experiment']} {task['stage']} "
                f"epoch {progress.get('current_epoch', '?')}/{progress.get('maximum_epochs', '?')} "
                f"best {progress.get('best_val_mIoU', '?')}@{progress.get('best_epoch', '?')}"
            )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=("blue-green", "blue", "both"),
        default="blue-green",
        help="Label policy to run; defaults to the thesis's principal blue+green policy",
    )
    parser.add_argument("--models", default="all", help="Comma-separated model slugs, or 'all'")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--jobs", type=int, default=2, help="Maximum concurrent GPU processes")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--cosine-decay-epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--with-review", action="store_true", help="Generate full review images after evaluation")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.max_epochs < 1 or args.cosine_decay_epochs < 1 or args.min_epochs < 1:
        parser.error("epoch values must be at least 1")
    if args.min_epochs > args.max_epochs:
        parser.error("--min-epochs cannot exceed --max-epochs")
    if args.patience < 1:
        parser.error("--patience must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    identifier = suite_id(args)
    suite_root = ROOT / "outputs" / identifier
    generated_root = Path("configs/generated") / identifier
    if args.status_only:
        return print_saved_status(suite_root)

    tasks = generate_tasks(args, generated_root)
    if not args.prepare_only:
        validate_inputs(tasks)
    suite_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "suite": identifier,
        "created_at": utc_now(),
        "command": sys.argv,
        "tasks": [
            {
                "model": task.model,
                "policy": task.policy,
                "seed": task.seed,
                "experiment": task.experiment,
                "config": str(task.config_path),
            }
            for task in tasks
        ],
    }
    (suite_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Prepared {len(tasks)} isolated configurations under {generated_root}", flush=True)
    started_at = utc_now()
    write_status(tasks, suite_root, args, started_at)
    if args.prepare_only:
        print(f"Manifest: {suite_root / 'manifest.json'}", flush=True)
        return 0

    logs_root = suite_root / "logs"
    failed_tasks: list[Task] = []
    last_status = 0.0
    try:
        while True:
            for task in tasks:
                if task.process is None:
                    continue
                return_code = task.process.poll()
                if return_code is None:
                    continue
                if task.log_handle is not None:
                    task.log_handle.write(f"[{utc_now()}] END stage={task.stage} return_code={return_code}\n")
                    task.log_handle.close()
                    task.log_handle = None
                finished_stage = task.stage
                task.process = None
                task.stage = None
                if return_code != 0:
                    task.failed = True
                    task.failure = f"stage {finished_stage} exited with code {return_code}"
                    failed_tasks.append(task)
                    print(f"[{utc_now()}] FAILED {task.experiment}: {task.failure}", flush=True)
                else:
                    print(f"[{utc_now()}] finished {task.experiment} stage={finished_stage}", flush=True)
                    if finished_stage == "eval":
                        metrics_payload = read_json(task.test_metrics) or {}
                        metrics = metrics_payload.get("metrics", {})
                        print(
                            f"  result mIoU={metrics.get('mIoU', 'n/a')} "
                            f"F1={metrics.get('f1_traversable', 'n/a')} "
                            f"FSR={metrics.get('false_safe_rate', 'n/a')} "
                            f"FBR={metrics.get('false_block_rate', 'n/a')}",
                            flush=True,
                        )

            active = [task for task in tasks if task.process is not None]
            available_slots = args.jobs - len(active)
            if available_slots > 0:
                for task in tasks:
                    if available_slots <= 0:
                        break
                    if task.process is not None or task.failed:
                        continue
                    stage = next_stage(task, args.with_review)
                    if stage is None:
                        continue
                    # Avoid two memory-heavy ROD training jobs at once. A ROD
                    # job may still share the GPU with one lightweight model.
                    if stage == "train" and task.model == "rod_vits" and any(
                        running.model == "rod_vits" and running.stage == "train" for running in active
                    ):
                        continue
                    launch_stage(task, stage, logs_root, args.top_k)
                    active.append(task)
                    available_slots -= 1

            now = time.monotonic()
            if now - last_status >= args.poll_seconds:
                write_status(tasks, suite_root, args, started_at)
                last_status = now

            unfinished = [
                task
                for task in tasks
                if task.process is not None or (not task.failed and next_stage(task, args.with_review) is not None)
            ]
            if not unfinished:
                break
            time.sleep(min(args.poll_seconds, 5.0))
    except KeyboardInterrupt:
        print("Interrupt received; terminating active child processes...", flush=True)
        for task in tasks:
            if task.process is not None:
                task.process.terminate()
        return 130
    finally:
        write_status(tasks, suite_root, args, started_at)

    if failed_tasks:
        print(
            f"Suite finished with {len(failed_tasks)} failed task(s). "
            "Rerun the same command after inspecting logs.",
            flush=True,
        )
        return 1
    print(
        f"Suite complete. Results: {suite_root / 'results.csv'}; "
        f"model summary: {suite_root / 'summary.csv'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
