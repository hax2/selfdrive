#!/usr/bin/env python3
"""Keep a bounded number of existing convergence-suite workers runnable.

This controller does not launch training itself. It discovers direct worker
processes owned by ``run_convergence_suite.py`` launchers, pauses excess
workers with SIGSTOP, and resumes paused workers with SIGCONT when capacity is
available. Model, optimizer, and epoch state remain resident in each paused
process.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    state: str
    start_ticks: int
    command: str


@dataclass(frozen=True)
class Worker:
    process: ProcessInfo
    stage: str
    config: str

    @property
    def name(self) -> str:
        return Path(self.config).stem

    @property
    def paused(self) -> bool:
        return self.process.state in {"T", "t"}

    @property
    def finished(self) -> bool:
        return self.process.state in {"Z", "X", "x"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_process(pid: int) -> ProcessInfo | None:
    proc = Path("/proc") / str(pid)
    try:
        stat_text = (proc / "stat").read_text()
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        ).strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    closing_paren = stat_text.rfind(")")
    if closing_paren < 0:
        return None
    fields = stat_text[closing_paren + 2 :].split()
    if len(fields) < 20:
        return None
    return ProcessInfo(
        pid=pid,
        state=fields[0],
        ppid=int(fields[1]),
        start_ticks=int(fields[19]),
        command=command,
    )


def process_table() -> dict[int, ProcessInfo]:
    table: dict[int, ProcessInfo] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        process = read_process(int(entry.name))
        if process is not None:
            table[process.pid] = process
    return table


def command_argument(command: str, option: str) -> str | None:
    parts = command.split()
    try:
        index = parts.index(option)
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def discover_workers(
    table: dict[int, ProcessInfo], suite: str
) -> list[Worker]:
    launchers = {
        process.pid
        for process in table.values()
        if "run_convergence_suite.py" in process.command
    }
    workers: list[Worker] = []
    suite_fragment = f"configs/generated/{suite}/"
    for process in table.values():
        if process.ppid not in launchers:
            continue
        if "-m ttfm.cli" not in process.command:
            continue
        if suite_fragment not in process.command:
            continue
        config = command_argument(process.command, "--config")
        if config is None:
            continue
        stage = next(
            (candidate for candidate in ("train", "eval", "review") if candidate in process.command.split()),
            "unknown",
        )
        workers.append(Worker(process=process, stage=stage, config=config))
    return workers


def descendants(pid: int, table: dict[int, ProcessInfo]) -> list[int]:
    children_by_parent: dict[int, list[int]] = {}
    for process in table.values():
        children_by_parent.setdefault(process.ppid, []).append(process.pid)
    found: list[int] = []
    stack = list(children_by_parent.get(pid, []))
    while stack:
        child = stack.pop()
        found.append(child)
        stack.extend(children_by_parent.get(child, []))
    return found


def send_signal(pid: int, signum: signal.Signals, *, dry_run: bool) -> None:
    if dry_run:
        return
    try:
        os.kill(pid, signum)
    except ProcessLookupError:
        pass


def pause_worker(worker: Worker, table: dict[int, ProcessInfo], dry_run: bool) -> None:
    # Stop the producer first so it cannot create another data-loader child
    # while the tree is being frozen.
    send_signal(worker.process.pid, signal.SIGSTOP, dry_run=dry_run)
    for child in descendants(worker.process.pid, table):
        send_signal(child, signal.SIGSTOP, dry_run=dry_run)


def resume_worker(worker: Worker, table: dict[int, ProcessInfo], dry_run: bool) -> None:
    # Resume data-loader children before their consumer.
    for child in reversed(descendants(worker.process.pid, table)):
        send_signal(child, signal.SIGCONT, dry_run=dry_run)
    send_signal(worker.process.pid, signal.SIGCONT, dry_run=dry_run)


def reconcile(
    suite: str,
    max_active: int,
    *,
    dry_run: bool,
) -> tuple[list[Worker], list[str]]:
    table = process_table()
    workers = discover_workers(table, suite)
    active = [worker for worker in workers if not worker.paused and not worker.finished]
    paused = [worker for worker in workers if worker.paused]
    actions: list[str] = []

    if len(active) > max_active:
        excess = len(active) - max_active
        # Preserve older work and pause the newest training processes first.
        candidates = sorted(
            active,
            key=lambda worker: (
                worker.stage != "train",
                -worker.process.start_ticks,
            ),
        )
        for worker in candidates[:excess]:
            pause_worker(worker, table, dry_run)
            actions.append(f"pause pid={worker.process.pid} {worker.name} stage={worker.stage}")
    elif len(active) < max_active and paused:
        available = max_active - len(active)
        # Resume the oldest paused processes first.
        for worker in sorted(paused, key=lambda item: item.process.start_ticks)[:available]:
            resume_worker(worker, table, dry_run)
            actions.append(f"resume pid={worker.process.pid} {worker.name} stage={worker.stage}")

    return workers, actions


def write_status(path: Path, workers: list[Worker], actions: list[str], max_active: int) -> None:
    payload = {
        "updated_at": utc_now(),
        "controller_pid": os.getpid(),
        "max_active": max_active,
        "counts": {
            "active": sum(not worker.paused and not worker.finished for worker in workers),
            "paused": sum(worker.paused for worker in workers),
            "finished_waiting_for_launcher": sum(worker.finished for worker in workers),
        },
        "actions": actions,
        "workers": [
            {
                "pid": worker.process.pid,
                "state": worker.process.state,
                "stage": worker.stage,
                "name": worker.name,
            }
            for worker in sorted(workers, key=lambda item: item.process.start_ticks)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        default="convergence_blue_green_e300_c60_m60_p25",
        help="Generated configuration/suite directory name",
    )
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--status-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.max_active < 1:
        parser.error("--max-active must be at least 1")
    if args.poll_seconds <= 0 or args.status_seconds <= 0:
        parser.error("poll intervals must be positive")
    return args


def main() -> int:
    args = parse_args()
    status_path = ROOT / "outputs" / args.suite / "wave_controller_status.json"
    pid_path = ROOT / "outputs" / args.suite / "wave_controller.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    if pid_path.exists():
        try:
            previous_pid = int(pid_path.read_text().strip())
            os.kill(previous_pid, 0)
        except (ValueError, ProcessLookupError):
            pass
        else:
            raise SystemExit(f"Controller already appears to be running as PID {previous_pid}")
    pid_path.write_text(f"{os.getpid()}\n")

    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    last_status = 0.0
    last_counts: tuple[int, int] | None = None
    try:
        while not stopping:
            workers, actions = reconcile(
                args.suite,
                args.max_active,
                dry_run=args.dry_run,
            )
            active_count = sum(
                not worker.paused and not worker.finished for worker in workers
            )
            paused_count = sum(worker.paused for worker in workers)
            counts = (active_count, paused_count)
            now = time.monotonic()
            if actions or counts != last_counts or now - last_status >= args.status_seconds:
                write_status(status_path, workers, actions, args.max_active)
                print(
                    f"[{utc_now()}] active={active_count}/{args.max_active} "
                    f"paused={paused_count} discovered={len(workers)}",
                    flush=True,
                )
                for action in actions:
                    print(f"  {action}", flush=True)
                last_status = now
                last_counts = counts
            if args.once:
                break
            time.sleep(args.poll_seconds)
    finally:
        try:
            if int(pid_path.read_text().strip()) == os.getpid():
                pid_path.unlink(missing_ok=True)
        except (FileNotFoundError, ValueError):
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
