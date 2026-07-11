from __future__ import annotations

import argparse
import platform
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from ttfm.metrics import predict_from_logits
from ttfm.model import build_model
from ttfm.utils import load_yaml


def image_paths(input_dir: Path, limit: int | None) -> list[Path]:
    paths = [path for path in sorted(input_dir.iterdir()) if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    return paths[:limit] if limit is not None else paths


def load_batch(paths: list[Path], width: int, height: int, device: torch.device) -> torch.Tensor:
    arrays = []
    for path in paths:
        with Image.open(path) as opened:
            image = opened.convert("RGB").resize((width, height), resample=Image.Resampling.BILINEAR)
        arrays.append(np.array(image).astype(np.float32).transpose(2, 0, 1) / 255.0)
    return torch.from_numpy(np.stack(arrays)).to(device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    index = min(int(round((len(values) - 1) * pct)), len(values) - 1)
    return sorted(values)[index]


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")
    return torch.device(requested_device)


def run_benchmark(
    config_path: Path,
    input_dir: Path,
    batch_size: int,
    warmup: int,
    repeats: int,
    limit: int | None,
    requested_device: str,
    threads: int | None,
    compile: bool = False,
) -> dict[str, Any]:
    if threads is not None:
        torch.set_num_threads(threads)

    config = load_yaml(config_path)
    checkpoint_path = Path(config["outputs_dir"]) / config["experiment_name"] / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    device = _resolve_device(requested_device)
    model = build_model(
        num_classes=2,
        model_name=config["training"].get("model_name", "pidnet-s"),
        model_config=config["training"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if compile:
        print("Compiling model via torch.compile...")
        model = torch.compile(model)

    width, height = config["training"]["input_size"]
    threshold = float(config.get("postprocessing", {}).get("traversable_threshold", 0.5))
    paths = image_paths(input_dir, limit)
    if not paths:
        raise ValueError(f"No images found in {input_dir}")

    batches = [paths[index : index + batch_size] for index in range(0, len(paths), batch_size)]
    tensors = [load_batch(batch, width, height, device) for batch in batches]

    with torch.no_grad():
        for _ in range(warmup):
            for tensor in tensors:
                logits = model(tensor)
                predict_from_logits(logits, traversable_threshold=threshold)
        synchronize(device)

        latencies = []
        total_images = 0
        start_total = perf_counter()
        for _ in range(repeats):
            for tensor in tensors:
                synchronize(device)
                start = perf_counter()
                logits = model(tensor)
                predict_from_logits(logits, traversable_threshold=threshold)
                synchronize(device)
                latencies.append(perf_counter() - start)
                total_images += tensor.shape[0]
        total_seconds = perf_counter() - start_total

    per_image_ms = [latency / tensors[index % len(tensors)].shape[0] * 1000.0 for index, latency in enumerate(latencies)]
    return {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "input_dir": str(input_dir),
        "device": str(device),
        "requested_device": requested_device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "images_per_repeat": len(paths),
        "total_images": total_images,
        "total_seconds": total_seconds,
        "fps": total_images / total_seconds if total_seconds > 0 else 0.0,
        "latency_ms_mean_per_image": float(np.mean(per_image_ms)),
        "latency_ms_p50_per_image": percentile(per_image_ms, 0.50),
        "latency_ms_p95_per_image": percentile(per_image_ms, 0.95),
        "compiled": compile,
        "note": "Forward pass only: no mask writing, overlay writing, or disk output. Use --device cpu --batch-size 1 for robot-style CPU latency.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model forward-pass inference without visualization I/O.")
    parser.add_argument("--config", default="configs/blue_green_second_run.yaml")
    parser.add_argument("--input-dir", default="data_processed_blue_green/test/images")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=None, help="Override torch CPU thread count, e.g. 4 for a small robot CPU.")
    parser.add_argument("--compile", action="store_true", help="Compile model via torch.compile")
    parser.add_argument("--output", default="reports/gpu_inference_benchmark.json")
    args = parser.parse_args()

    payload = run_benchmark(
        Path(args.config),
        Path(args.input_dir),
        args.batch_size,
        args.warmup,
        args.repeats,
        args.limit,
        args.device,
        args.threads,
        args.compile,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
