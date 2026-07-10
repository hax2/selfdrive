from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ttfm.data import SegmentationDataset
from ttfm.metrics import metrics_from_confusion, predict_from_logits, update_confusion
from ttfm.model import build_model
from ttfm.utils import load_yaml


def percentile(values: list[float], fraction: float) -> float:
    return float(np.percentile(np.asarray(values), fraction * 100))


def latency_summary(seconds: list[float]) -> dict[str, float]:
    milliseconds = np.asarray(seconds) * 1000.0
    mean_ms = float(milliseconds.mean())
    return {
        "samples": len(seconds),
        "mean_ms": mean_ms,
        "p50_ms": percentile(seconds, 0.50) * 1000.0,
        "p95_ms": percentile(seconds, 0.95) * 1000.0,
        "p99_ms": percentile(seconds, 0.99) * 1000.0,
        "fps_from_mean_latency": 1000.0 / mean_ms,
    }


def cpu_tensor_from_rgb(array: np.ndarray, width: int, height: int, dtype: torch.dtype) -> torch.Tensor:
    image = Image.fromarray(array).resize((width, height), resample=Image.Resampling.BILINEAR)
    resized = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).div_(255.0).unsqueeze(0)
    return tensor.to(dtype=dtype)


def synchronize() -> None:
    torch.cuda.synchronize()


def load_model(config: dict[str, Any], checkpoint_path: Path, precision: str) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(
        num_classes=2,
        model_name=config["training"].get("model_name", "pidnet-s"),
        model_config=config["training"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if precision == "fp16":
        model.half()
    return model.cuda()


def evaluate_accuracy(
    model: torch.nn.Module,
    config: dict[str, Any],
    precision: str,
    threshold: float,
    max_samples: int | None,
) -> dict[str, Any]:
    dataset = SegmentationDataset(
        Path(config["processed_root"]),
        "test",
        tuple(config["training"]["input_size"]),
        max_samples=max_samples,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    dtype = torch.float16 if precision == "fp16" else torch.float32
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device="cuda", dtype=dtype)
            logits = model(images)
            confusion = update_confusion(confusion, logits.float().cpu(), batch["mask"], threshold)
    return metrics_from_confusion(confusion)


def benchmark(
    config_path: Path,
    checkpoint_path: Path,
    input_dir: Path,
    precision: str,
    limit: int,
    warmup: int,
    repeats: int,
    accuracy_limit: int | None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the edge deployment benchmark")
    torch.backends.cudnn.benchmark = True
    config = load_yaml(config_path)
    threshold = float(config.get("postprocessing", {}).get("traversable_threshold", 0.5))
    width, height = config["training"]["input_size"]
    paths = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}][:limit]
    if not paths:
        raise ValueError(f"No images found in {input_dir}")
    native_frames = []
    for path in paths:
        with Image.open(path) as opened:
            native_frames.append(np.asarray(opened.convert("RGB")).copy())

    dtype = torch.float16 if precision == "fp16" else torch.float32
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = load_model(config, checkpoint_path, precision)
    synchronize()
    allocated_after_load = torch.cuda.memory_allocated()
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    first_cpu = cpu_tensor_from_rgb(native_frames[0], width, height, dtype)
    with torch.inference_mode():
        for _ in range(warmup):
            first_gpu = first_cpu.cuda()
            prediction = predict_from_logits(model(first_gpu).float(), threshold)
            prediction.cpu()
        synchronize()

        forward_latencies = []
        for _ in range(repeats):
            for frame in native_frames:
                cpu_tensor = cpu_tensor_from_rgb(frame, width, height, dtype)
                gpu_tensor = cpu_tensor.cuda()
                synchronize()
                start = perf_counter()
                prediction = predict_from_logits(model(gpu_tensor).float(), threshold)
                synchronize()
                forward_latencies.append(perf_counter() - start)

        pipeline_latencies = []
        for _ in range(repeats):
            for frame in native_frames:
                synchronize()
                start = perf_counter()
                cpu_tensor = cpu_tensor_from_rgb(frame, width, height, dtype)
                gpu_tensor = cpu_tensor.cuda()
                prediction = predict_from_logits(model(gpu_tensor).float(), threshold)
                prediction.cpu()
                synchronize()
                pipeline_latencies.append(perf_counter() - start)

    peak_allocated = torch.cuda.max_memory_allocated()
    accuracy = evaluate_accuracy(model, config, precision, threshold, accuracy_limit)
    synchronize()

    return {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "input_dir": str(input_dir),
        "model_name": config["training"].get("model_name", "pidnet-s"),
        "precision": precision,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "batch_size": 1,
        "input_size": [width, height],
        "internal_encoder_size": config["training"].get("rod_encoder_image_size"),
        "warmup": warmup,
        "repeats": repeats,
        "unique_frames": len(native_frames),
        "timing_scope": {
            "forward": "GPU forward, softmax, and threshold; excludes H2D and mask D2H",
            "pipeline": "decoded host RGB resize/normalise, H2D, forward, threshold, and mask D2H; excludes disk/camera/planner",
        },
        "forward": latency_summary(forward_latencies),
        "pipeline": latency_summary(pipeline_latencies),
        "parameter_memory_mib": parameter_bytes / 1024**2,
        "cuda_allocated_after_load_mib": allocated_after_load / 1024**2,
        "cuda_peak_allocated_mib": peak_allocated / 1024**2,
        "fp16_test_metrics": accuracy,
        "accuracy_samples": accuracy_limit or 544,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-1 FP16 edge-style segmentation benchmark")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-dir", default="data_processed_blue_green/test/images", type=Path)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--accuracy-limit", type=int, default=0, help="Zero evaluates the complete 544-image test split")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = benchmark(
        args.config,
        args.checkpoint,
        args.input_dir,
        args.precision,
        args.limit,
        args.warmup,
        args.repeats,
        args.accuracy_limit or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
