#!/usr/bin/env python3
"""TensorRT edge-deployment benchmark.

Exports a model to ONNX, builds a TensorRT FP16 engine, and benchmarks
forward-only and full-pipeline latencies.  Produces the same JSON schema
as edge_deployment_benchmark.py with extra TRT-specific fields.

Requirements:
    pip install --extra-index-url https://pypi.nvidia.com/ \
        tensorrt-cu13-bindings tensorrt-cu13-libs onnx
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import onnx
import tensorrt as trt
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ttfm.data import SegmentationDataset
from ttfm.metrics import metrics_from_confusion, update_confusion
from ttfm.model import build_model
from ttfm.utils import load_yaml

# ───────────────────────── helpers ──────────────────────────────────────────

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


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


def cpu_tensor_from_rgb(
    array: np.ndarray, width: int, height: int, dtype: np.dtype
) -> np.ndarray:
    """Resize & normalise an RGB uint8 array → CHW float array."""
    image = Image.fromarray(array).resize(
        (width, height), resample=Image.Resampling.BILINEAR
    )
    resized = np.asarray(image, dtype=np.float32) / 255.0
    return resized.transpose(2, 0, 1)[np.newaxis].astype(dtype)  # 1×3×H×W


def synchronize() -> None:
    torch.cuda.synchronize()


# ───────────────────── ONNX export ─────────────────────────────────────────

def export_onnx(
    config: dict[str, Any],
    checkpoint_path: Path,
    onnx_path: Path,
    width: int,
    height: int,
) -> None:
    """Export a PyTorch model checkpoint to ONNX."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(
        num_classes=2,
        model_name=config["training"].get("model_name", "pidnet-s"),
        model_config=config["training"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dummy = torch.randn(1, 3, height, width)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,  # Legacy exporter: single self-contained file.
    )
    # Validate
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX exported and validated: {onnx_path}")


# ───────────────── TensorRT engine build ───────────────────────────────────

def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool,
    width: int,
    height: int,
) -> trt.ICudaEngine:
    """Build (or load cached) a TensorRT engine from an ONNX model."""
    if engine_path.exists():
        print(f"  Loading cached TRT engine: {engine_path}")
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())

    print("  Building TRT engine (this may take 30-120 s)…")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Set fixed optimisation profile for batch=1.
    profile = builder.create_optimization_profile()
    shape = (1, 3, height, width)
    profile.set_shape("image", min=shape, opt=shape, max=shape)
    config.add_optimization_profile(profile)

    start = perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_time = perf_counter() - start
    if serialized is None:
        raise RuntimeError("TRT engine build failed")
    print(f"  TRT engine built in {build_time:.1f} s")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  Engine saved: {engine_path}")

    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(serialized)


# ──────────────── TRT inference context ────────────────────────────────────

class TRTInferenceContext:
    """Thin wrapper around a TRT execution context with pre-allocated I/O."""

    def __init__(self, engine: trt.ICudaEngine, dtype: np.dtype) -> None:
        self.context = engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.dtype = dtype

        # Discover I/O tensor names and shapes.
        self.input_name = engine.get_tensor_name(0)
        self.output_name = engine.get_tensor_name(1)
        input_shape = engine.get_tensor_shape(self.input_name)
        output_shape = engine.get_tensor_shape(self.output_name)

        # Pre-allocate device buffers via PyTorch (easiest CUDA alloc).
        torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
        self.d_input = torch.empty(
            tuple(input_shape), dtype=torch_dtype, device="cuda"
        )
        self.d_output = torch.empty(
            tuple(output_shape), dtype=torch_dtype, device="cuda"
        )

        # Bind addresses.
        self.context.set_tensor_address(
            self.input_name, self.d_input.data_ptr()
        )
        self.context.set_tensor_address(
            self.output_name, self.d_output.data_ptr()
        )

    def infer(self, host_input: np.ndarray) -> torch.Tensor:
        """Copy input H→D, run inference, return device output tensor."""
        self.d_input.copy_(
            torch.from_numpy(host_input).to(self.d_input.device)
        )
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.d_output

    def infer_from_device(self, device_input: torch.Tensor) -> torch.Tensor:
        """Run inference with input already on device."""
        self.d_input.copy_(device_input)
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.d_output


# ─────────────── post-processing (matching PyTorch benchmark) ──────────────

def predict_from_logits_np(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    """Softmax → threshold on class-1 probability.  Matches ttfm.metrics."""
    probs = torch.softmax(logits.float(), dim=1)[:, 1]
    return (probs >= threshold).to(torch.int64)


# ─────────────── accuracy evaluation via TRT ───────────────────────────────

def evaluate_accuracy_trt(
    trt_ctx: TRTInferenceContext,
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
    np_dtype = np.float16 if precision == "fp16" else np.float32
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    for batch in loader:
        host_np = batch["image"].numpy().astype(np_dtype)
        logits = trt_ctx.infer(host_np).clone()
        confusion = update_confusion(
            confusion, logits.float().cpu(), batch["mask"], threshold
        )
    from ttfm.metrics import metrics_from_confusion
    return metrics_from_confusion(confusion)


# ─────────────────────── main benchmark ────────────────────────────────────

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
        raise RuntimeError("CUDA is required")
    config = load_yaml(config_path)
    threshold = float(
        config.get("postprocessing", {}).get("traversable_threshold", 0.5)
    )
    width, height = config["training"]["input_size"]
    np_dtype = np.float16 if precision == "fp16" else np.float32

    # Load test images.
    paths = [
        p
        for p in sorted(input_dir.iterdir())
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ][:limit]
    if not paths:
        raise ValueError(f"No images found in {input_dir}")
    native_frames = []
    for path in paths:
        with Image.open(path) as opened:
            native_frames.append(np.asarray(opened.convert("RGB")).copy())

    # 1. ONNX export.
    scratch = Path("reports") / ".trt_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    model_name = config["training"].get("model_name", "pidnet-s")
    onnx_path = scratch / f"{model_name}_{precision}.onnx"
    engine_path = scratch / f"{model_name}_{precision}.engine"

    print(f"Step 1/4: Exporting {model_name} to ONNX …")
    export_onnx(config, checkpoint_path, onnx_path, width, height)

    # 2. Build TRT engine.
    print(f"Step 2/4: Building TensorRT {precision.upper()} engine …")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    build_start = perf_counter()
    engine = build_engine(
        onnx_path, engine_path, fp16=(precision == "fp16"), width=width, height=height
    )
    build_time = perf_counter() - build_start

    trt_ctx = TRTInferenceContext(engine, np_dtype)
    synchronize()
    allocated_after_load = torch.cuda.memory_allocated()

    # Count parameters from the original PyTorch model for the report.
    pt_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parameter_bytes = sum(
        v.numel() * v.element_size() for v in pt_checkpoint["model_state"].values()
    )

    # 3. Warmup + benchmark.
    print(f"Step 3/4: Benchmarking ({warmup} warmup, {repeats}× {len(native_frames)} frames) …")
    first_np = cpu_tensor_from_rgb(native_frames[0], width, height, np_dtype)

    for _ in range(warmup):
        logits = trt_ctx.infer(first_np)
        _ = predict_from_logits_np(logits, threshold)
    synchronize()

    # Forward-only: input already on device, measure inference + postprocess.
    forward_latencies: list[float] = []
    for _ in range(repeats):
        for frame in native_frames:
            host_np = cpu_tensor_from_rgb(frame, width, height, np_dtype)
            device_input = torch.from_numpy(host_np).cuda()
            synchronize()
            start = perf_counter()
            logits = trt_ctx.infer_from_device(device_input)
            _ = predict_from_logits_np(logits, threshold)
            synchronize()
            forward_latencies.append(perf_counter() - start)

    # Full pipeline: CPU preprocess → H2D → inference → postprocess → D2H.
    pipeline_latencies: list[float] = []
    for _ in range(repeats):
        for frame in native_frames:
            synchronize()
            start = perf_counter()
            host_np = cpu_tensor_from_rgb(frame, width, height, np_dtype)
            logits = trt_ctx.infer(host_np)
            prediction = predict_from_logits_np(logits, threshold)
            prediction.cpu()
            synchronize()
            pipeline_latencies.append(perf_counter() - start)

    peak_allocated = torch.cuda.max_memory_allocated()

    # 4. Accuracy.
    print("Step 4/4: Evaluating accuracy …")
    accuracy = evaluate_accuracy_trt(
        trt_ctx, config, precision, threshold, accuracy_limit
    )
    synchronize()

    return {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "input_dir": str(input_dir),
        "model_name": model_name,
        "precision": precision,
        "backend": "tensorrt",
        "tensorrt_version": trt.__version__,
        "engine_build_time_s": round(build_time, 1),
        "engine_path": str(engine_path),
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
            "forward": "TRT infer (device→device) + softmax/threshold; excludes H2D and mask D2H",
            "pipeline": "CPU resize/normalise, H2D, TRT infer, softmax/threshold, mask D2H; excludes disk/camera/planner",
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
    parser = argparse.ArgumentParser(
        description="TensorRT FP16 edge-style segmentation benchmark"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--input-dir",
        default="data_processed_blue_green/test/images",
        type=Path,
    )
    parser.add_argument(
        "--precision", choices=("fp16", "fp32"), default="fp16"
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--accuracy-limit",
        type=int,
        default=0,
        help="Zero evaluates the complete 544-image test split",
    )
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
