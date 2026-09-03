#!/usr/bin/env python3
"""
build_visualizer_data.py
Consolidates all experimental ledgers, convergence curves, hardware benchmarks,
ablation results, and qualitative sample metadata into a unified JSON dataset
for the TTFM Traversability Web Visualizer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = WORKSPACE_ROOT / "visualizer" / "public" / "data" / "benchmark_data.json"

MODEL_PRETTY_NAMES = {
    "fpn_efficientnet_b0": "FPN/EfficientNet-B0",
    "unet_efficientnet_b0": "U-Net/EfficientNet-B0",
    "fpn_mobilenetv2": "FPN/MobileNetV2",
    "segformer_b0": "SegFormer-B0",
    "unet_mobilenetv2": "U-Net/MobileNetV2",
    "rod_vits": "ROD ViT-S",
    "pidnet_s": "PIDNet-S",
    "bisenetv2": "BiSeNetV2",
    "ddrnet23_slim": "DDRNet-23-Slim",
}

MODEL_ARCH_DETAILS = {
    "FPN/EfficientNet-B0": {
        "family": "Lightweight FPN",
        "backbone": "EfficientNet-B0 (Pretrained)",
        "decoder": "Feature Pyramid Network",
        "description": "Best overall test accuracy and lowest False-Safe Rate. Balances high mIoU with efficient depthwise separable convolution features."
    },
    "U-Net/EfficientNet-B0": {
        "family": "Encoder-Decoder",
        "backbone": "EfficientNet-B0 (Pretrained)",
        "decoder": "U-Net with Skip Connections",
        "description": "Symmetric U-Net architecture with strong boundary recovery and high precision traversability predictions."
    },
    "FPN/MobileNetV2": {
        "family": "Lightweight FPN",
        "backbone": "MobileNetV2 (Inverted Residuals)",
        "decoder": "Feature Pyramid Network",
        "description": "Ultra-lightweight real-time FPN with inverted bottleneck residual blocks, delivering 147 FPS on H100 and low CPU overhead."
    },
    "SegFormer-B0": {
        "family": "Transformer",
        "backbone": "MiT-B0 (Hierarchical ViT)",
        "decoder": "All-MLP Decoder",
        "description": "Lightweight transformer with multi-scale self-attention without positional encodings, yielding robust long-range context."
    },
    "U-Net/MobileNetV2": {
        "family": "Encoder-Decoder",
        "backbone": "MobileNetV2 (Pretrained)",
        "decoder": "U-Net Skip Connections",
        "description": "Compact encoder-decoder with MobileNetV2 features, highly responsive across varying terrain granularities."
    },
    "ROD ViT-S": {
        "family": "Foundation Model Transfer",
        "backbone": "EfficientSAM ViT-S (Frozen 29.1M weights)",
        "decoder": "Custom Traversability Head",
        "description": "Foundation model zero/few-shot approach using frozen Segment Anything ViT-S backbone. Strong features but significantly higher latency and parameter cost."
    },
    "PIDNet-S": {
        "family": "Three-Branch Real-Time CNN",
        "backbone": "Three-Branch (Detail, Context, Boundary)",
        "decoder": "PID Control Fusion",
        "description": "Real-time edge champion with proportional-integral-derivative boundary alignment. Achieves 1038 FPS forward pass on RTX 5060 with TensorRT."
    },
    "BiSeNetV2": {
        "family": "Bilateral Real-Time CNN",
        "backbone": "Bilateral (Detail Branch + Semantic Branch)",
        "decoder": "Bilateral Guided Aggregation",
        "description": "Two-stream high-speed architecture separating spatial details from high-level semantics. Excellent speed on embedded CPUs."
    },
    "DDRNet-23-Slim": {
        "family": "Dual-Resolution Real-Time CNN",
        "backbone": "Dual-Resolution ResNet-style",
        "decoder": "Bilateral Fusion",
        "description": "Trained from scratch. Dual-stream resolution mechanism reaching 245 FPS on H100 and up to 36.5 FPS on compiled Ryzen CPU."
    }
}


def load_json_safe(path: Path) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: Failed to parse {path}: {e}")
            return None
    return None


def main():
    print("Building visualizer dataset from workspace artifacts...")

    # 1. Load Ledgers
    conv_ledger = load_json_safe(WORKSPACE_ROOT / "reports" / "convergence_blue_green_verified_ledger.json") or {}
    bg_ledger = load_json_safe(WORKSPACE_ROOT / "reports" / "blue_green_verified_ledger.json") or {}
    bo_ledger = load_json_safe(WORKSPACE_ROOT / "reports" / "blue_only_verified_ledger.json") or {}

    # Hardware summaries
    edge_summary = load_json_safe(WORKSPACE_ROOT / "reports" / "edge_deployment_summary.json") or {}
    edge_pidnet_trt = load_json_safe(WORKSPACE_ROOT / "reports" / "edge_pidnet_rtx5060_trt.json") or {}
    edge_rod_trt = load_json_safe(WORKSPACE_ROOT / "reports" / "edge_rod_rtx5060_trt.json") or {}
    edge_pidnet_py = load_json_safe(WORKSPACE_ROOT / "reports" / "edge_pidnet_rtx5060.json") or {}
    edge_rod_py = load_json_safe(WORKSPACE_ROOT / "reports" / "edge_rod_rtx5060.json") or {}
    ablations_data = load_json_safe(WORKSPACE_ROOT / "reports" / "controlled_ablation_summary.json") or {}
    test_review_summary = load_json_safe(WORKSPACE_ROOT / "outputs" / "mixed_binary_traversability" / "test_review" / "summary.json") or {}

    # 2. Extract Convergence Suite Models
    convergence_models = []
    for agg in conv_ledger.get("aggregates", []):
        name = agg["model"]
        arch_info = MODEL_ARCH_DETAILS.get(name, {})
        
        # Get individual runs matching this model
        runs = [r for r in conv_ledger.get("runs", []) if r.get("model") == name]
        run_items = []
        for r in runs:
            m = r.get("metrics", {})
            run_items.append({
                "seed": r.get("seed"),
                "best_epoch": r.get("best_epoch"),
                "epochs_completed": r.get("epochs_completed"),
                "mIoU": m.get("mIoU"),
                "f1": m.get("f1_traversable"),
                "fsr": m.get("false_safe_rate"),
                "fbr": m.get("false_block_rate"),
                "iou_traversable": m.get("iou_traversable"),
                "iou_untraversable": m.get("iou_untraversable"),
                "precision": m.get("precision_traversable"),
                "recall": m.get("recall_traversable"),
                "loss": r.get("best_val_mIoU"),
                "confusion_counts": m.get("confusion_counts_traversable")
            })

        convergence_models.append({
            "model": name,
            "family": arch_info.get("family", "CNN"),
            "backbone": arch_info.get("backbone", "N/A"),
            "decoder": arch_info.get("decoder", "N/A"),
            "description": arch_info.get("description", ""),
            "selected_epochs": agg.get("selected_epochs", []),
            "mIoU_mean": agg["mIoU"]["mean"],
            "mIoU_std": agg["mIoU"]["std_population"],
            "f1_mean": agg["f1_traversable"]["mean"],
            "f1_std": agg["f1_traversable"]["std_population"],
            "fsr_mean": agg["false_safe_rate"]["mean"],
            "fsr_std": agg["false_safe_rate"]["std_population"],
            "fbr_mean": agg["false_block_rate"]["mean"],
            "fbr_std": agg["false_block_rate"]["std_population"],
            "gain_pp": agg.get("mIoU_gain_percentage_points", 0.0),
            "fixed_15_mIoU": agg.get("fixed_15_epoch_mIoU", {}).get("mean", 0.0) if isinstance(agg.get("fixed_15_epoch_mIoU"), dict) else (agg.get("fixed_15_epoch_mIoU") if isinstance(agg.get("fixed_15_epoch_mIoU"), (int, float)) else 0.0),
            "runs": run_items
        })
    # Sort by mIoU descending
    convergence_models.sort(key=lambda x: x["mIoU_mean"], reverse=True)

    # 3. Extract Blue+Green 15-epoch Models
    blue_green_models = []
    for agg in bg_ledger.get("aggregates", []):
        name = agg["model"]
        arch_info = MODEL_ARCH_DETAILS.get(name, {})
        runs = [r for r in bg_ledger.get("runs", []) if r.get("model") == name]
        run_items = []
        for r in runs:
            m = r.get("test_metrics", {}).get("metrics", {})
            run_items.append({
                "seed": r.get("seed"),
                "mIoU": m.get("mIoU"),
                "f1": m.get("f1_traversable"),
                "fsr": m.get("false_safe_rate"),
                "fbr": m.get("false_block_rate"),
                "precision": m.get("precision_traversable"),
                "recall": m.get("recall_traversable")
            })

        h100 = agg.get("h100_eager", {})
        ryzen_comp = agg.get("ryzen_compiled", {})
        ryzen_eag = agg.get("ryzen_eager", {})

        blue_green_models.append({
            "model": name,
            "family": arch_info.get("family", "CNN"),
            "backbone": arch_info.get("backbone", "N/A"),
            "decoder": arch_info.get("decoder", "N/A"),
            "description": arch_info.get("description", ""),
            "parameters": agg.get("parameters", 0),
            "trainable_parameters": agg.get("trainable_parameters", 0),
            "mIoU_mean": agg["mIoU"]["mean"],
            "mIoU_std": agg["mIoU"]["std_population"],
            "f1_mean": agg["f1_traversable"]["mean"],
            "f1_std": agg["f1_traversable"]["std_population"],
            "fsr_mean": agg["false_safe_rate"]["mean"],
            "fsr_std": agg["false_safe_rate"]["std_population"],
            "fbr_mean": agg["false_block_rate"]["mean"],
            "fbr_std": agg["false_block_rate"]["std_population"],
            "h100_fps": h100.get("fps") if h100.get("available") else None,
            "h100_latency_ms": h100.get("mean_ms") if h100.get("available") else None,
            "ryzen_compiled_fps": ryzen_comp.get("fps") if ryzen_comp.get("available") else None,
            "ryzen_compiled_latency_ms": ryzen_comp.get("mean_ms") if ryzen_comp.get("available") else None,
            "ryzen_eager_fps": ryzen_eag.get("fps") if ryzen_eag.get("available") else None,
            "runs": run_items
        })
    blue_green_models.sort(key=lambda x: x["mIoU_mean"], reverse=True)

    # 4. Extract Blue-Only 15-epoch Models
    blue_only_models = []
    for agg in bo_ledger.get("aggregates", []):
        name = agg["model"]
        arch_info = MODEL_ARCH_DETAILS.get(name, {})
        runs = [r for r in bo_ledger.get("runs", []) if r.get("model") == name]
        run_items = []
        for r in runs:
            m = r.get("test_metrics", {}).get("metrics", {})
            run_items.append({
                "seed": r.get("seed"),
                "mIoU": m.get("mIoU"),
                "f1": m.get("f1_traversable"),
                "fsr": m.get("false_safe_rate"),
                "fbr": m.get("false_block_rate"),
                "precision": m.get("precision_traversable"),
                "recall": m.get("recall_traversable")
            })

        h100 = agg.get("h100_eager", {})
        ryzen_comp = agg.get("ryzen_compiled", {})
        ryzen_eag = agg.get("ryzen_eager", {})

        blue_only_models.append({
            "model": name,
            "family": arch_info.get("family", "CNN"),
            "backbone": arch_info.get("backbone", "N/A"),
            "decoder": arch_info.get("decoder", "N/A"),
            "description": arch_info.get("description", ""),
            "parameters": agg.get("parameters", 0),
            "trainable_parameters": agg.get("trainable_parameters", 0),
            "mIoU_mean": agg["mIoU"]["mean"],
            "mIoU_std": agg["mIoU"]["std_population"],
            "f1_mean": agg["f1_traversable"]["mean"],
            "f1_std": agg["f1_traversable"]["std_population"],
            "fsr_mean": agg["false_safe_rate"]["mean"],
            "fsr_std": agg["false_safe_rate"]["std_population"],
            "fbr_mean": agg["false_block_rate"]["mean"],
            "fbr_std": agg["false_block_rate"]["std_population"],
            "h100_fps": h100.get("fps") if h100.get("available") else None,
            "h100_latency_ms": h100.get("mean_ms") if h100.get("available") else None,
            "ryzen_compiled_fps": ryzen_comp.get("fps") if ryzen_comp.get("available") else None,
            "ryzen_compiled_latency_ms": ryzen_comp.get("mean_ms") if ryzen_comp.get("available") else None,
            "ryzen_eager_fps": ryzen_eag.get("fps") if ryzen_eag.get("available") else None,
            "runs": run_items
        })
    blue_only_models.sort(key=lambda x: x["mIoU_mean"], reverse=True)

    # 5. Extract Convergence Epoch Curves for Seed 1337
    curves_data = {}
    curve_files = list(WORKSPACE_ROOT.glob("outputs/convergence_blue_green_*_seed1337_e300_c60_m60_p25/history.json"))
    for cf in curve_files:
        raw_slug = cf.parent.name.replace("convergence_blue_green_", "").replace("_seed1337_e300_c60_m60_p25", "")
        pretty = MODEL_PRETTY_NAMES.get(raw_slug, raw_slug)
        hist = load_json_safe(cf) or []
        
        # Subsample if needed, or keep all points
        epoch_points = []
        for pt in hist:
            vm = pt.get("val_metrics", {})
            epoch_points.append({
                "epoch": pt.get("epoch"),
                "val_mIoU": vm.get("mIoU"),
                "val_fsr": vm.get("false_safe_rate"),
                "val_fbr": vm.get("false_block_rate"),
                "val_f1": vm.get("f1_traversable"),
                "val_loss": pt.get("val_loss"),
                "train_loss": pt.get("train_loss"),
                "lr": pt.get("learning_rate")
            })
        curves_data[pretty] = epoch_points

    # 6. Edge Deployment Benchmark Comparison
    hardware_benchmarks = {
        "platforms": [
            {
                "id": "h100",
                "name": "NVIDIA H100 NVL MIG 3g.47gb",
                "type": "Cloud / Server GPU",
                "notes": "Batch size 1, 640x384 input resolution, eager forward pass."
            },
            {
                "id": "rtx5060",
                "name": "NVIDIA GeForce RTX 5060 (Edge GPU)",
                "type": "Embedded / Edge GPU",
                "notes": "Batch size 1, 640x384, comparing FP16 PyTorch eager vs TensorRT 10.16 FP16 Engine."
            },
            {
                "id": "ryzen5500",
                "name": "AMD Ryzen 5 5500 (6-Core CPU)",
                "type": "Embedded Robot CPU",
                "notes": "Batch size 1, 640x384, PyTorch 2.1+ Eager vs Inductor AOT/JIT Compiled."
            }
        ],
        "rtx5060_details": {
            "pidnet_s": {
                "model": "PIDNet-S",
                "trt_forward_ms": edge_pidnet_trt.get("forward", {}).get("mean_ms", 0.963),
                "trt_forward_fps": edge_pidnet_trt.get("forward", {}).get("fps_from_mean_latency", 1037.97),
                "trt_pipeline_ms": edge_pidnet_trt.get("pipeline", {}).get("mean_ms", 9.40),
                "trt_pipeline_fps": edge_pidnet_trt.get("pipeline", {}).get("fps_from_mean_latency", 106.43),
                "pytorch_forward_ms": edge_summary.get("pidnet_s", {}).get("forward_mean_ms", 8.37),
                "pytorch_forward_fps": edge_summary.get("pidnet_s", {}).get("forward_fps", 119.41),
                "pytorch_pipeline_ms": edge_summary.get("pidnet_s", {}).get("pipeline_mean_ms", 14.90),
                "pytorch_pipeline_fps": edge_summary.get("pidnet_s", {}).get("pipeline_fps", 67.13),
                "param_memory_mib": edge_summary.get("pidnet_s", {}).get("parameter_memory_mib", 14.54),
                "peak_vram_mib": edge_summary.get("pidnet_s", {}).get("peak_allocated_mib", 57.86),
                "test_miou_fp16": edge_summary.get("pidnet_s", {}).get("fp16_full_test_miou", 0.89396),
                "test_miou_fp32": edge_summary.get("pidnet_s", {}).get("fp32_full_test_miou", 0.89394)
            },
            "rod_vits": {
                "model": "ROD ViT-S",
                "trt_available": False,
                "pytorch_forward_ms": edge_summary.get("rod", {}).get("forward_mean_ms", 31.39),
                "pytorch_forward_fps": edge_summary.get("rod", {}).get("forward_fps", 31.86),
                "pytorch_pipeline_ms": edge_summary.get("rod", {}).get("pipeline_mean_ms", 38.36),
                "pytorch_pipeline_fps": edge_summary.get("rod", {}).get("pipeline_fps", 26.07),
                "param_memory_mib": edge_summary.get("rod", {}).get("parameter_memory_mib", 55.52),
                "peak_vram_mib": edge_summary.get("rod", {}).get("peak_allocated_mib", 679.97),
                "test_miou_fp16": edge_summary.get("rod", {}).get("fp16_full_test_miou", 0.92357),
                "test_miou_fp32": edge_summary.get("rod", {}).get("fp32_full_test_miou", 0.92357)
            }
        },
        "model_comparison_table": []
    }

    # Build cross-platform comparison table for all 9 models
    for m in blue_green_models:
        name = m["model"]
        h100_fps = m.get("h100_fps")
        ryzen_comp = m.get("ryzen_compiled_fps")
        ryzen_eag = m.get("ryzen_eager_fps")
        params = m.get("parameters")
        
        rtx_trt_fps = 1037.97 if name == "PIDNet-S" else None
        rtx_py_fps = 119.41 if name == "PIDNet-S" else (31.86 if name == "ROD ViT-S" else None)

        hardware_benchmarks["model_comparison_table"].append({
            "model": name,
            "params": params,
            "h100_fps": h100_fps,
            "rtx5060_trt_fps": rtx_trt_fps,
            "rtx5060_pytorch_fps": rtx_py_fps,
            "ryzen_compiled_fps": ryzen_comp,
            "ryzen_eager_fps": ryzen_eag,
            "mIoU_bg": m["mIoU_mean"],
            "fsr_bg": m["fsr_mean"],
            "fbr_bg": m["fbr_mean"]
        })

    # 7. Controlled Ablation Results
    ablations_list = []
    for ab in ablations_data.get("runs", []):
        exp = ab.get("experiment", "")
        friendly_title = {
            "controlled_ablation_baseline": "Baseline (Standard Recipe)",
            "controlled_ablation_loss_false_safe": "Loss: Penalise False-Safe (+0.5x weight)",
            "controlled_ablation_augment_off": "Augmentations: OFF (No Random Transforms)",
            "controlled_ablation_resolution_720x448": "Resolution: 720x448 (Higher Res Input)",
            "controlled_ablation_class_weights_neutral": "Class Weights: Neutral (1.0 vs 1.0)"
        }.get(exp, exp)

        ablations_list.append({
            "experiment": exp,
            "title": friendly_title,
            "mIoU": ab.get("mIoU"),
            "fsr": ab.get("FSR"),
            "fbr": ab.get("FBR"),
            "confusion": ab.get("confusion_counts_traversable", {})
        })

    # 8. Frozen PIDNet Reference Detail
    frozen_pidnet = {
        "mIoU": 0.893941,
        "iou_traversable": 0.878946,
        "iou_untraversable": 0.908936,
        "precision": 0.941592,
        "recall": 0.929631,
        "f1": 0.935573,
        "false_safe_rate": 0.043176,
        "false_block_rate": 0.070369,
        "loss": 0.149528,
        "total_test_pixels": 133693440,
        "tn": 73151244,
        "fp": 3300886,
        "fn": 4027996,
        "tp": 53213314
    }

    # 9. Qualitative Visual Inspection Samples
    qualitative_gallery = {
        "worst_false_safe": [
            {
                "id": "mixed_pln_1278",
                "title": "Severe False Safe in Hazard Depression",
                "category": "worst_false_safe",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_pln_1278.png",
                "overlay_url": "/media/outputs/mixed_binary_traversability/test_review/all_overlays/mixed_pln_1278.png",
                "detail_img": "/media/reports/figures/qualitative/pidnet_failure_false_safe_mixed_pln_1278.png",
                "description": "The model incorrectly classifies an untraversable trench/depression as traversable path, posing an acute rollover hazard.",
                "fsr": 0.235,
                "fbr": 0.021
            },
            {
                "id": "mixed_pln_610",
                "title": "False Safe on Dense Off-Road Brush",
                "category": "worst_false_safe",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_pln_610.png",
                "overlay_url": "/media/outputs/mixed_binary_traversability/test_review/all_overlays/mixed_pln_610.png",
                "detail_img": "/media/outputs/mixed_binary_traversability/test_review/all_overlays/mixed_pln_610.png",
                "description": "Vegetation clusters with irregular shadows misclassified as passable ground.",
                "fsr": 0.198,
                "fbr": 0.034
            },
            {
                "id": "mixed_pln_546",
                "title": "Overhanging Branch & Shadow Ambiguity",
                "category": "worst_false_safe",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_pln_546.png",
                "overlay_url": "/media/outputs/mixed_binary_traversability/test_review/all_overlays/mixed_pln_546.png",
                "detail_img": "/media/outputs/mixed_binary_traversability/test_review/all_overlays/mixed_pln_546.png",
                "description": "Sun glare through overhanging vegetation dilutes depth cues, causing boundary over-expansion.",
                "fsr": 0.182,
                "fbr": 0.041
            }
        ],
        "worst_false_block": [
            {
                "id": "mixed_425",
                "title": "False Block on Path Dust & Texture Variation",
                "category": "worst_false_block",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_425.png",
                "overlay_url": "/media/outputs/mixed_binary_traversability/test_review/all_overlays/mixed_425.png",
                "detail_img": "/media/reports/figures/qualitative/pidnet_failure_false_block_mixed_425.png",
                "description": "Sharp surface texture transition on perfectly flat gravel causes false obstacle prediction, triggering phantom braking.",
                "fsr": 0.012,
                "fbr": 0.442
            },
            {
                "id": "mixed_306",
                "title": "False Block in Tree Canopy Shadow",
                "category": "worst_false_block",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_306.png",
                "overlay_url": "/media/outputs/mixed_binary_traversability/test_review/all_overlays/mixed_306.png",
                "detail_img": "/media/outputs/mixed_binary_traversability/test_review/all_overlays/mixed_306.png",
                "description": "Dense ground shadows from pine canopy cause false negative block classification.",
                "fsr": 0.009,
                "fbr": 0.381
            },
            {
                "id": "mixed_394",
                "title": "Transition False Block at Curve Entrance",
                "category": "worst_false_block",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_394.png",
                "overlay_url": "/media/outputs/mixed_binary_traversability/test_review/all_overlays/mixed_394.png",
                "detail_img": "/media/outputs/mixed_binary_traversability/test_review/worst_false_block/mixed_394.png",
                "description": "Unpaved track turns into light brush; model is over-conservative on the inside turn.",
                "fsr": 0.015,
                "fbr": 0.354
            }
        ],
        "worst_boundary": [
            {
                "id": "mixed_459",
                "title": "Soft Trail Edge Boundary Ambiguity",
                "category": "worst_boundary",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_459.png",
                "overlay_url": "/media/outputs/mixed_binary_traversability/test_review/all_overlays/mixed_459.png",
                "detail_img": "/media/reports/figures/qualitative/pidnet_failure_boundary_mixed_459.png",
                "description": "Gradual dirt-to-grass transition lacking sharp physical edges.",
                "fsr": 0.048,
                "fbr": 0.092
            },
            {
                "id": "mixed_313",
                "title": "Boundary Error on Undulating Grass Verge",
                "category": "worst_boundary",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_313.png",
                "overlay_url": "/media/outputs/mixed_binary_traversability/test_review/all_overlays/mixed_313.png",
                "detail_img": "/media/outputs/mixed_binary_traversability/test_review/all_overlays/mixed_313.png",
                "description": "Irregular terrain fringe where ground truth boundary exhibits high local variance.",
                "fsr": 0.051,
                "fbr": 0.088
            }
        ],
        "percentiles": [
            {
                "id": "pidnet_percentile_10",
                "title": "10th Percentile (Hardest 10%) - mixed_pln_961",
                "category": "percentiles",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_pln_961.png",
                "detail_img": "/media/reports/figures/qualitative/pidnet_percentile_10_mixed_pln_961.png",
                "description": "Challenging terrain with steep light contrasts and ambiguous traversable boundary."
            },
            {
                "id": "pidnet_percentile_30",
                "title": "30th Percentile - mixed_pln_1209",
                "category": "percentiles",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_pln_1209.png",
                "detail_img": "/media/reports/figures/qualitative/pidnet_percentile_30_mixed_pln_1209.png",
                "description": "Rough dirt track with sparse grass patches."
            },
            {
                "id": "pidnet_percentile_50",
                "title": "50th Percentile (Median) - mixed_pln_119",
                "category": "percentiles",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_pln_119.png",
                "detail_img": "/media/reports/figures/qualitative/pidnet_percentile_50_mixed_pln_119.png",
                "description": "Representative median performance sample with sharp path delineation."
            },
            {
                "id": "pidnet_percentile_70",
                "title": "70th Percentile - mixed_pln_1033",
                "category": "percentiles",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_pln_1033.png",
                "detail_img": "/media/reports/figures/qualitative/pidnet_percentile_70_mixed_pln_1033.png",
                "description": "Clean trail geometry with strong foreground traversability confidence."
            },
            {
                "id": "pidnet_percentile_90",
                "title": "90th Percentile (Top 10%) - mixed_89",
                "category": "percentiles",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_89.png",
                "detail_img": "/media/reports/figures/qualitative/pidnet_percentile_90_mixed_89.png",
                "description": "Near-flawless mask alignment matching ground truth annotations across both near and far fields."
            }
        ],
        "alice_sequence": [
            {
                "id": "alice_seq01",
                "title": "ALICE Sequence Frame 01",
                "category": "alice_sequence",
                "img_url": "/media/reports/figures/qualitative/pidnet_alice_sequence01.png",
                "description": "Field robotics deployment trial: ALICE mobile rover encountering unpaved field terrain."
            },
            {
                "id": "alice_seq02",
                "title": "ALICE Sequence Frame 02",
                "category": "alice_sequence",
                "img_url": "/media/reports/figures/qualitative/pidnet_alice_sequence02.png",
                "description": "Temporal progression: PIDNet-S maintains stable corridor detection across robot pitch movements."
            },
            {
                "id": "alice_seq04",
                "title": "ALICE Sequence Frame 04",
                "category": "alice_sequence",
                "img_url": "/media/reports/figures/qualitative/pidnet_alice_sequence04.png",
                "description": "Navigation corridor tracking under varying natural lighting."
            },
            {
                "id": "alice_seq05",
                "title": "ALICE Sequence Frame 05",
                "category": "alice_sequence",
                "img_url": "/media/reports/figures/qualitative/pidnet_alice_sequence05.png",
                "description": "Stable boundary prediction avoiding corridor flicker or false positive safe regions."
            }
        ],
        "success_cases": [
            {
                "id": "mixed_pln_1188",
                "title": "Success Case - mixed_pln_1188",
                "category": "success_cases",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_pln_1188.png",
                "detail_img": "/media/reports/figures/qualitative/pidnet_success_mixed_pln_1188.png",
                "description": "Exemplary boundary crispness on double-track forest road."
            },
            {
                "id": "mixed_pln_1282",
                "title": "Success Case - mixed_pln_1282",
                "category": "success_cases",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_pln_1282.png",
                "detail_img": "/media/reports/figures/qualitative/pidnet_success_mixed_pln_1282.png",
                "description": "Clean segmentation over open meadow with clear horizon separation."
            },
            {
                "id": "mixed_pln_1306",
                "title": "Success Case - mixed_pln_1306",
                "category": "success_cases",
                "triptych_url": "/media/outputs/mixed_binary_traversability/test_review/all_triptychs/mixed_pln_1306.png",
                "detail_img": "/media/reports/figures/qualitative/pidnet_success_mixed_pln_1306.png",
                "description": "High fidelity segmentation of narrow path flanked by obstacles."
            }
        ]
    }

    # 10. Thesis Theory Figures
    thesis_figures = [
        {
            "id": "01_semantic_segmentation_concept",
            "title": "Semantic Segmentation in Unstructured Off-Road Domains",
            "img_url": "/media/look_here/01_semantic_segmentation_concept.png",
            "description": "Formulation of binary traversability segmentation mapping RGB camera frames to pixel-level traversable/untraversable masks for path planning."
        },
        {
            "id": "02_depthwise_convolution",
            "title": "Depthwise Separable Convolutions & Inverted Residuals",
            "img_url": "/media/look_here/02_depthwise_convolution.png",
            "description": "Architectural principle enabling real-time lightweight backbones (MobileNetV2, EfficientNet-B0) by factorizing spatial filtering and channel cross-projection."
        },
        {
            "id": "03_frozen_transfer_learning",
            "title": "Frozen Foundation Model Transfer Learning (ROD / SAM)",
            "img_url": "/media/look_here/03_frozen_transfer_learning.png",
            "description": "Visualisation of frozen EfficientSAM ViT-S encoder paired with a lightweight traversability decoder head for few-shot adaptation."
        },
        {
            "id": "04_cat_data_partition",
            "title": "CaT Dataset Hierarchy & Data Partitioning",
            "img_url": "/media/look_here/04_cat_data_partition.png",
            "description": "Hierarchical breakdown of the CaT off-road dataset into 1,002 train, 266 val, and 544 test samples across blue and green terrain classes."
        },
        {
            "id": "05_confusion_matrix_visual",
            "title": "Pixel-Level Confusion Matrix & Asymmetric Error Space",
            "img_url": "/media/look_here/05_confusion_matrix_visual.png",
            "description": "Definition of True Positives, True Negatives, False Safe (FP, collision hazard), and False Block (FN, unnecessary stop) over 133.7M pixel decisions."
        },
        {
            "id": "06_tensorrt_frame_budget",
            "title": "Edge Compute Frame Budget Allocation",
            "img_url": "/media/look_here/06_tensorrt_frame_budget.png",
            "description": "Breakdown of the 100 Hz / 30 Hz autonomous robot control loop, showing inference vs camera I/O, resizing, host-to-device, and planner overhead."
        },
        {
            "id": "07_safety_evidence_ladder",
            "title": "Safety Evidence Ladder for Autonomous Vehicles",
            "img_url": "/media/look_here/07_safety_evidence_ladder.png",
            "description": "Hierarchical validation framework linking offline benchmarks, multi-seed statistical significance, asymmetric safety bounds, and on-robot deployment."
        },
        {
            "id": "convergence_validation_curves",
            "title": "Convergence Suite Multi-Architecture Validation Curves",
            "img_url": "/media/look_here/convergence_validation_curves.png",
            "description": "Empirical validation curves over 300 epochs showing early stopping trigger points under patience 25."
        },
        {
            "id": "convergence_miou_comparison",
            "title": "Convergence-Aware vs 15-Epoch Fixed Budget mIoU",
            "img_url": "/media/look_here/convergence_miou_comparison.png",
            "description": "Paired bar comparison showing systematic accuracy gains unlocked by convergence scheduling across all 9 models."
        },
        {
            "id": "convergence_selected_epochs",
            "title": "Selected Checkpoint Epoch Distribution",
            "img_url": "/media/look_here/convergence_selected_epochs.png",
            "description": "Distribution of best-validation epochs across seeds 1337, 2027, and 4242, illustrating varying model convergence rates."
        }
    ]

    # 11. Index of all 544 test samples
    test_samples = []
    triptych_dir = WORKSPACE_ROOT / "outputs" / "mixed_binary_traversability" / "test_review" / "all_triptychs"
    if triptych_dir.exists():
        for f in sorted(triptych_dir.glob("*.png")):
            name = f.stem
            test_samples.append({
                "id": name,
                "triptych_url": f"/media/outputs/mixed_binary_traversability/test_review/all_triptychs/{f.name}",
                "overlay_url": f"/media/outputs/mixed_binary_traversability/test_review/all_overlays/{f.name}",
                "raw_rgb_url": f"/media/data_processed_blue_green/test/images/{f.name}",
                "raw_mask_url": f"/media/data_processed_blue_green/test/masks/{f.name}"
            })

    # Assemble complete visualizer payload
    payload = {
        "meta": {
            "title": "Learning Binary Traversability from Monocular RGB for Off-Road Navigation",
            "subtitle": "Interactive Research Visualizer & Comprehensive Benchmark Explorer",
            "author": "Master's Thesis / UC3M (Universidad Carlos III de Madrid)",
            "dataset": {
                "name": "CaT (CAVS Traversability Dataset)",
                "total_samples": 1812,
                "train_samples": 1002,
                "val_samples": 266,
                "test_samples": 544,
                "resolution": "640x384",
                "test_pixel_decisions": 133693440
            },
            "date": "2026",
            "total_architectures": 9,
            "total_controlled_runs": len(conv_ledger.get("runs", [])) + len(bg_ledger.get("runs", [])) + len(bo_ledger.get("runs", []))
        },
        "kpis": [
            {
                "label": "Top Test Accuracy",
                "value": "0.9423",
                "sub": "FPN/EfficientNet-B0 (mIoU)",
                "color": "cyan"
            },
            {
                "label": "Edge TensorRT Speed",
                "value": "1038 FPS",
                "sub": "PIDNet-S on RTX 5060 (0.96 ms)",
                "color": "emerald"
            },
            {
                "label": "Lowest False-Safe Rate",
                "value": "2.55%",
                "sub": "FPN/EfficientNet-B0 (FSR safety)",
                "color": "indigo"
            },
            {
                "label": "Max Convergence Gain",
                "value": "+3.73 pp",
                "sub": "PIDNet-S (vs 15-epoch budget)",
                "color": "amber"
            }
        ],
        "suites": {
            "convergence": convergence_models,
            "blue_green": blue_green_models,
            "blue_only": blue_only_models
        },
        "curves": curves_data,
        "hardware": hardware_benchmarks,
        "ablations": ablations_list,
        "frozen_reference": frozen_pidnet,
        "qualitative": qualitative_gallery,
        "thesis_figures": thesis_figures,
        "test_samples": test_samples
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Successfully compiled {len(payload)} sections into {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
