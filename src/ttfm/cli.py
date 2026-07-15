from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import run_audit
from .eval import run_evaluation
from .export import run_export
from .infer import run_inference
from .preprocess import build_processed_dataset
from .review import run_review
from .train import run_training
from .utils import load_yaml


def _load_config(path: Path) -> dict:
    config = load_yaml(path)
    config["config_path"] = str(path)
    return config


def _print_pretrain_summary(config: dict) -> None:
    processed_summary = json.loads((Path(config["processed_root"]) / "summary.json").read_text())
    if "preprocessing" not in config:
        print("Dataset:", processed_summary.get("dataset_name", "unknown"))
        print("Sample counts:", processed_summary["split_counts"])
        print("Positive class:", processed_summary.get("positive_class", "traversable"))
        print("Mask rule:", processed_summary.get("mask_rule", "recorded in processed manifest"))
        return
    mapping_filename = config.get("preprocessing", {}).get("mapping_filename", "discovered_mapping.yaml")
    mapping = load_yaml(Path(config["configs_dir"]) / mapping_filename)
    print("Chosen supervision source:", mapping["chosen_supervision_source"])
    print("Unique raw class IDs:", mapping["global_unique_int_map_ids"])
    print("Inferred color-to-ID correspondence:")
    for color_key, details in mapping["inferred_palette_id_mapping"].items():
        print(f"  {color_key} ({details['palette_name']}) -> {details['best_id']} (confidence={details['confidence']:.4f})")
    print("Final binary mapping:", mapping["binary_id_mapping"])
    print("Sample counts:", processed_summary["split_counts"])
    print("Risks:")
    for risk in mapping["risks"]:
        print(f"  - {risk}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Traversability segmentation pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit")
    subparsers.add_parser("preprocess")
    subparsers.add_parser("train")
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--split", default="test")
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--input-dir", required=True)
    infer_parser.add_argument("--output-dir", required=True)
    subparsers.add_parser("export")
    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--split", default="test")
    review_parser.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()
    config = _load_config(Path(args.config))

    if args.command == "audit":
        preprocessing = config.get("preprocessing", {})
        payload = run_audit(
            Path(config["raw_root"]),
            Path(config["reports_dir"]),
            Path(config["configs_dir"]),
            traversable_palette_names=list(preprocessing.get("traversable_palette_names", ["blue"])),
            mapping_filename=str(preprocessing.get("mapping_filename", "discovered_mapping.yaml")),
        )
        print(json.dumps({"validated_sample_count": payload["validated_sample_count"]}, indent=2))
    elif args.command == "preprocess":
        payload = build_processed_dataset(config)
        print(json.dumps(payload, indent=2))
    elif args.command == "train":
        _print_pretrain_summary(config)
        payload = run_training(config)
        print(json.dumps(payload, indent=2))
    elif args.command == "eval":
        payload = run_evaluation(config, split=args.split)
        print(json.dumps(payload, indent=2))
    elif args.command == "infer":
        payload = run_inference(config, Path(args.input_dir), Path(args.output_dir))
        print(json.dumps(payload, indent=2))
    elif args.command == "export":
        payload = run_export(config)
        print(json.dumps(payload, indent=2))
    elif args.command == "review":
        payload = run_review(config, split=args.split, top_k=args.top_k)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
