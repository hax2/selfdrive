#!/usr/bin/env python3
"""Validate ORFD's official split and build the standard TTFM binary manifest."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


PUBLISHED_COUNTS = {"train": 8398, "val": 1245, "test": 2555}
ACADEMIC_TORRENTS_COUNTS = {"train": 8392, "val": 1245, "test": 2193}
ACCEPTED_PROFILES = {
    "published_complete": PUBLISHED_COUNTS,
    "academic_torrents_incomplete_mirror": ACADEMIC_TORRENTS_COUNTS,
}
SOURCE_SPLITS = {"train": "training", "val": "validation", "test": "testing"}


def _find_dataset_root(raw_root: Path) -> Path:
    candidates = [raw_root, raw_root / "ORFD"]
    candidates.extend(path for path in raw_root.rglob("ORFD") if path.is_dir())
    for candidate in candidates:
        if all((candidate / source).is_dir() for source in SOURCE_SPLITS.values()):
            return candidate
    raise FileNotFoundError(
        f"Could not find an ORFD root below {raw_root}; expected training, validation, and testing directories"
    )


def _safe_id(relative_parent: Path, stem: str) -> str:
    prefix = "__".join(relative_parent.parts)
    value = f"{prefix}__{stem}" if prefix else stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _link_image(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        destination.symlink_to(os.path.relpath(source, destination.parent))


def _mask_for_image(image_path: Path) -> Path:
    gt_dir = image_path.parent.parent / "gt_image"
    candidates = [gt_dir / f"{image_path.stem}_fillcolor.png", gt_dir / image_path.name]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one ground-truth mask for {image_path}, found: {matches}")
    return matches[0]


def _write_binary_mask(source: Path, destination: Path, expected_size: tuple[int, int]) -> None:
    with Image.open(source) as opened:
        rgb = np.asarray(opened.convert("RGB"))
    if (rgb.shape[1], rgb.shape[0]) != expected_size:
        raise ValueError(f"Image/mask size mismatch for {source}: {expected_size} vs {(rgb.shape[1], rgb.shape[0])}")
    # This intentionally matches the official ORFD loader: after BGR->RGB conversion,
    # label_image[:, :, 2] > 200 is the traversable class.
    binary = (rgb[:, :, 2] > 200).astype(np.uint8)
    Image.fromarray(binary).save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/ORFD"))
    parser.add_argument("--processed-root", type=Path, default=Path("data_processed_orfd"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary_path = args.processed_root / "summary.json"
    if summary_path.is_file() and not args.force:
        summary = json.loads(summary_path.read_text())
        if summary.get("split_counts") in ACCEPTED_PROFILES.values():
            print(f"Verified processed ORFD already exists: {args.processed_root}")
            return

    dataset_root = _find_dataset_root(args.raw_root)
    building_root = args.processed_root.with_name(args.processed_root.name + "_building")
    if building_root.exists():
        shutil.rmtree(building_root)
    for split in SOURCE_SPLITS:
        (building_root / split / "images").mkdir(parents=True, exist_ok=True)
        (building_root / split / "masks").mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    images_by_split = {
        split: sorted(
            path
            for directory in (dataset_root / source_name).rglob("image_data")
            for path in directory.glob("*.png")
        )
        for split, source_name in SOURCE_SPLITS.items()
    }
    split_counts = {split: len(images) for split, images in images_by_split.items()}
    matching_profiles = [name for name, counts in ACCEPTED_PROFILES.items() if counts == split_counts]
    if not matching_profiles:
        raise RuntimeError(
            f"Unrecognized ORFD split counts: {split_counts}. Expected either the complete published profile "
            f"{PUBLISHED_COUNTS} or the known incomplete Academic Torrents mirror {ACADEMIC_TORRENTS_COUNTS}."
        )
    dataset_profile = matching_profiles[0]
    if dataset_profile != "published_complete":
        print(
            "WARNING: using the incomplete Academic Torrents ORFD mirror. This supports a controlled "
            "ROD/FPN comparison but not exact comparison with the paper's full-test result.",
            flush=True,
        )
    positive_pixels = 0
    total_pixels = 0
    for split, source_name in SOURCE_SPLITS.items():
        source_root = dataset_root / source_name
        images = images_by_split[split]
        seen_ids: set[str] = set()
        for index, image_path in enumerate(images, start=1):
            relative_parent = image_path.parent.parent.relative_to(source_root)
            sample_id = _safe_id(relative_parent, image_path.stem)
            if sample_id in seen_ids:
                raise RuntimeError(f"Duplicate ORFD sample id in {split}: {sample_id}")
            seen_ids.add(sample_id)
            mask_path = _mask_for_image(image_path)
            with Image.open(image_path) as opened:
                image_size = opened.size
                opened.verify()
            output_image = building_root / split / "images" / f"{sample_id}.png"
            output_mask = building_root / split / "masks" / f"{sample_id}.png"
            _link_image(image_path.resolve(), output_image)
            _write_binary_mask(mask_path, output_mask, image_size)
            with Image.open(output_mask) as opened:
                binary = np.asarray(opened)
            positive_pixels += int(binary.sum())
            total_pixels += int(binary.size)
            manifest.append(
                {
                    "split": split,
                    "source_dataset": "ORFD",
                    "sample_id": sample_id,
                    "image_path": str(output_image.relative_to(building_root)),
                    "mask_path": str(output_mask.relative_to(building_root)),
                    "label_source": "gt_image_blue_channel_gt_200",
                    "raw_label_path": str(mask_path.relative_to(dataset_root)),
                }
            )
            if index % 500 == 0 or index == len(images):
                print(f"Prepared ORFD {split}: {index}/{len(images)}", flush=True)

    if positive_pixels == 0 or positive_pixels == total_pixels:
        raise RuntimeError("ORFD binary masks are degenerate; refusing to train")
    (building_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "dataset_name": "ORFD",
        "source_root": str(dataset_root),
        "processed_root": str(args.processed_root),
        "dataset_profile": dataset_profile,
        "split_counts": split_counts,
        "published_split_counts": PUBLISHED_COUNTS,
        "academic_torrents_split_counts": ACADEMIC_TORRENTS_COUNTS,
        "comparable_to_published_full_test": dataset_profile == "published_complete",
        "positive_class": "traversable",
        "mask_rule": "PIL RGB channel 2 > 200 (matches official ORFD loader after BGR-to-RGB conversion)",
        "traversable_pixel_fraction": positive_pixels / total_pixels,
        "image_storage": "hard links when possible, relative symlinks otherwise",
    }
    (building_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.processed_root.exists():
        shutil.rmtree(args.processed_root)
    building_root.rename(args.processed_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
