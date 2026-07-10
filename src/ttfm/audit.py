from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .utils import ensure_dir, save_json, save_yaml


DATASET_NAMES = ["Brown_Field", "Main_Trail", "Power_Line", "mixed"]
SPLITS = ["Train", "Test"]
PIXEL_INSPECTION_LIMIT = 4
PALETTE_NAMES = {
    "0,0,0": "black",
    "27,122,235": "blue",
    "56,162,4": "green",
    "245,34,45": "red",
}


@dataclass
class SampleRecord:
    dataset: str
    split: str
    sample_id: str
    image_path: str
    mask_path: str
    anno_path: str | None
    int_map_path: str | None
    image_size: tuple[int, int] | None
    mask_size: tuple[int, int] | None
    int_map_size: tuple[int, int] | None
    exact_int_pair: bool
    status: str


def parse_numeric_id(path: Path) -> str:
    stem = path.stem
    for prefix in ("img_", "mask_", "anno_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def pil_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image)


def relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _cooccurrence(mask_rgb: np.ndarray, int_map: np.ndarray) -> dict[str, dict[str, int]]:
    overlap: dict[str, dict[str, int]] = {}
    for color in np.unique(mask_rgb.reshape(-1, 3), axis=0):
        color_key = ",".join(str(int(v)) for v in color.tolist())
        selector = np.all(mask_rgb == color, axis=-1)
        ids, counts = np.unique(int_map[selector], return_counts=True)
        overlap[color_key] = {str(int(i)): int(c) for i, c in zip(ids.tolist(), counts.tolist())}
    return overlap


def _best_id_map(cooccurrence: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for color_key, counts in cooccurrence.items():
        total = sum(counts.values())
        best_id, best_count = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0]
        result[color_key] = {
            "best_id": int(best_id),
            "confidence": best_count / total if total else 0.0,
            "palette_name": PALETTE_NAMES.get(color_key, "unknown"),
            "distribution": {k: round(v / total, 6) for k, v in counts.items()} if total else {},
        }
    return result


def run_audit(
    raw_root: Path,
    reports_dir: Path,
    configs_dir: Path,
    traversable_palette_names: list[str] | None = None,
    mapping_filename: str = "discovered_mapping.yaml",
) -> dict[str, Any]:
    traversable_palette_names = traversable_palette_names or ["blue"]
    dataset_summary: dict[str, Any] = {}
    all_validated_samples: list[dict[str, Any]] = []
    global_raw_ids: Counter[int] = Counter()
    global_colors: Counter[str] = Counter()
    global_cooccurrence: dict[str, Counter[int]] = defaultdict(Counter)

    for dataset in DATASET_NAMES:
        dataset_root = raw_root / dataset
        dataset_entry: dict[str, Any] = {"splits": {}}
        for split in SPLITS:
            split_root = dataset_root / split
            imgs_dir = split_root / "imgs"
            masks_dir = split_root / "masks"
            annos_dir = split_root / "annos"
            int_maps_dir = annos_dir / "int_maps"

            img_files = sorted(p for p in imgs_dir.iterdir() if p.is_file())
            mask_files = sorted(p for p in masks_dir.iterdir() if p.is_file())
            anno_files = sorted(p for p in annos_dir.iterdir() if p.is_file())
            int_map_files = sorted(p for p in int_maps_dir.iterdir() if p.is_file())

            img_by_id = {parse_numeric_id(p): p for p in img_files}
            mask_by_id = {parse_numeric_id(p): p for p in mask_files}
            anno_by_id = {parse_numeric_id(p): p for p in anno_files}
            int_map_candidates: dict[str, list[Path]] = defaultdict(list)
            for path in int_map_files:
                int_map_candidates[parse_numeric_id(path)].append(path)

            common_ids = sorted(set(img_by_id) & set(mask_by_id))
            sample_records: list[SampleRecord] = []
            resolutions = Counter()
            mask_colors = Counter()
            int_values = Counter()
            corrupt_files: list[str] = []
            duplicate_int_ids = sorted(k for k, v in int_map_candidates.items() if len(v) > 1)

            for index, sample_id in enumerate(common_ids):
                image_path = img_by_id[sample_id]
                mask_path = mask_by_id[sample_id]
                anno_path = anno_by_id.get(sample_id)
                candidate_int_maps = int_map_candidates.get(sample_id, [])
                int_map_path = candidate_int_maps[0] if len(candidate_int_maps) == 1 else None

                try:
                    image_size = pil_size(image_path)
                    mask_size = pil_size(mask_path)
                except Exception:
                    corrupt_files.extend([relative(image_path, raw_root), relative(mask_path, raw_root)])
                    continue

                int_map_size = None
                exact_int_pair = False
                status = "palette_only"
                if int_map_path is not None:
                    try:
                        int_map_size = pil_size(int_map_path)
                        exact_int_pair = int_map_size == mask_size
                        status = "validated_int_map" if exact_int_pair else "int_map_size_mismatch"
                    except Exception:
                        corrupt_files.append(relative(int_map_path, raw_root))
                        int_map_path = None

                sample_records.append(
                    SampleRecord(
                        dataset=dataset,
                        split=split,
                        sample_id=sample_id,
                        image_path=relative(image_path, raw_root),
                        mask_path=relative(mask_path, raw_root),
                        anno_path=relative(anno_path, raw_root) if anno_path else None,
                        int_map_path=relative(int_map_path, raw_root) if int_map_path else None,
                        image_size=image_size,
                        mask_size=mask_size,
                        int_map_size=int_map_size,
                        exact_int_pair=exact_int_pair,
                        status=status,
                    )
                )

                resolutions[image_size] += 1

                if index >= PIXEL_INSPECTION_LIMIT or not exact_int_pair or int_map_path is None:
                    continue

                try:
                    mask_rgb = load_rgb(mask_path)
                    unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
                    for color in unique_colors.tolist():
                        color_key = ",".join(str(int(v)) for v in color)
                        mask_colors[color_key] += 1
                        global_colors[color_key] += 1
                except Exception:
                    corrupt_files.append(relative(mask_path, raw_root))
                    continue

                try:
                    int_map = load_gray(int_map_path)
                    for value in np.unique(int_map).tolist():
                        int_values[int(value)] += 1
                        global_raw_ids[int(value)] += 1
                    cooccurrence = _cooccurrence(mask_rgb, int_map)
                    for color_key, counts in cooccurrence.items():
                        for raw_id, count in counts.items():
                            global_cooccurrence[color_key][int(raw_id)] += count
                    all_validated_samples.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "sample_id": sample_id,
                            "image_path": relative(image_path, raw_root),
                            "mask_path": relative(mask_path, raw_root),
                            "int_map_path": relative(int_map_path, raw_root),
                            "cooccurrence": cooccurrence,
                        }
                    )
                except Exception:
                    corrupt_files.append(relative(int_map_path, raw_root))

            dataset_entry["splits"][split] = {
                "counts": {
                    "imgs": len(img_files),
                    "masks": len(mask_files),
                    "annos_files": len(anno_files),
                    "int_maps": len(int_map_files),
                    "paired_image_mask": len(common_ids),
                    "validated_int_map_pairs": sum(1 for s in sample_records if s.exact_int_pair),
                    "palette_only_pairs": sum(1 for s in sample_records if s.status != "validated_int_map"),
                },
                "extensions": {
                    "imgs": sorted({p.suffix.lower() for p in img_files}),
                    "masks": sorted({p.suffix.lower() for p in mask_files}),
                    "annos": sorted({p.suffix.lower() for p in anno_files}),
                    "int_maps": sorted({p.suffix.lower() for p in int_map_files}),
                },
                "image_resolutions": {f"{w}x{h}": c for (w, h), c in sorted(resolutions.items())},
                "unique_mask_colors": sorted(mask_colors),
                "unique_int_map_values": sorted(int_values),
                "pixel_inspection_limit": PIXEL_INSPECTION_LIMIT,
                "duplicate_int_map_numeric_ids": duplicate_int_ids[:50],
                "corrupt_files": sorted(set(corrupt_files)),
                "examples": [record.__dict__ for record in sample_records[:5]],
            }
        dataset_summary[dataset] = dataset_entry

    cooccurrence_summary = {
        color_key: {str(raw_id): int(count) for raw_id, count in sorted(counter.items())}
        for color_key, counter in global_cooccurrence.items()
    }
    inferred_mapping = _best_id_map(cooccurrence_summary)

    traversable_ids = {
        details["best_id"]
        for details in inferred_mapping.values()
        if details["palette_name"] in traversable_palette_names
    }

    binary_id_mapping = {str(raw_id): int(raw_id in traversable_ids) for raw_id in sorted(global_raw_ids)}
    color_binary_mapping = {
        color_key: int(details["palette_name"] in traversable_palette_names)
        for color_key, details in inferred_mapping.items()
    }

    payload = {
        "raw_root": str(raw_root),
        "dataset_summary": dataset_summary,
        "validated_sample_count": len(all_validated_samples),
        "validated_samples_examples": all_validated_samples[:10],
        "global_unique_int_map_ids": sorted(global_raw_ids),
        "global_unique_mask_colors": sorted(global_colors),
        "global_color_to_id_cooccurrence": cooccurrence_summary,
        "inferred_palette_id_mapping": inferred_mapping,
        "traversable_palette_names": traversable_palette_names,
        "traversable_raw_ids": sorted(traversable_ids),
        "binary_id_mapping": binary_id_mapping,
        "binary_color_mapping": color_binary_mapping,
        "chosen_supervision_source": "int_maps when exact image/mask alignment is validated; palette masks otherwise",
        "risks": [
            "The raw pairing key is the full sample suffix after the file prefix, not the numeric basename alone.",
            "Resolutions vary across subsets (1024x644, 960x604, 720x452, 1920x1208), so resizing policy must stay label-safe.",
            "A numeric-only matcher would collide on keys like 1 vs pln_1 and silently mispair samples.",
        ],
    }

    ensure_dir(reports_dir)
    ensure_dir(configs_dir)
    save_json(reports_dir / "dataset_audit.json", payload)
    save_yaml(configs_dir / mapping_filename, payload)

    lines = [
        "# Dataset Audit",
        "",
        f"- Raw root: `{raw_root}`",
        f"- Chosen supervision source: {payload['chosen_supervision_source']}",
        f"- Traversable palette classes: {payload['traversable_palette_names']}",
        f"- Traversable raw IDs: {payload['traversable_raw_ids']}",
        f"- Pixel-inspected exact int_map pairs: {len(all_validated_samples)}",
        f"- Global raw class IDs: {payload['global_unique_int_map_ids']}",
        "",
        "## Inferred Palette to Raw ID Mapping",
        "",
    ]
    for color_key, details in inferred_mapping.items():
        lines.append(
            f"- `{color_key}` ({details['palette_name']}): raw id `{details['best_id']}`, confidence `{details['confidence']:.4f}`"
        )
    lines.extend(["", "## Risks", ""])
    for risk in payload["risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", "## Split Summary", ""])
    for dataset, entry in dataset_summary.items():
        lines.append(f"### {dataset}")
        for split, split_entry in entry["splits"].items():
            counts = split_entry["counts"]
            lines.append(
                f"- {split}: imgs={counts['imgs']}, masks={counts['masks']}, int_maps={counts['int_maps']}, "
                f"validated_int_map_pairs={counts['validated_int_map_pairs']}, palette_only_pairs={counts['palette_only_pairs']}"
            )
    (reports_dir / "dataset_audit.md").write_text("\n".join(lines) + "\n")

    return payload
