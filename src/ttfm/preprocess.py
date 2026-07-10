from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .audit import parse_numeric_id, run_audit
from .utils import ensure_dir, load_yaml, save_json


@dataclass
class PreparedSample:
    split: str
    source_dataset: str
    sample_id: str
    image_path: str
    mask_path: str
    label_source: str
    raw_label_path: str


def _rgb_key(color: np.ndarray) -> str:
    return ",".join(str(int(v)) for v in color.tolist())


def _stable_bucket(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _numeric_lookup(paths: list[Path]) -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = {}
    for path in paths:
        lookup.setdefault(parse_numeric_id(path), []).append(path)
    return lookup


def _write_binary_mask_from_int_map(source: Path, destination: Path, binary_id_mapping: dict[str, int]) -> None:
    with Image.open(source) as image:
        array = np.array(image)
    output = np.zeros_like(array, dtype=np.uint8)
    for raw_id_str, binary_value in binary_id_mapping.items():
        output[array == int(raw_id_str)] = np.uint8(binary_value)
    out_image = Image.fromarray(output, mode="L")
    out_image.save(destination)
    out_image.close()


def _write_binary_mask_from_palette(source: Path, destination: Path, color_binary_mapping: dict[str, int]) -> None:
    with Image.open(source) as image:
        array = np.array(image.convert("RGB"))
    output = np.zeros(array.shape[:2], dtype=np.uint8)
    for color in np.unique(array.reshape(-1, 3), axis=0):
        key = _rgb_key(color)
        output[np.all(array == color, axis=-1)] = np.uint8(color_binary_mapping.get(key, 0))
    out_image = Image.fromarray(output, mode="L")
    out_image.save(destination)
    out_image.close()


def build_processed_dataset(config: dict[str, Any]) -> dict[str, Any]:
    raw_root = Path(config["raw_root"])
    processed_root = Path(config["processed_root"])
    reports_dir = Path(config["reports_dir"])
    configs_dir = Path(config["configs_dir"])
    split_source = config.get("dataset_name", "mixed")
    val_fraction = float(config["preprocessing"].get("val_fraction", 0.2))
    seed = int(config["seed"])
    traversable_palette_names = list(config["preprocessing"].get("traversable_palette_names", ["blue"]))
    mapping_filename = str(config["preprocessing"].get("mapping_filename", "discovered_mapping.yaml"))

    audit_path = reports_dir / "dataset_audit.json"
    if not audit_path.exists():
        run_audit(
            raw_root=raw_root,
            reports_dir=reports_dir,
            configs_dir=configs_dir,
            traversable_palette_names=traversable_palette_names,
            mapping_filename=mapping_filename,
        )
    mapping_path = configs_dir / mapping_filename
    if not mapping_path.exists():
        run_audit(
            raw_root=raw_root,
            reports_dir=reports_dir,
            configs_dir=configs_dir,
            traversable_palette_names=traversable_palette_names,
            mapping_filename=mapping_filename,
        )
    mapping = load_yaml(mapping_path)

    if processed_root.exists():
        shutil.rmtree(processed_root)
    for split in ["train", "val", "test"]:
        ensure_dir(processed_root / split / "images")
        ensure_dir(processed_root / split / "masks")

    dataset_root = raw_root / split_source
    train_root = dataset_root / "Train"
    test_root = dataset_root / "Test"

    def prepare_split(source_root: Path, destination_split: str, allow_validation_split: bool) -> list[PreparedSample]:
        img_files = sorted(p for p in (source_root / "imgs").iterdir() if p.is_file())
        mask_files = sorted(p for p in (source_root / "masks").iterdir() if p.is_file())
        int_map_files = sorted(p for p in (source_root / "annos" / "int_maps").iterdir() if p.is_file())

        img_lookup = _numeric_lookup(img_files)
        mask_lookup = _numeric_lookup(mask_files)
        int_lookup = _numeric_lookup(int_map_files)

        prepared: list[PreparedSample] = []
        for sample_id in sorted(set(img_lookup) & set(mask_lookup)):
            if len(img_lookup[sample_id]) != 1 or len(mask_lookup[sample_id]) != 1:
                raise RuntimeError(f"Ambiguous image/mask pairing for sample key '{sample_id}' in {source_root}")
            image_path = img_lookup[sample_id][0]
            palette_path = mask_lookup[sample_id][0]
            int_candidates = int_lookup.get(sample_id, [])
            exact_int_map = None
            if len(int_candidates) == 1:
                with Image.open(palette_path) as palette_image, Image.open(int_candidates[0]) as int_image:
                    if palette_image.size == int_image.size:
                        exact_int_map = int_candidates[0]

            target_split = destination_split
            if allow_validation_split:
                target_split = "val" if _stable_bucket(f"{split_source}:{sample_id}", seed) < val_fraction else "train"

            stem = f"{split_source}_{sample_id}"
            output_image = processed_root / target_split / "images" / f"{stem}.png"
            output_mask = processed_root / target_split / "masks" / f"{stem}.png"

            shutil.copy2(image_path, output_image)
            if exact_int_map is not None:
                _write_binary_mask_from_int_map(exact_int_map, output_mask, mapping["binary_id_mapping"])
                label_source = "int_map"
                raw_label_path = str(exact_int_map.relative_to(raw_root))
            else:
                _write_binary_mask_from_palette(palette_path, output_mask, mapping["binary_color_mapping"])
                label_source = "palette_mask_fallback"
                raw_label_path = str(palette_path.relative_to(raw_root))

            prepared.append(
                PreparedSample(
                    split=target_split,
                    source_dataset=split_source,
                    sample_id=sample_id,
                    image_path=str(output_image.relative_to(processed_root)),
                    mask_path=str(output_mask.relative_to(processed_root)),
                    label_source=label_source,
                    raw_label_path=raw_label_path,
                )
            )
        return prepared

    train_samples = prepare_split(train_root, "train", allow_validation_split=True)
    test_samples = prepare_split(test_root, "test", allow_validation_split=False)
    manifest = [sample.__dict__ for sample in train_samples + test_samples]
    save_json(processed_root / "manifest.json", manifest)

    split_counts = {
        split: sum(1 for sample in manifest if sample["split"] == split)
        for split in ["train", "val", "test"]
    }
    label_source_counts = {}
    for sample in manifest:
        label_source_counts[sample["label_source"]] = label_source_counts.get(sample["label_source"], 0) + 1

    summary = {
        "dataset_name": split_source,
        "processed_root": str(processed_root),
        "split_counts": split_counts,
        "label_source_counts": label_source_counts,
        "binary_id_mapping": mapping["binary_id_mapping"],
        "binary_color_mapping": mapping["binary_color_mapping"],
        "traversable_palette_names": mapping["traversable_palette_names"],
        "traversable_raw_ids": mapping["traversable_raw_ids"],
        "chosen_supervision_source": mapping["chosen_supervision_source"],
    }
    save_json(processed_root / "summary.json", summary)
    return summary
