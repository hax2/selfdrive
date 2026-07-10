from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ttfm.metrics import predict_from_logits
from ttfm.model import build_model
from ttfm.utils import ensure_dir, load_yaml
from ttfm.visualize import blend, mask_to_color, save_mask, save_overlay


def choose_samples(files: list[Path], sample_count: int) -> list[Path]:
    if sample_count >= len(files):
        return files
    if sample_count <= 1:
        return [files[len(files) // 2]]
    step = (len(files) - 1) / (sample_count - 1)
    indices = sorted({min(len(files) - 1, round(i * step)) for i in range(sample_count)})
    return [files[index] for index in indices]


def predict_mask(
    model: torch.nn.Module,
    image_path: Path,
    input_size: tuple[int, int],
    device: torch.device,
    traversable_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as opened:
        original = opened.convert("RGB")
    original_array = np.array(original)
    resized = original.resize(input_size, resample=Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.array(resized).astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred = predict_from_logits(logits, traversable_threshold=traversable_threshold)[0].cpu().numpy().astype(np.uint8)
    pred_image = Image.fromarray(pred, mode="L").resize(original.size, resample=Image.Resampling.NEAREST)
    pred_array = np.array(pred_image)
    pred_image.close()
    resized.close()
    original.close()
    return original_array, pred_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sampled preview inference on the alice folder.")
    parser.add_argument("--config", default="configs/conservative.yaml")
    parser.add_argument("--input-dir", default="alice")
    parser.add_argument("--output-dir", default="outputs/alice_preview")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--checkpoint", default="best.pt")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    checkpoint_path = Path(config["outputs_dir"]) / config["experiment_name"] / "checkpoints" / args.checkpoint
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    triptychs_dir = ensure_dir(output_dir / "triptychs")
    overlays_dir = ensure_dir(output_dir / "overlays")
    masks_dir = ensure_dir(output_dir / "masks")

    files = sorted([path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    chosen = choose_samples(files, args.sample_count)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=2,
        model_name=config["training"].get("model_name", "pidnet-s"),
        model_config=config["training"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    width, height = config["training"]["input_size"]
    traversable_threshold = float(config.get("postprocessing", {}).get("traversable_threshold", 0.5))
    for image_path in chosen:
        original, pred = predict_mask(model, image_path, (width, height), device, traversable_threshold)
        stem = image_path.stem
        save_mask(pred, masks_dir / f"{stem}.png")
        save_overlay(original, pred, overlays_dir / f"{stem}.png")
        preview = np.concatenate([original, blend(original, pred), mask_to_color(pred)], axis=1)
        Image.fromarray(preview).save(triptychs_dir / f"{stem}.png")

    print(f"Processed {len(chosen)} sampled images from {input_dir}")
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
