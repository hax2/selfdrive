from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from .model import build_model
from .utils import ensure_dir
from .visualize import blend, mask_to_color
from .metrics import predict_from_logits


def run_inference(config: dict, input_dir: Path, output_dir: Path, checkpoint_name: str = "best.pt") -> dict:
    checkpoint_path = Path(config["outputs_dir"]) / config["experiment_name"] / "checkpoints" / checkpoint_name
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
    output_dir = ensure_dir(output_dir)
    count = 0
    total_forward_seconds = 0.0
    total_wall_start = perf_counter()
    traversable_threshold = float(config.get("postprocessing", {}).get("traversable_threshold", 0.5))
    with torch.no_grad():
        for path in sorted(input_dir.iterdir()):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            with Image.open(path) as opened:
                original = opened.convert("RGB")
            resized = original.resize((width, height), resample=Image.Resampling.BILINEAR)
            tensor = torch.from_numpy(np.array(resized).astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_start = perf_counter()
            logits = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_forward_seconds += perf_counter() - forward_start
            pred = predict_from_logits(logits, traversable_threshold=traversable_threshold)[0].cpu().numpy().astype(np.uint8)
            pred_img = Image.fromarray(pred, mode="L").resize(original.size, resample=Image.Resampling.NEAREST)
            pred_array = np.array(pred_img)
            mask_image = Image.fromarray(mask_to_color(pred_array))
            mask_image.save(output_dir / f"{path.stem}_mask.png")
            mask_image.close()
            overlay_image = Image.fromarray(blend(np.array(original), pred_array))
            overlay_image.save(output_dir / f"{path.stem}_overlay.png")
            overlay_image.close()
            pred_img.close()
            resized.close()
            original.close()
            count += 1
    total_wall_seconds = perf_counter() - total_wall_start
    avg_forward_ms = total_forward_seconds / max(count, 1) * 1000.0
    avg_wall_ms = total_wall_seconds / max(count, 1) * 1000.0
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "images_processed": count,
        "traversable_threshold": traversable_threshold,
        "inference_seconds": total_forward_seconds,
        "inference_fps": count / total_forward_seconds if total_forward_seconds > 0 else 0.0,
        "average_inference_latency_ms": avg_forward_ms,
        "wall_seconds": total_wall_seconds,
        "wall_fps": count / total_wall_seconds if total_wall_seconds > 0 else 0.0,
        "average_wall_latency_ms": avg_wall_ms,
    }
