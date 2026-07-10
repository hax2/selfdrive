from __future__ import annotations

from pathlib import Path

import torch

from .model import build_model
from .utils import ensure_dir


def run_export(config: dict, checkpoint_name: str = "best.pt") -> dict:
    try:
        import onnx  # noqa: F401
    except Exception as exc:
        raise RuntimeError("ONNX export requires the optional 'onnx' package.") from exc

    checkpoint_path = Path(config["outputs_dir"]) / config["experiment_name"] / "checkpoints" / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(
        num_classes=2,
        model_name=config["training"].get("model_name", "pidnet-s"),
        model_config=config["training"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    width, height = config["training"]["input_size"]
    dummy = torch.randn(1, 3, height, width)
    export_dir = ensure_dir(Path(config["outputs_dir"]) / config["experiment_name"] / "export")
    output_path = export_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return {"onnx_path": str(output_path)}
