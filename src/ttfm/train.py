from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

from .data import SegmentationDataset, compute_class_weights
from .metrics import metrics_from_confusion, predict_from_logits, update_confusion
from .model import build_model
from .utils import ensure_dir, save_json, set_seed
from .visualize import save_triptych


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    targets = (targets == 1).float()
    intersection = (probs * targets).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def false_safe_penalty_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    negative_mask = (targets == 0).float()
    penalty = (probs * negative_mask).sum(dim=(1, 2)) / negative_mask.sum(dim=(1, 2)).clamp_min(1.0)
    return penalty.mean()


def _amp_dtype(device: torch.device, preferred: str) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if preferred == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    ce_loss: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    traversable_threshold: float,
    false_safe_penalty_weight: float,
    ce_loss_weight: float,
    dice_loss_weight: float,
    gradient_accumulation_steps: int = 1,
    batch_scheduler: Any | None = None,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    running_loss = 0.0
    examples: list[dict[str, Any]] = []
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
            loss = ce_loss_weight * ce_loss(logits, masks) + dice_loss_weight * dice_loss(logits, masks)
            if false_safe_penalty_weight > 0:
                loss = loss + false_safe_penalty_weight * false_safe_penalty_loss(logits, masks)
        if training:
            remainder = len(loader) % gradient_accumulation_steps
            final_group_size = remainder or gradient_accumulation_steps
            in_final_group = batch_index >= len(loader) - final_group_size
            accumulation_divisor = final_group_size if in_final_group else gradient_accumulation_steps
            (loss / accumulation_divisor).backward()
            accumulation_complete = (batch_index + 1) % gradient_accumulation_steps == 0
            if accumulation_complete or batch_index + 1 == len(loader):
                optimizer.step()
                if batch_scheduler is not None:
                    batch_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item() * images.shape[0]
        confusion = update_confusion(
            confusion,
            logits.detach().cpu(),
            masks.detach().cpu(),
            traversable_threshold=traversable_threshold,
        )
        preds = predict_from_logits(logits.detach().cpu(), traversable_threshold=traversable_threshold)
        for index in range(min(2, images.shape[0])):
            examples.append(
                {
                    "image": (images[index].detach().cpu().permute(1, 2, 0).numpy() * 255).astype("uint8"),
                    "gt": masks[index].detach().cpu().numpy().astype("uint8"),
                    "pred": preds[index].numpy().astype("uint8"),
                    "name": Path(batch["image_path"][index]).stem,
                }
            )
    average_loss = running_loss / max(len(loader.dataset), 1)
    return average_loss, metrics_from_confusion(confusion), examples[:8]


def _build_class_weights(config: dict[str, Any], processed_root: Path, device: torch.device) -> torch.Tensor:
    training_cfg = config["training"]
    if "class_weights" in training_cfg:
        return torch.tensor(training_cfg["class_weights"], dtype=torch.float32, device=device)

    class_weights = compute_class_weights(processed_root)
    safety_bias = float(training_cfg.get("untraversable_weight_bias", 1.0))
    traversable_bias = float(training_cfg.get("traversable_weight_bias", 1.0))
    class_weights[0] *= safety_bias
    class_weights[1] *= traversable_bias
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    return class_weights.to(device)


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    processed_root = Path(config["processed_root"])
    output_root = ensure_dir(Path(config["outputs_dir"]) / config["experiment_name"])
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    viz_dir = ensure_dir(output_root / "visuals")

    image_size = tuple(config["training"]["input_size"])
    batch_size = int(config["training"]["batch_size"])
    val_batch_size = int(config["training"].get("val_batch_size", batch_size))
    epochs = int(config["training"]["epochs"])
    learning_rate = float(config["training"]["learning_rate"])
    num_workers = int(config["training"].get("num_workers", 0))
    traversable_threshold = float(config.get("postprocessing", {}).get("traversable_threshold", 0.5))
    false_safe_penalty_weight = float(config["training"].get("false_safe_penalty_weight", 0.0))
    ce_loss_weight = float(config["training"].get("ce_loss_weight", 1.0))
    dice_loss_weight = float(config["training"].get("dice_loss_weight", 1.0))
    gradient_accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))

    max_train_samples = config["training"].get("max_train_samples")
    max_val_samples = config["training"].get("max_val_samples")
    train_dataset = SegmentationDataset(
        processed_root,
        "train",
        image_size,
        max_samples=max_train_samples,
        augment=bool(config["training"].get("augment", True)),
        seed=int(config["seed"]),
    )
    val_dataset = SegmentationDataset(
        processed_root,
        "val",
        image_size,
        max_samples=max_val_samples,
        augment=False,
        seed=int(config["seed"]),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = _amp_dtype(device, config["training"].get("mixed_precision", "bf16"))
    class_weights = _build_class_weights(config, processed_root, device)

    model = build_model(
        num_classes=2,
        model_name=config["training"].get("model_name", "pidnet-s"),
        model_config=config["training"],
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=float(config["training"]["weight_decay"]))
    scheduler_name = str(config["training"].get("lr_scheduler", "cosine"))
    scheduler_steps_per_batch = scheduler_name == "poly"
    if scheduler_steps_per_batch:
        optimizer_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
        total_optimizer_steps = max(optimizer_steps_per_epoch * epochs, 1)
        poly_power = float(config["training"].get("poly_power", 0.9))
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1.0 - min(step, total_optimizer_steps) / total_optimizer_steps) ** poly_power,
        )
    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(config["training"].get("min_learning_rate", 1e-5)),
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    best_metric = -1.0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        train_loss, train_metrics, _ = _run_epoch(
            model,
            train_loader,
            optimizer,
            ce_loss,
            device,
            amp_dtype,
            traversable_threshold,
            false_safe_penalty_weight,
            ce_loss_weight,
            dice_loss_weight,
            gradient_accumulation_steps,
            scheduler if scheduler_steps_per_batch else None,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        val_loss, val_metrics, examples = _run_epoch(
            model,
            val_loader,
            None,
            ce_loss,
            device,
            amp_dtype,
            traversable_threshold,
            false_safe_penalty_weight,
            ce_loss_weight,
            dice_loss_weight,
        )
        if not scheduler_steps_per_batch:
            scheduler.step()
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)
        save_json(output_root / "history.json", history)
        print(
            f"epoch={epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mIoU={val_metrics['mIoU']:.4f}",
            flush=True,
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "class_weights": class_weights.cpu(),
        }
        torch.save(checkpoint, checkpoints_dir / "last.pt")

        if val_metrics["mIoU"] > best_metric:
            best_metric = float(val_metrics["mIoU"])
            torch.save(checkpoint, checkpoints_dir / "best.pt")
            for example in examples[:4]:
                save_triptych(example["image"], example["gt"], example["pred"], viz_dir / f"val_best_{example['name']}.png")

    summary = {
        "experiment_name": config["experiment_name"],
        "device": str(device),
        "amp_dtype": str(amp_dtype) if amp_dtype else "disabled",
        "input_size": list(image_size),
        "augment": bool(config["training"].get("augment", True)),
        "class_weights": class_weights.cpu().tolist(),
        "traversable_threshold": traversable_threshold,
        "ce_loss_weight": ce_loss_weight,
        "dice_loss_weight": dice_loss_weight,
        "false_safe_penalty_weight": false_safe_penalty_weight,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "val_batch_size": val_batch_size,
        "lr_scheduler": scheduler_name,
        "best_val_mIoU": best_metric,
        "history": history,
    }
    save_json(output_root / "train_summary.json", summary)
    return summary
