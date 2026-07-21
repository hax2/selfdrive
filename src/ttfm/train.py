from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


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
        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"Non-finite {'training' if training else 'validation'} loss "
                f"at batch {batch_index}: {loss.detach().cpu().item()}"
            )
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
    save_last_checkpoint = bool(config["training"].get("save_last_checkpoint", True))
    save_optimizer_state = bool(config["training"].get("save_optimizer_state", True))

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
    cosine_decay_epochs: int | None = None
    if scheduler_steps_per_batch:
        optimizer_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
        total_optimizer_steps = max(optimizer_steps_per_epoch * epochs, 1)
        poly_power = float(config["training"].get("poly_power", 0.9))
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1.0 - min(step, total_optimizer_steps) / total_optimizer_steps) ** poly_power,
        )
    elif scheduler_name == "cosine":
        cosine_decay_epochs = int(config["training"].get("cosine_decay_epochs", epochs))
        if cosine_decay_epochs < 1:
            raise ValueError("training.cosine_decay_epochs must be at least 1")
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_decay_epochs,
            eta_min=float(config["training"].get("min_learning_rate", 1e-5)),
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    best_metric = -1.0
    history: list[dict[str, Any]] = []
    early_stopping_cfg = config["training"].get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False))
    early_stopping_metric = str(early_stopping_cfg.get("metric", "mIoU"))
    early_stopping_mode = str(early_stopping_cfg.get("mode", "max"))
    early_stopping_patience = int(early_stopping_cfg.get("patience", 8))
    early_stopping_min_delta = float(early_stopping_cfg.get("min_delta", 0.0))
    early_stopping_min_epochs = int(early_stopping_cfg.get("min_epochs", 1))
    if early_stopping_mode not in {"min", "max"}:
        raise ValueError("training.early_stopping.mode must be 'min' or 'max'")
    if early_stopping_patience < 1:
        raise ValueError("training.early_stopping.patience must be at least 1")
    if early_stopping_min_epochs < 1:
        raise ValueError("training.early_stopping.min_epochs must be at least 1")
    if early_stopping_min_delta < 0:
        raise ValueError("training.early_stopping.min_delta must be non-negative")
    early_stopping_best = math.inf if early_stopping_mode == "min" else -math.inf
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason: str | None = None
    best_epoch: int | None = None
    run_started_at = _utc_now()
    run_start = perf_counter()
    progress_path = output_root / "training_progress.json"
    save_json(
        progress_path,
        {
            "status": "running",
            "experiment_name": config["experiment_name"],
            "started_at": run_started_at,
            "current_epoch": 0,
            "maximum_epochs": epochs,
            "cosine_decay_epochs": cosine_decay_epochs,
        },
    )

    for epoch in range(1, epochs + 1):
        epoch_start = perf_counter()
        train_dataset.set_epoch(epoch)
        train_start = perf_counter()
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
        train_seconds = perf_counter() - train_start
        if device.type == "cuda":
            torch.cuda.empty_cache()
        validation_start = perf_counter()
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
        validation_seconds = perf_counter() - validation_start
        if early_stopping_metric == "loss":
            monitored_value = float(val_loss)
        else:
            if early_stopping_metric not in val_metrics:
                available = ", ".join(sorted(val_metrics))
                raise ValueError(
                    f"Unknown early-stopping metric '{early_stopping_metric}'. "
                    f"Use 'loss' or one of: {available}"
                )
            monitored_value = float(val_metrics[early_stopping_metric])
        if not math.isfinite(monitored_value):
            raise FloatingPointError(
                f"Non-finite validation metric '{early_stopping_metric}' at epoch {epoch}: {monitored_value}"
            )
        if not scheduler_steps_per_batch and (
            scheduler_name != "cosine" or cosine_decay_epochs is None or epoch <= cosine_decay_epochs
        ):
            scheduler.step()

        checkpoint = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "config": config,
            "class_weights": class_weights.cpu(),
        }
        if save_optimizer_state:
            checkpoint["optimizer_state"] = optimizer.state_dict()
        if save_last_checkpoint:
            torch.save(checkpoint, checkpoints_dir / "last.pt")

        if val_metrics["mIoU"] > best_metric:
            best_metric = float(val_metrics["mIoU"])
            best_epoch = epoch
            torch.save(checkpoint, checkpoints_dir / "best.pt")
            for example in examples[:4]:
                save_triptych(example["image"], example["gt"], example["pred"], viz_dir / f"val_best_{example['name']}.png")

        should_stop = False
        if early_stopping_enabled:
            if early_stopping_mode == "max":
                meaningful_improvement = monitored_value > early_stopping_best + early_stopping_min_delta
            else:
                meaningful_improvement = monitored_value < early_stopping_best - early_stopping_min_delta
            if meaningful_improvement:
                early_stopping_best = monitored_value
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epoch >= early_stopping_min_epochs and epochs_without_improvement >= early_stopping_patience:
                stopped_early = True
                should_stop = True
                stop_reason = (
                    f"validation {early_stopping_metric} did not improve by at least "
                    f"{early_stopping_min_delta:g} for {early_stopping_patience} epochs"
                )
        epoch_seconds = perf_counter() - epoch_start
        recent_epoch_seconds = [
            float(item.get("timing", {}).get("epoch_seconds", epoch_seconds)) for item in history[-4:]
        ] + [epoch_seconds]
        rolling_epoch_seconds = sum(recent_epoch_seconds) / len(recent_epoch_seconds)
        eta_to_max_seconds = rolling_epoch_seconds * max(epochs - epoch, 0)
        eta_to_early_stop_seconds: float | None = None
        if early_stopping_enabled:
            epochs_until_minimum = max(early_stopping_min_epochs - epoch, 0)
            epochs_until_patience = max(early_stopping_patience - epochs_without_improvement, 0)
            planned_remaining_epochs = min(
                max(epochs_until_minimum, epochs_until_patience),
                max(epochs - epoch, 0),
            )
            eta_to_early_stop_seconds = rolling_epoch_seconds * planned_remaining_epochs

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "timing": {
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "epoch_seconds": epoch_seconds,
                "rolling_epoch_seconds": rolling_epoch_seconds,
                "elapsed_seconds": perf_counter() - run_start,
                "eta_to_early_stop_seconds": eta_to_early_stop_seconds,
                "eta_to_max_seconds": eta_to_max_seconds,
            },
        }
        history.append(epoch_record)
        save_json(output_root / "history.json", history)
        save_json(
            progress_path,
            {
                "status": "stopping" if should_stop else "running",
                "experiment_name": config["experiment_name"],
                "started_at": run_started_at,
                "updated_at": _utc_now(),
                "current_epoch": epoch,
                "maximum_epochs": epochs,
                "cosine_decay_epochs": cosine_decay_epochs,
                "best_epoch": best_epoch,
                "best_val_mIoU": best_metric,
                "current_val_mIoU": float(val_metrics["mIoU"]),
                "current_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epochs_without_improvement": epochs_without_improvement,
                "early_stopping_patience": early_stopping_patience if early_stopping_enabled else None,
                **epoch_record["timing"],
            },
        )
        print(
            f"epoch={epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mIoU={val_metrics['mIoU']:.4f} best={best_metric:.4f}@{best_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.3g} epoch_s={epoch_seconds:.1f} "
            f"rolling_s={rolling_epoch_seconds:.1f} bad_epochs={epochs_without_improvement} "
            f"eta_stop={_format_duration(eta_to_early_stop_seconds)} "
            f"eta_cap={_format_duration(eta_to_max_seconds)}",
            flush=True,
        )
        if should_stop:
            print(f"early_stop epoch={epoch}: {stop_reason}", flush=True)
            break

    total_training_seconds = perf_counter() - run_start
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
        "save_last_checkpoint": save_last_checkpoint,
        "save_optimizer_state": save_optimizer_state,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "val_batch_size": val_batch_size,
        "lr_scheduler": scheduler_name,
        "cosine_decay_epochs": cosine_decay_epochs,
        "best_val_mIoU": best_metric,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "maximum_epochs": epochs,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "started_at": run_started_at,
        "finished_at": _utc_now(),
        "total_training_seconds": total_training_seconds,
        "average_epoch_seconds": total_training_seconds / max(len(history), 1),
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "metric": early_stopping_metric,
            "mode": early_stopping_mode,
            "patience": early_stopping_patience,
            "min_delta": early_stopping_min_delta,
            "min_epochs": early_stopping_min_epochs,
            "best_monitored_value": early_stopping_best if early_stopping_enabled else None,
        },
        "history": history,
    }
    save_json(output_root / "train_summary.json", summary)
    save_json(
        progress_path,
        {
            "status": "completed",
            "experiment_name": config["experiment_name"],
            "started_at": run_started_at,
            "finished_at": summary["finished_at"],
            "current_epoch": len(history),
            "maximum_epochs": epochs,
            "cosine_decay_epochs": cosine_decay_epochs,
            "best_epoch": best_epoch,
            "best_val_mIoU": best_metric,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "total_training_seconds": total_training_seconds,
            "average_epoch_seconds": summary["average_epoch_seconds"],
            "eta_to_early_stop_seconds": 0.0,
            "eta_to_max_seconds": 0.0,
        },
    )
    return summary
