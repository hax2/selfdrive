from __future__ import annotations

import torch


def predict_from_logits(logits: torch.Tensor, traversable_threshold: float = 0.5) -> torch.Tensor:
    if logits.shape[1] != 2:
        return logits.argmax(dim=1)
    probs = torch.softmax(logits, dim=1)[:, 1]
    return (probs >= traversable_threshold).to(torch.int64)


def update_confusion(confusion: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, traversable_threshold: float = 0.5) -> torch.Tensor:
    preds = predict_from_logits(logits, traversable_threshold=traversable_threshold).view(-1)
    targets = targets.view(-1)
    num_classes = confusion.shape[0]
    indices = targets * num_classes + preds
    bins = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion += bins.reshape(num_classes, num_classes)
    return confusion


def metrics_from_confusion(confusion: torch.Tensor) -> dict:
    raw_confusion = confusion.to(torch.int64)
    confusion = confusion.double()
    tp = confusion.diag()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    precision = tp[1] / max((tp[1] + fp[1]).item(), 1.0)
    recall = tp[1] / max((tp[1] + fn[1]).item(), 1.0)
    f1 = 2 * precision * recall / max((precision + recall).item(), 1e-8)
    tn = confusion[0, 0]
    false_safe = confusion[0, 1] / max((tn + confusion[0, 1]).item(), 1.0)
    false_block = confusion[1, 0] / max((tp[1] + confusion[1, 0]).item(), 1.0)
    traversable_counts = {
        "tp": int(raw_confusion[1, 1].item()),
        "fp": int(raw_confusion[0, 1].item()),
        "tn": int(raw_confusion[0, 0].item()),
        "fn": int(raw_confusion[1, 0].item()),
    }
    return {
        "mIoU": float(iou.mean().item()),
        "iou_untraversable": float(iou[0].item()),
        "iou_traversable": float(iou[1].item()),
        "precision_traversable": float(precision.item()),
        "recall_traversable": float(recall.item()),
        "f1_traversable": float(f1.item()),
        "false_safe_rate": float(false_safe.item()),
        "false_block_rate": float(false_block.item()),
        "fbr_traversable": float(false_block.item()),
        "confusion_counts_traversable": traversable_counts,
        "confusion_count_definition": "For traversable as positive: TP=gt1/pred1, FP=gt0/pred1, TN=gt0/pred0, FN=gt1/pred0; FBR=FN/(FN+TP).",
        "confusion_matrix": raw_confusion.tolist(),
    }
