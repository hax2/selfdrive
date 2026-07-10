from __future__ import annotations

import torch
from torch import nn

from .vendor_pidnet import get_pred_model


class PIDNetSegNet(nn.Module):
    def __init__(self, num_classes: int = 2, variant: str = "pidnet-s") -> None:
        super().__init__()
        self.variant = variant
        self.backbone = get_pred_model(name=variant, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        return torch.nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(
    num_classes: int = 2,
    model_name: str = "pidnet-s",
    model_config: dict | None = None,
) -> nn.Module:
    if model_name == "pidnet-s":
        return PIDNetSegNet(num_classes=num_classes, variant=model_name)
    if model_name == "rod-vits":
        from .rod import RODSegNet

        return RODSegNet(num_classes=num_classes, model_config=model_config)
    raise ValueError(f"Unsupported model_name: {model_name}")
