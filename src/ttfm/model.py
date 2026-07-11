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


class FullResolutionSegNet(nn.Module):
    """Normalize third-party segmentation outputs to one full-resolution logits tensor."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
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
    if model_name == "ddrnet-23-slim":
        from .vendor_ddrnet import BasicBlock, DualResNet

        return FullResolutionSegNet(
            DualResNet(
                BasicBlock,
                [2, 2, 2, 2],
                num_classes=num_classes,
                planes=32,
                spp_planes=128,
                head_planes=64,
                augment=False,
            )
        )
    if model_name == "bisenetv2":
        from .vendor_bisenetv2 import BiSeNetV2

        pretrained = bool((model_config or {}).get("encoder_weights", "imagenet"))
        return FullResolutionSegNet(BiSeNetV2(n_classes=num_classes, aux_mode="eval", pretrained=pretrained))
    if model_name.startswith("smp:"):
        import segmentation_models_pytorch as smp

        # Expected format: smp:arch:encoder_name, e.g. smp:FPN:mobilenet_v3_small
        parts = model_name.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid smp model name format: {model_name}. Expected smp:arch:encoder.")

        arch = parts[1]
        encoder = parts[2]

        return smp.create_model(
            arch=arch,
            encoder_name=encoder,
            encoder_weights=(model_config or {}).get("encoder_weights", "imagenet"),
            in_channels=3,
            classes=num_classes,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")
