from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).square().mean(1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * normalized + self.bias[:, None, None]


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(attended.transpose(1, 2).reshape(batch, tokens, channels))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class EfficientSamViTS(nn.Module):
    """EfficientSAM ViT-S image encoder used by ROD."""

    def __init__(self, image_size: int = 1024, patch_size: int = 16, dim: int = 384) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError(f"ROD encoder image size {image_size} must be divisible by patch size {patch_size}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(patch_size, dim)
        # EfficientSAM was pretrained at 224x224 and interpolates this 14x14 grid.
        self.pos_embed = nn.Parameter(torch.zeros(1, 14 * 14 + 1, dim))
        self.blocks = nn.ModuleList(TransformerBlock(dim, num_heads=6) for _ in range(12))
        self.neck = nn.Sequential(
            nn.Conv2d(dim, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

    def _position_embedding(self, height: int, width: int) -> torch.Tensor:
        positions = self.pos_embed[:, 1:]
        side = math.isqrt(positions.shape[1])
        positions = positions.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        if (side, side) != (height, width):
            positions = F.interpolate(positions, size=(height, width), mode="bicubic", align_corners=False)
        return positions.permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        x = x + self._position_embedding(x.shape[1], x.shape[2])
        height, width = x.shape[1:3]
        x = x.reshape(x.shape[0], height * width, x.shape[-1])
        latent_features = []
        for block in self.blocks:
            x = block(x)
            latent_features.append(x.reshape(x.shape[0], height, width, x.shape[-1]))
        image_embedding = self.neck(latent_features[-1].permute(0, 3, 1, 2))
        return image_embedding, latent_features


class ConvModule(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UpsampleResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            ConvModule(channels, channels, 3, padding=1),
            ConvModule(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return x + self.convs(x)


class RodDecoder(nn.Module):
    """Released ROD decoder: blocks 2-12, sequential fusion, and multiscale head."""

    def __init__(self, num_classes: int = 2, encoder_dim: int = 384) -> None:
        super().__init__()
        self.selected_indices = tuple(range(1, 12))
        self.projections = nn.ModuleList(
            nn.Sequential(ConvModule(encoder_dim, 128, 1), ConvModule(128, 128, 3, padding=1))
            for _ in self.selected_indices
        )
        self.fusion_layers = nn.ModuleList(ConvModule(128, 128, 3, padding=1) for _ in self.selected_indices)
        self.post_fusion = nn.Sequential(ConvModule(128, 128, 3, padding=1), ConvModule(128, 128, 3, padding=1))
        self.expand = ConvModule(128, 256, 1)
        self.upsample1 = UpsampleResidual(256)
        self.upsample2 = UpsampleResidual(256)
        self.fuse = ConvModule(256 * 4, 256, 1)
        self.prediction = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, inputs: tuple[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
        image_embedding, latent_features = inputs
        projected = [
            layer(latent_features[index].permute(0, 3, 1, 2))
            for index, layer in zip(self.selected_indices, self.projections)
        ]
        fused: torch.Tensor | None = None
        for feature, layer in zip(projected, self.fusion_layers):
            if fused is not None:
                feature = feature + fused
            fused = feature + layer(feature)
        if fused is None:
            raise RuntimeError("ROD decoder received no encoder features")
        feature0 = self.expand(fused + self.post_fusion(fused))
        feature1 = self.upsample1(feature0)
        feature2 = self.upsample2(feature1)
        output_size = feature2.shape[-2:]
        multiscale = [feature2, feature1, feature0, image_embedding]
        multiscale = [F.interpolate(x, size=output_size, mode="bilinear", align_corners=False) for x in multiscale]
        return self.prediction(self.fuse(torch.cat(multiscale, dim=1)))


class RODSegNet(nn.Module):
    def __init__(self, num_classes: int = 2, model_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = model_config or {}
        encoder_size = int(config.get("rod_encoder_image_size", 1024))
        self.encoder = EfficientSamViTS(image_size=encoder_size)
        self.decoder = RodDecoder(num_classes=num_classes)
        self._load_encoder_checkpoint(Path(config.get("rod_encoder_checkpoint", "weights/efficient_sam_vits.pt")))
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def _load_encoder_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"ROD requires the pretrained EfficientSAM ViT-S checkpoint at {checkpoint_path}. "
                "Run scripts/download_efficient_sam_vits.sh or set training.rod_encoder_checkpoint."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        prefix = "image_encoder."
        encoder_state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        if not encoder_state:
            raise ValueError(f"No {prefix} weights found in {checkpoint_path}")
        self.encoder.load_state_dict(encoder_state, strict=True)

    def train(self, mode: bool = True) -> RODSegNet:
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        encoder_input = F.interpolate(
            x, size=(self.encoder.image_size, self.encoder.image_size), mode="bilinear", align_corners=False
        )
        mean = encoder_input.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = encoder_input.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with torch.no_grad():
            encoder_features = self.encoder((encoder_input - mean) / std)
        logits = self.decoder(encoder_features)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
