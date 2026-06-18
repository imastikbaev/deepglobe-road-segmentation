"""Classical U-Net for binary road segmentation."""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Four-level U-Net. Output is one-channel logits (no sigmoid in the model)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        features = [base_channels * (2**i) for i in range(4)]
        self.encoders = nn.ModuleList()
        current = in_channels
        for feature in features:
            self.encoders.append(DoubleConv(current, feature))
            current = feature

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = features[-1] * 2
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(current, feature, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(feature * 2, feature))
            current = feature
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([skip, x], dim=1))
        return self.head(x)


def build_model(config: dict) -> UNet:
    model_cfg = config.get("model", {})
    return UNet(
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=1,
        base_channels=model_cfg.get("base_channels", 32),
    )
