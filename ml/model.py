"""Small, fixed-input foreground and instance-boundary model."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, source: int, target: int):
        super().__init__(
            nn.Conv2d(source, target, 3, padding=1, bias=False),
            nn.BatchNorm2d(target),
            nn.ReLU(inplace=True),
            nn.Conv2d(target, target, 3, padding=1, bias=False),
            nn.BatchNorm2d(target),
            nn.ReLU(inplace=True),
        )


class StickerBoundaryNet(nn.Module):
    """Channel 0: foreground; channel 1: visible instance boundary."""

    def __init__(self, width: int = 16):
        super().__init__()
        self.enc0 = ConvBlock(3, width)
        self.enc1 = ConvBlock(width, width * 2)
        self.enc2 = ConvBlock(width * 2, width * 4)
        self.enc3 = ConvBlock(width * 4, width * 8)
        self.dec2 = ConvBlock(width * 12, width * 4)
        self.dec1 = ConvBlock(width * 6, width * 2)
        self.dec0 = ConvBlock(width * 3, width)
        self.output = nn.Conv2d(width, 2, 1)

    @staticmethod
    def up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x0 = self.enc0(image)
        x1 = self.enc1(F.max_pool2d(x0, 2))
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        y2 = self.dec2(self.up(x3, x2))
        y1 = self.dec1(self.up(y2, x1))
        y0 = self.dec0(self.up(y1, x0))
        return self.output(y0)
