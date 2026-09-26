"""One-mask sticker segmenter conditioned on an instance box and center."""

from __future__ import annotations

import torch
from torch import nn

from model import StickerBoundaryNet, ConvBlock


class PromptedStickerNet(StickerBoundaryNet):
    """Input: RGB, box interior, center heatmap. Output: one outer-rim mask."""

    def __init__(self, width: int = 16):
        super().__init__(width)
        self.enc0 = ConvBlock(5, width)
        self.output = nn.Conv2d(width, 1, 1)

    def initialize_from_boundary_model(self, path: str) -> None:
        source = torch.load(path, map_location="cpu", weights_only=True)
        target = self.state_dict()
        for name in target:
            if name == "enc0.0.weight":
                target[name][:, :3] = source[name]
                target[name][:, 3:] = 0
            elif name.startswith("output."):
                target[name] = source[name][:1].clone()
            else:
                target[name] = source[name]
        self.load_state_dict(target)
