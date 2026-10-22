"""Feature blocks: StarBlock, CSP-Star, and SPPF."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv, RepConv


class StarBlock(nn.Module):
    """x + W2(SiLU(W1 x) * SiLU(W3x3 x)). W2's BatchNorm gamma starts at 0."""

    def __init__(self, channels: int):
        super().__init__()
        self.p1 = Conv(channels, channels, 1)
        self.spatial = RepConv(channels, channels, 3)
        self.p2 = Conv(channels, channels, 1, act=False)
        nn.init.zeros_(self.p2.bn.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.p2(self.p1(x) * self.spatial(x))


class CSPStar(nn.Module):
    """1x1 split, n StarBlocks on half the channels (all kept), 1x1 merge."""

    def __init__(self, c_in: int, c_out: int, n: int = 1, e: float = 0.5):
        super().__init__()
        hidden = int(c_out * e)
        self.cv1 = Conv(c_in, 2 * hidden, 1)
        self.blocks = nn.ModuleList(StarBlock(hidden) for _ in range(int(n)))
        self.cv2 = Conv((2 + int(n)) * hidden, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).chunk(2, 1)
        outputs = [a, b]
        for block in self.blocks:
            b = block(b)
            outputs.append(b)
        return self.cv2(torch.cat(outputs, 1))


class SPPF(nn.Module):
    """Spatial pyramid pooling, fast."""

    def __init__(self, c1: int, c2: int, kernel_size: int = 5):
        super().__init__()
        hidden = c1 // 2
        self.reduce = Conv(c1, hidden, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        self.project = Conv(hidden * 4, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        return self.project(torch.cat((x, p1, p2, self.pool(p2)), dim=1))
