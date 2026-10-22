"""Backbones: the CSP-Star backbone with SPPF and attention at P5 (and optionally P4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import RepConv
from .blocks import CSPStar, SPPF
from .attention import PSA


def _stage(channels, depth):
    return CSPStar(channels, channels, depth) if int(depth) > 0 else nn.Identity()


class Backbone(nn.Module):
    """Stem, P2, three CSP-Star stages, SPPF, attention. Returns P3, P4, P5."""

    def __init__(self, stem_channels, p2_channels, widths, depths, attn_p5=1, attn_p4=0):
        super().__init__()
        c3, c4, c5 = widths
        d2, d3, d4, d5 = depths
        self.stem = RepConv(3, stem_channels, 3, 2)
        self.p2 = RepConv(stem_channels, p2_channels, 3, 2)
        self.stage2 = _stage(p2_channels, d2)
        self.down3 = RepConv(p2_channels, c3, 3, 2)
        self.stage3 = _stage(c3, d3)
        self.down4 = RepConv(c3, c4, 3, 2)
        self.stage4 = _stage(c4, d4)
        self.attn4 = PSA(c4, attn_p4) if int(attn_p4) > 0 else nn.Identity()
        self.down5 = RepConv(c4, c5, 3, 2)
        self.stage5 = _stage(c5, d5)
        self.sppf = SPPF(c5, c5)
        self.attn5 = PSA(c5, attn_p5) if int(attn_p5) > 0 else nn.Identity()
        self.out_channels = (c3, c4, c5)

    def forward(self, x: torch.Tensor):
        p3 = self.stage3(self.down3(self.stage2(self.p2(self.stem(x)))))
        p4 = self.attn4(self.stage4(self.down4(p3)))
        p5 = self.attn5(self.sppf(self.stage5(self.down5(p4))))
        return p3, p4, p5
