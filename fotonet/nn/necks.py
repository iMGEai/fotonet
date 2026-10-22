"""Necks: top-down then bottom-up pyramid with CSP-Star junctions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import RepConv
from .blocks import CSPStar


class Neck(nn.Module):
    """Top-down then bottom-up pyramid with CSP-Star junctions."""

    def __init__(self, in_channels, neck_channels, neck_depths):
        super().__init__()
        c3, c4, c5 = in_channels
        n3, n4, n5 = neck_channels
        d3, d4, d5 = neck_depths
        self.td4 = CSPStar(c5 + c4, n4, d4)
        self.td3 = CSPStar(n4 + c3, n3, d3)
        self.down3 = RepConv(n3, n3, 3, 2)
        self.bu4 = CSPStar(n3 + n4, n4, d4)
        self.down4 = RepConv(n4, n4, 3, 2)
        self.bu5 = CSPStar(n4 + c5, n5, d5)
        self.out_channels = (n3, n4, n5)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        tracing = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        if not tracing and x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def forward(self, feats):
        p3, p4, p5 = feats
        t4 = self.td4(torch.cat((self._resize_like(p5, p4), p4), 1))
        t3 = self.td3(torch.cat((self._resize_like(t4, p3), p3), 1))
        o4 = self.bu4(torch.cat((self.down3(t3), t4), 1))
        o5 = self.bu5(torch.cat((self.down4(o4), p5), 1))
        return t3, o4, o5
