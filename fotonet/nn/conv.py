"""Convolution parts: Conv (conv + BatchNorm + SiLU) and RepConv (3x3 with foldable 1x1 and identity branches)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """Convolution, BatchNorm, optional SiLU."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=False) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class RepConv(nn.Module):
    """3x3 conv with training-only 1x1 and identity branches.

    ``reparameterize()`` folds all branches into ``conv`` (with bias) and turns
    ``bn`` into an identity, so the deployed module is one 3x3 convolution.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, act: bool = True):
        super().__init__()
        if k != 3:
            raise ValueError("RepConv is a 3x3 primitive.")
        self.c1, self.c2, self.s = int(c1), int(c2), int(s)
        self.conv = nn.Conv2d(c1, c2, 3, s, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.conv_1x1 = nn.Conv2d(c1, c2, 1, s, 0, bias=False)
        self.bn_1x1 = nn.BatchNorm2d(c2)
        self.bn_id = nn.BatchNorm2d(c1) if (c1 == c2 and s == 1) else None
        self.act = nn.SiLU(inplace=False) if act else nn.Identity()
        self.deployed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deployed:
            return self.act(self.conv(x))
        y = self.bn(self.conv(x)) + self.bn_1x1(self.conv_1x1(x))
        if self.bn_id is not None:
            y = y + self.bn_id(x)
        return self.act(y)

    @staticmethod
    def _fold(kernel, bn):
        std = (bn.running_var + bn.eps).sqrt()
        scale = bn.weight / std
        return kernel * scale.reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * scale

    @torch.no_grad()
    def reparameterize(self):
        if self.deployed:
            return self
        k3, b3 = self._fold(self.conv.weight, self.bn)
        k1, b1 = self._fold(F.pad(self.conv_1x1.weight, (1, 1, 1, 1)), self.bn_1x1)
        kernel, bias = k3 + k1, b3 + b1
        if self.bn_id is not None:
            identity = torch.zeros_like(self.conv.weight)
            channels = torch.arange(self.c1, device=identity.device)
            identity[channels, channels, 1, 1] = 1.0
            ki, bi = self._fold(identity, self.bn_id)
            kernel, bias = kernel + ki, bias + bi
        fused = nn.Conv2d(self.c1, self.c2, 3, self.s, 1, bias=True).to(kernel.device, kernel.dtype)
        fused.weight.copy_(kernel)
        fused.bias.copy_(bias)
        self.conv = fused
        self.bn = nn.Identity()
        del self.conv_1x1, self.bn_1x1, self.bn_id
        self.bn_id = None
        self.deployed = True
        return self


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fold an inference BatchNorm into the preceding convolution (one conv with bias)."""
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    weight = conv.weight.reshape(conv.out_channels, -1)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused.weight.data.copy_((weight * scale.reshape(-1, 1)).reshape_as(conv.weight))
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.bias.data.copy_(bn.bias + (bias - bn.running_mean) * scale)
    return fused
