"""Attention parts: global multi-head self-attention and the PSA block."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv


class Attention(nn.Module):
    """Global multi-head self-attention over a feature map, with a depthwise positional term."""

    def __init__(self, dim: int, head_dim: int = 32):
        super().__init__()
        self.heads = max(int(dim) // int(head_dim), 1)
        if dim % self.heads:
            raise ValueError(f"attention width {dim} is not divisible into {self.heads} heads.")
        self.head_dim = dim // self.heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.pe = Conv(dim, dim, 3, g=dim, act=False)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).reshape(b, 3, self.heads, self.head_dim, h * w).unbind(1)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(b, c, h, w)
        return self.proj(out + self.pe(v.reshape(b, c, h, w)))


class PSABlock(nn.Module):
    """Attention then a 2x MLP, both residual and zero-initialized (identity at step 0)."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.attn = Attention(dim)
        self.ffn = nn.Sequential(Conv(dim, hidden, 1), nn.Conv2d(hidden, dim, 1))
        nn.init.zeros_(self.attn.proj.weight)
        nn.init.zeros_(self.ffn[1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.ffn(x)


class PSA(nn.Module):
    """1x1 split, attention blocks on half the channels, 1x1 merge."""

    def __init__(self, channels: int, blocks: int = 1):
        super().__init__()
        self.hidden = channels // 2
        self.cv1 = Conv(channels, 2 * self.hidden, 1)
        self.blocks = nn.Sequential(*(PSABlock(self.hidden) for _ in range(int(blocks))))
        self.cv2 = Conv(2 * self.hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.hidden, self.hidden), dim=1)
        return self.cv2(torch.cat((a, self.blocks(b)), 1))
