"""fotonet2n (Hi Bird): CNN + Area-Attention + HyperACE + GeoSA Top-K Hybrid Detector.

Architecture overview:
- Backbone: Rebalanced CSPRep (unstarved P2/P3/P4 bottlenecks) with unified R-ELAN Area Attention
  (area=4 at P4 40x40, area=1 global at P5 20x20) + V2SPPF(shortcut=True).
- Neck: HyperACE + FullPAD-Lite linear-complexity O(N) cross-scale hypergraph fusion (E=24 latent
  hyperedges at 40x40) injecting gated global context into P3/P4/P5, followed by a slim PAN with
  fused RepConv stride-2 downsamples and global attention in bu5.
- Head: Hybrid-TopK decoupled architecture:
  - Stage 0: Lightweight dense proposal head (Rep3x3 reg + DW5x5/PW1x1 cls) selecting Top-300
    candidates on raw max logits, plus a training-only auxiliary O2M-C CNN branch (stripped at deploy).
  - Stage 1: 2-layer Geometry-Biased Sparse Self-Attention (GeoSA) decoder over the Top-300 queries
    with pairwise 5-D box-relation attention bias, configurable context re-sampling (``edge_sample``
    4-edge-midpoint bilinear sampling or ``hyper_xattn`` pure-GEMM hyperedge cross-attention), and
    iterative residual box + logit refinement.
  - Training-only Contrastive Denoising (CDN) via ``head.forward_cdn(...)`` (stripped at deploy).

All BatchNorm2d layers live inside Conv/RepConv so ``fuse()`` folds every single one cleanly,
leaving zero BatchNorm2d in the deployed graph. Eval output is single-pass NMS-free ``[B, 300, nc + 4]``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fotonet.nn.conv import Conv, RepConv, fuse_conv_bn
from fotonet.nn.heads import DetectHead, FEATURE_STRIDES, _clip_norm_xywh

from .graph import RepBottleneck, TRAINING_ONLY_PREFIXES, V2SPPF
from .kernels import (
    fused_box_refine,
    fused_edge_sample,
    fused_geo_bias,
    fused_stage0_decode,
    triton_available,
)
from .spec import GRAPH_VERSION, architecture_fingerprint, get_spec, normalize_layout, validate_layout

HYBRID_TRAINING_ONLY_PREFIXES = (*TRAINING_ONLY_PREFIXES, "head.cdn_label_emb.")


class AreaAttention(nn.Module):
    """YOLOv12-style spatial area-partitioned multi-head self-attention with 5x5 DW positional encoding.

    All Conv2d+BatchNorm2d projections are wrapped in ``Conv`` so ``fuse()`` leaves zero BatchNorm2d.
    Partitions the flattened spatial sequence into ``area`` equal strips (H*W is always divisible by 4
    after stride-32 input padding, so area=4 works cleanly for every valid input resolution).
    """

    def __init__(self, dim: int, area: int = 1, head_dim: int = 32, pe_kernel: int = 5):
        super().__init__()
        self.dim = int(dim)
        self.area = max(int(area), 1)
        self.heads = max(self.dim // int(head_dim), 1)
        if self.dim % self.heads:
            raise ValueError(f"AreaAttention dim {self.dim} not divisible by {self.heads} heads.")
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim ** -0.5
        self.qkv = Conv(self.dim, 3 * self.dim, 1, act=False)
        self.pe = Conv(self.dim, self.dim, pe_kernel, g=self.dim, act=False)
        self.proj = Conv(self.dim, self.dim, 1, act=False)
        nn.init.zeros_(self.proj.bn.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        pe_v = self.pe(v)
        tracing = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        if self.area > 1 and (tracing or n % self.area == 0):
            a = self.area
            if tracing:
                q = q.reshape(b, self.heads, self.head_dim, a, -1).permute(0, 3, 1, 4, 2)  # [B, A, H, L, D]
                k = k.reshape(b, self.heads, self.head_dim, a, -1).permute(0, 3, 1, 2, 4)  # [B, A, H, D, L]
                v_t = v.reshape(b, self.heads, self.head_dim, a, -1).permute(0, 3, 1, 4, 2)
                attn = (torch.matmul(q, k) * self.scale).softmax(dim=-1)
                out = torch.matmul(attn, v_t).permute(0, 2, 4, 1, 3).reshape(b, c, h, w)
            else:
                l_len = n // a
                q = q.reshape(b, self.heads, self.head_dim, a, l_len).permute(0, 3, 1, 4, 2).reshape(b * a, self.heads, l_len, self.head_dim)
                k = k.reshape(b, self.heads, self.head_dim, a, l_len).permute(0, 3, 1, 4, 2).reshape(b * a, self.heads, l_len, self.head_dim)
                v_t = v.reshape(b, self.heads, self.head_dim, a, l_len).permute(0, 3, 1, 4, 2).reshape(b * a, self.heads, l_len, self.head_dim)
                out = F.scaled_dot_product_attention(q, k, v_t)
                out = out.reshape(b, a, self.heads, l_len, self.head_dim).permute(0, 2, 4, 1, 3).reshape(b, c, h, w)
        else:
            if tracing:
                q = q.reshape(b, self.heads, self.head_dim, n).transpose(-2, -1)  # [B, H, N, D]
                k = k.reshape(b, self.heads, self.head_dim, n)                    # [B, H, D, N]
                v_t = v.reshape(b, self.heads, self.head_dim, n).transpose(-2, -1)
                attn = (torch.matmul(q, k) * self.scale).softmax(dim=-1)
                out = torch.matmul(attn, v_t).transpose(-2, -1).reshape(b, c, h, w)
            else:
                q = q.reshape(b, self.heads, self.head_dim, n).transpose(-2, -1)
                k = k.reshape(b, self.heads, self.head_dim, n).transpose(-2, -1)
                v_t = v.reshape(b, self.heads, self.head_dim, n).transpose(-2, -1)
                out = F.scaled_dot_product_attention(q, k, v_t).transpose(-2, -1).reshape(b, c, h, w)
        return self.proj(out + pe_v)


class RELANBlock(nn.Module):
    """AreaAttention + compact 1.5x FFN with R-ELAN learnable residual scaling (gamma init 0.01)."""

    def __init__(self, dim: int, area: int = 1, mlp_ratio: float = 1.5, gamma_init: float = 0.01):
        super().__init__()
        hidden = max(int(round(dim * float(mlp_ratio))), 16)
        self.attn = AreaAttention(dim, area=area)
        self.ffn = nn.Sequential(Conv(dim, hidden, 1), Conv(hidden, dim, 1, act=False))
        nn.init.zeros_(self.ffn[1].bn.weight)
        self.gamma_attn = nn.Parameter(torch.full((1, dim, 1, 1), float(gamma_init)))
        self.gamma_ffn = nn.Parameter(torch.full((1, dim, 1, 1), float(gamma_init)))
        self._gamma_attn_folded = False
        self._gamma_ffn_folded = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attn(x)
        x = x + (a if self._gamma_attn_folded else self.gamma_attn * a)
        f = self.ffn(x)
        x = x + (f if self._gamma_ffn_folded else self.gamma_ffn * f)
        return x


class UnifiedRELANStage(nn.Module):
    """Single CSP split-merge unifying RepBottlenecks and Area-Attention R-ELAN blocks."""

    def __init__(self, c_in: int, c_out: int, hidden: int, n_conv: int = 1, e_conv: float = 0.5,
                 n_attn: int = 0, area: int = 1, mlp_ratio: float = 1.5, gamma_init: float = 0.01):
        super().__init__()
        self.cv1 = Conv(c_in, 2 * hidden, 1)
        blocks: list[nn.Module] = []
        for _ in range(int(n_conv)):
            blocks.append(RepBottleneck(hidden, e_conv))
        for _ in range(int(n_attn)):
            blocks.append(RELANBlock(hidden, area=area, mlp_ratio=mlp_ratio, gamma_init=gamma_init))
        self.blocks = nn.ModuleList(blocks)
        self.cv2 = Conv((2 + len(blocks)) * hidden, c_out, 1)
        self.gamma_stage = (
            nn.Parameter(torch.full((1, c_out, 1, 1), float(gamma_init)))
            if (int(c_in) == int(c_out) and int(n_attn) > 0)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).chunk(2, 1)
        outputs = [a, b]
        for block in self.blocks:
            b = block(b)
            outputs.append(b)
        out = self.cv2(torch.cat(outputs, 1))
        if self.gamma_stage is not None:
            return x + self.gamma_stage * out
        return out


class HyperACE(nn.Module):
    """Linear-complexity O(N*E*C) cross-scale hypergraph correlation + FullPAD-Lite distribution.

    Fuses aligned [AvgPool(P3), P4, Upsample(P5)] at P4 resolution (40x40), constructs E=24 learnable
    soft global hyperedges, runs two-step vertex->hyperedge->vertex message passing purely via GEMMs,
    and distributes gated global context back to P3, P4, and P5.
    """

    def __init__(self, in_channels: tuple[int, int, int], hidden: int = 64, num_edges: int = 24,
                 gate_init: float = 0.01):
        super().__init__()
        c3, c4, c5 = (int(v) for v in in_channels)
        self.hidden = int(hidden)
        self.num_edges = int(num_edges)
        self.fuse_in = Conv(c3 + c4 + c5, self.hidden, 1)
        self.hyper_inc = Conv(self.hidden, self.num_edges, 1, act=False)
        self.edge_proj = nn.Linear(self.hidden, self.hidden)
        self.edge_norm = nn.LayerNorm(self.hidden)
        self.edge_act = nn.SiLU(inplace=False)
        self.gamma_hyp = nn.Parameter(torch.full((1, self.hidden, 1, 1), float(gate_init)))

        # FullPAD-Lite gated distribution projections (all BN-wrapped Conv -> cleanly fused)
        self.dist_p3 = Conv(self.hidden, c3, 1, act=False)
        self.dist_p4 = Conv(self.hidden, c4, 1, act=False)
        self.dist_p5 = Conv(self.hidden, c5, 1, act=False)
        self.alpha3 = nn.Parameter(torch.full((1, c3, 1, 1), float(gate_init)))
        self.alpha4 = nn.Parameter(torch.full((1, c4, 1, 1), float(gate_init)))
        self.alpha5 = nn.Parameter(torch.full((1, c5, 1, 1), float(gate_init)))
        self._alphas_folded = False

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        b, _, h4, w4 = p4.shape
        tracing = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        static_exp = tracing and bool(getattr(self, "_static_export", False))
        p3_d = F.avg_pool2d(p3, kernel_size=2, stride=2)
        if not tracing and p3_d.shape[-2:] != (h4, w4):
            p3_d = F.interpolate(p3, size=(h4, w4), mode="bilinear", align_corners=False)
        if static_exp and int(p5.shape[-2]) * 2 == int(h4) and int(p5.shape[-1]) * 2 == int(w4):
            p5_u = F.interpolate(p5, scale_factor=2.0, mode="nearest")
        else:
            p5_u = F.interpolate(p5, size=(h4, w4), mode="nearest")
        x = self.fuse_in(torch.cat((p3_d, p4, p5_u), dim=1))  # [B, C_h, H4, W4]

        # Soft incidence matrix H: [B, E, N]
        h_mat = self.hyper_inc(x).flatten(2).softmax(dim=1)
        x_flat = x.flatten(2)                                  # [B, C_h, N]
        # Vertex -> Hyperedge: Z = H @ X^T -> [B, E, C_h]
        if tracing and not static_exp:
            n_sqrt = torch._shape_as_tensor(h_mat)[2].to(device=h_mat.device, dtype=h_mat.dtype).sqrt().clamp(min=1.0)
        else:
            n_sqrt = max(float(int(h4) * int(w4)) ** 0.5, 1.0)
        z = torch.bmm(h_mat, x_flat.transpose(1, 2)) / n_sqrt
        z_enh = self.edge_act(self.edge_norm(self.edge_proj(z)))  # [B, E, C_h]
        # Hyperedge -> Vertex: [B, C_h, E] @ [B, E, N] -> [B, C_h, N]
        x_msg = torch.bmm(z_enh.transpose(1, 2), h_mat).reshape(b, self.hidden, h4, w4)
        x_out = x + self.gamma_hyp * x_msg

        # FullPAD-Lite distribution: project P3 at 40x40 then upsample (4x cheaper); pool P5 to 20x20 then project
        d3_raw = self.dist_p3(x_out)
        if static_exp and int(d3_raw.shape[-2]) * 2 == int(p3.shape[-2]) and int(d3_raw.shape[-1]) * 2 == int(p3.shape[-1]):
            d3 = F.interpolate(d3_raw, scale_factor=2.0, mode="nearest")
        else:
            d3 = F.interpolate(d3_raw, size=p3.shape[-2:], mode="nearest")
        d4 = self.dist_p4(x_out)
        x_p5 = F.avg_pool2d(x_out, kernel_size=2, stride=2)
        if not tracing and x_p5.shape[-2:] != p5.shape[-2:]:
            x_p5 = F.interpolate(x_out, size=p5.shape[-2:], mode="bilinear", align_corners=False)
        d5 = self.dist_p5(x_p5)

        if self._alphas_folded:
            return p3 + d3, p4 + d4, p5 + d5, x_out, z_enh
        return p3 + self.alpha3 * d3, p4 + self.alpha4 * d4, p5 + self.alpha5 * d5, x_out, z_enh


class V2HybridBackbone(nn.Module):
    """Unstarved CSPRep + Unified R-ELAN Area Attention at P4 (area=4) and P5 (area=1) + V2SPPF."""

    def __init__(self, stem: int, p2: int, stages, stage_attn, mlp_ratio: float = 1.5):
        super().__init__()
        s2, s3, s4, s5 = stages
        (n2a, a2), (n3a, a3), (n4a, a4), (n5a, a5) = stage_attn
        self.stem = RepConv(3, stem, 3, 2)
        self.p2 = RepConv(stem, p2, 3, 2)
        self.stage2 = UnifiedRELANStage(p2, s2[0], s2[1], s2[2], s2[3], n2a, a2, mlp_ratio=mlp_ratio)
        self.down3 = RepConv(s2[0], s2[0], 3, 2)
        self.stage3 = UnifiedRELANStage(s2[0], s3[0], s3[1], s3[2], s3[3], n3a, a3, mlp_ratio=mlp_ratio)
        self.down4 = RepConv(s3[0], s3[0], 3, 2)
        self.stage4 = UnifiedRELANStage(s3[0], s4[0], s4[1], s4[2], s4[3], n4a, a4, mlp_ratio=mlp_ratio)
        self.down5 = RepConv(s4[0], s5[0], 3, 2)
        self.stage5 = UnifiedRELANStage(s5[0], s5[0], s5[1], s5[2], s5[3], n5a, a5, mlp_ratio=mlp_ratio)
        self.sppf = V2SPPF(s5[0], s5[0], shortcut=True, cv1_act=False)
        self.out_channels = (s3[0], s4[0], s5[0])

    def forward(self, x: torch.Tensor):
        p3 = self.stage3(self.down3(self.stage2(self.p2(self.stem(x)))))
        p4 = self.stage4(self.down4(p3))
        p5 = self.sppf(self.stage5(self.down5(p4)))
        return p3, p4, p5


class V2HybridNeck(nn.Module):
    """HyperACE + FullPAD-Lite cross-scale fusion feeding a slim RepConv PAN with bu5 global attention."""

    def __init__(self, in_channels, junctions, hyper_ace=(64, 24), neck_attention: bool = True,
                 mlp_ratio: float = 1.5):
        super().__init__()
        c3, c4, c5 = (int(v) for v in in_channels)
        td4, td3, bu4, bu5 = junctions
        hc, he = int(hyper_ace[0]), int(hyper_ace[1])
        self.hyper = HyperACE((c3, c4, c5), hidden=hc, num_edges=he)
        self.td4 = UnifiedRELANStage(c5 + c4, td4[0], td4[1], td4[2], td4[3], 0, 1, mlp_ratio=mlp_ratio)
        self.td3 = UnifiedRELANStage(td4[0] + c3, td3[0], td3[1], td3[2], td3[3], 0, 1, mlp_ratio=mlp_ratio)
        self.down3 = RepConv(td3[0], td3[0], 3, 2)
        self.bu4 = UnifiedRELANStage(td3[0] + td4[0], bu4[0], bu4[1], bu4[2], bu4[3], 0, 1, mlp_ratio=mlp_ratio)
        self.down4 = RepConv(bu4[0], bu4[0], 3, 2)
        self.bu5 = UnifiedRELANStage(
            bu4[0] + c5, bu5[0], bu5[1], bu5[2], bu5[3],
            n_attn=1 if neck_attention else 0, area=1, mlp_ratio=mlp_ratio,
        )
        self.out_channels = (td3[0], bu4[0], bu5[0])
        self.hyper_dim = hc
        self.hyper_edges = he

    def forward(self, feats):
        p3, p4, p5 = feats
        p3, p4, p5, hyper_map, hyper_tokens = self.hyper(p3, p4, p5)
        static_exp = (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()) and bool(getattr(self, "_static_export", False))
        if static_exp and int(p5.shape[-2]) * 2 == int(p4.shape[-2]) and int(p5.shape[-1]) * 2 == int(p4.shape[-1]):
            u5 = F.interpolate(p5, scale_factor=2.0, mode="nearest")
        else:
            u5 = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        t4 = self.td4(torch.cat((u5, p4), dim=1))
        if static_exp and int(t4.shape[-2]) * 2 == int(p3.shape[-2]) and int(t4.shape[-1]) * 2 == int(p3.shape[-1]):
            u4 = F.interpolate(t4, scale_factor=2.0, mode="nearest")
        else:
            u4 = F.interpolate(t4, size=p3.shape[-2:], mode="nearest")
        t3 = self.td3(torch.cat((u4, p3), dim=1))
        o4 = self.bu4(torch.cat((self.down3(t3), t4), dim=1))
        o5 = self.bu5(torch.cat((self.down4(o4), p5), dim=1))
        return (t3, o4, o5), hyper_map, hyper_tokens


class HybridStage0Branch(nn.Module):
    """Lightweight Stage-0 dense proposal / O2M-C branch (Rep3x3 reg + DW5x5/PW1x1 cls)."""

    def __init__(self, in_channels, nc: int, reg_width: int = 28, cls_width: int = 72):
        super().__init__()
        self.reg_stem = nn.ModuleList(RepConv(c, reg_width, 3) for c in in_channels)
        self.reg_pred = nn.ModuleList(nn.Conv2d(reg_width, 4, 1) for _ in in_channels)
        self.cls_stem = nn.ModuleList(
            nn.Sequential(Conv(c, c, 5, g=c), Conv(c, cls_width, 1))
            for c in in_channels
        )
        self.cls_pred = nn.ModuleList(nn.Conv2d(cls_width, nc, 1) for _ in in_channels)

    def forward(self, i: int, feat: torch.Tensor):
        cls_h = self.cls_stem[i](feat)
        reg_raw = self.reg_pred[i](self.reg_stem[i](feat))
        # Non-vanishing shifted ELU distance decoding
        reg_dist = F.elu(reg_raw, alpha=1.0) + 1.0
        return self.cls_pred[i](cls_h), reg_dist, cls_h

    def forward_raw(self, i: int, feat: torch.Tensor):
        """Return (cls_logits_map, raw_reg_map) prior to shifted-ELU for fused CUDA decoding."""
        cls_h = self.cls_stem[i](feat)
        reg_raw = self.reg_pred[i](self.reg_stem[i](feat))
        return self.cls_pred[i](cls_h), reg_raw


def _pairwise_box_relation(boxes: torch.Tensor) -> torch.Tensor:
    """Compute scale-invariant 5-D geometric relation [B, Q, Q, 5] between normalized cxcywh boxes."""
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w = boxes[..., 2].clamp(min=1e-4)
    h = boxes[..., 3].clamp(min=1e-4)

    dx = ((cx.unsqueeze(1) - cx.unsqueeze(2)) / w.unsqueeze(2)).clamp(-10.0, 10.0)
    dy = ((cy.unsqueeze(1) - cy.unsqueeze(2)) / h.unsqueeze(2)).clamp(-10.0, 10.0)
    dw = torch.log(w.unsqueeze(1) / w.unsqueeze(2)).clamp(-4.0, 4.0)
    dh = torch.log(h.unsqueeze(1) / h.unsqueeze(2)).clamp(-4.0, 4.0)

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    ix1 = torch.maximum(x1.unsqueeze(2), x1.unsqueeze(1))
    iy1 = torch.maximum(y1.unsqueeze(2), y1.unsqueeze(1))
    ix2 = torch.minimum(x2.unsqueeze(2), x2.unsqueeze(1))
    iy2 = torch.minimum(y2.unsqueeze(2), y2.unsqueeze(1))
    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)
    area = w * h
    iou = inter / (area.unsqueeze(2) + area.unsqueeze(1) - inter + 1e-6)
    return torch.stack((dx, dy, dw, dh, iou), dim=-1)


def _refine_boxes(boxes: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Apply smooth scale-equivariant residual box refinement keeping normalized cxcywh in [0, 1]."""
    w = boxes[..., 2].clamp(min=1e-4)
    h = boxes[..., 3].clamp(min=1e-4)
    cx_new = (boxes[..., 0] + delta[..., 0] * w).clamp(0.0, 1.0)
    cy_new = (boxes[..., 1] + delta[..., 1] * h).clamp(0.0, 1.0)
    w_new = (w * torch.exp(delta[..., 2].clamp(-2.0, 2.0))).clamp(1e-4, 1.0)
    h_new = (h * torch.exp(delta[..., 3].clamp(-2.0, 2.0))).clamp(1e-4, 1.0)
    return torch.stack((cx_new, cy_new, w_new, h_new), dim=-1)


class GeoSALayer(nn.Module):
    """Geometry-Biased Sparse Query Self-Attention + Context Re-Sampling + Residual Refinement."""

    def __init__(self, dim: int = 128, nc: int = 80, num_heads: int = 4, hyper_dim: int = 64,
                 context_mode: str = "edge_sample", gamma_init: float = 0.01):
        super().__init__()
        mode = str(context_mode).strip().lower()
        if mode not in ("edge_sample", "hyper_xattn"):
            raise ValueError(f"decoder_context must be 'edge_sample' or 'hyper_xattn', got {context_mode!r}.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.context_mode = mode

        # Pairwise 5-D geometric attention bias
        self.geo_mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.SiLU(inplace=False),
            nn.Linear(16, self.num_heads),
        )
        nn.init.zeros_(self.geo_mlp[2].weight)
        nn.init.zeros_(self.geo_mlp[2].bias)

        # Pre-norm Query Self-Attention (small-std init breaks triple zero-init deadlock so geo_mlp learns at step 1)
        self.norm1 = nn.LayerNorm(self.dim)
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.attn_out = nn.Linear(self.dim, self.dim)
        nn.init.normal_(self.attn_out.weight, std=0.01)
        nn.init.zeros_(self.attn_out.bias)

        # Context re-sampling branch (either 4-edge bilinear sampling OR pure-GEMM hyperedge cross-attention)
        self.norm_ctx = nn.LayerNorm(self.dim)
        if self.context_mode == "edge_sample":
            self.edge_proj = nn.Linear(4 * int(hyper_dim), self.dim)
            nn.init.normal_(self.edge_proj.weight, std=0.01)
            nn.init.zeros_(self.edge_proj.bias)
            self.gamma_ctx = nn.Parameter(torch.full((1, 1, self.dim), float(gamma_init)))
        else:
            self.ctx_q = nn.Linear(self.dim, self.dim)
            self.ctx_kv = nn.Linear(int(hyper_dim), 2 * self.dim)
            self.ctx_out = nn.Linear(self.dim, self.dim)
            nn.init.normal_(self.ctx_out.weight, std=0.01)
            nn.init.zeros_(self.ctx_out.bias)
            self.gamma_ctx = nn.Parameter(torch.full((1, 1, self.dim), float(gamma_init)))

        # Pre-norm 2x FFN
        self.norm2 = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, 2 * self.dim),
            nn.SiLU(inplace=False),
            nn.Linear(2 * self.dim, self.dim),
        )
        nn.init.normal_(self.ffn[2].weight, std=0.01)
        nn.init.zeros_(self.ffn[2].bias)

        # Residual box delta (exact zero-init) and logit refinement (small-std init for immediate step-1 flow)
        self.norm_out = nn.LayerNorm(self.dim)
        self.reg_delta = nn.Linear(self.dim, 4)
        self.cls_delta = nn.Linear(self.dim, int(nc))
        nn.init.zeros_(self.reg_delta.weight)
        nn.init.zeros_(self.reg_delta.bias)
        nn.init.normal_(self.cls_delta.weight, std=0.01)
        nn.init.zeros_(self.cls_delta.bias)
        self._gamma_ctx_folded = False

    def _edge_sample(self, boxes: torch.Tensor, hyper_map: torch.Tensor) -> torch.Tensor:
        """Sample 4 edge midpoints (top, bottom, left, right) of each box from hyper_map."""
        cx = boxes[..., 0]
        cy = boxes[..., 1]
        hw = 0.5 * boxes[..., 2]
        hh = 0.5 * boxes[..., 3]
        # [B, Q, 4, 2] normalized in [-1, 1] for grid_sample (x, y order)
        pts_x = torch.stack((cx, cx, cx - hw, cx + hw), dim=-1) * 2.0 - 1.0
        pts_y = torch.stack((cy - hh, cy + hh, cy, cy), dim=-1) * 2.0 - 1.0
        grid = torch.stack((pts_x.clamp(-1.0, 1.0), pts_y.clamp(-1.0, 1.0)), dim=-1).to(hyper_map.dtype)
        sampled = F.grid_sample(hyper_map, grid, mode="bilinear", padding_mode="border", align_corners=False)
        # sampled: [B, C_h, Q, 4] -> [B, Q, 4 * C_h]
        b, ch, q_len, _ = sampled.shape
        return sampled.permute(0, 2, 3, 1).reshape(b, q_len, 4 * ch)

    def forward(self, q: torch.Tensor, boxes: torch.Tensor, logits: torch.Tensor,
                hyper_map: torch.Tensor, hyper_tokens: torch.Tensor,
                attn_mask: torch.Tensor | None = None):
        b, nq, d = q.shape

        # 1. Geometry-Biased Query Self-Attention
        tracing = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        use_triton = (not self.training) and (not tracing) and triton_available(boxes)
        geo_bias = fused_geo_bias(boxes, self.geo_mlp) if use_triton else None
        if geo_bias is None:
            rel = _pairwise_box_relation(boxes.detach())                 # [B, Q, Q, 5]
            geo_bias = self.geo_mlp(rel).permute(0, 3, 1, 2)             # [B, H, Q, Q]

        qn = self.norm1(q)
        qkv = self.qkv(qn).reshape(b, nq, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qh, kh, vh = qkv.unbind(0)                                       # [B, H, Q, D_h]
        if tracing:
            attn = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale + geo_bias
            if attn_mask is not None:
                attn = attn + attn_mask.to(attn.dtype)
            attn = attn.softmax(dim=-1)
            sa_out = torch.matmul(attn, vh).transpose(1, 2).reshape(b, nq, d)
        else:
            bias = geo_bias if attn_mask is None else (geo_bias + attn_mask.to(geo_bias.dtype))
            sa_out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias.to(qh.dtype)).transpose(1, 2).reshape(b, nq, d)
        q = q + self.attn_out(sa_out)

        # 2. Context Re-Sampling (either 4-edge bilinear sampling or pure-GEMM hyperedge cross-attention)
        if self.context_mode == "edge_sample":
            edge_feat = fused_edge_sample(boxes, hyper_map) if use_triton else None
            if edge_feat is None:
                edge_feat = self._edge_sample(boxes.detach(), hyper_map)
            ctx_delta = self.edge_proj(edge_feat.to(q.dtype))
            q = q + (ctx_delta if self._gamma_ctx_folded else self.gamma_ctx * ctx_delta)
        else:
            q_ctx = self.ctx_q(self.norm_ctx(q)).reshape(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
            ne = hyper_tokens.shape[1]
            kv = self.ctx_kv(hyper_tokens.to(q.dtype)).reshape(b, ne, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            kc, vc = kv.unbind(0)
            if tracing:
                c_attn = (torch.matmul(q_ctx, kc.transpose(-2, -1)) * self.scale).softmax(dim=-1)
                c_out = torch.matmul(c_attn, vc).transpose(1, 2).reshape(b, nq, d)
            else:
                c_out = F.scaled_dot_product_attention(q_ctx, kc, vc).transpose(1, 2).reshape(b, nq, d)
            ctx_delta = self.ctx_out(c_out)
            q = q + (ctx_delta if self._gamma_ctx_folded else self.gamma_ctx * ctx_delta)

        # 3. FFN + Residual Box & Logit Refinement
        q = q + self.ffn(self.norm2(q))
        q_out = self.norm_out(q)
        reg_d = self.reg_delta(q_out)
        boxes_new = fused_box_refine(boxes, reg_d) if use_triton else None
        if boxes_new is None:
            boxes_new = _refine_boxes(boxes, reg_d)
        logits_new = logits + self.cls_delta(q_out)
        return q, boxes_new, logits_new


class V2HybridHead(nn.Module):
    """Stage-0 dense proposal head + Top-300 gather + 2-layer GeoSA query refinement decoder.

    Includes training-only auxiliary O2M-C CNN branch and CDN label embedding (both stripped at deploy).
    Preserves compatibility with ``probes._swap_heads`` in eval mode.
    """

    def __init__(self, in_channels, nc: int = 80, hidden=(28, 72), strides=FEATURE_STRIDES, imgsz: int = 640,
                 hyper_dim: int = 64, decoder_dim: int = 128, decoder_layers: int = 2,
                 num_queries: int = 300, decoder_context: str = "edge_sample",
                 query_feat_dim: int = 64, aux_objectness: bool = False):
        super().__init__()
        reg_width, cls_width = (int(v) for v in hidden)
        self.nc = int(nc)
        self.reg_max = 1
        self.imgsz = int(imgsz)
        self.strides = tuple(int(s) for s in strides)
        self.num_queries = int(num_queries)
        self.decoder_dim = int(decoder_dim)
        self.decoder_layers = int(decoder_layers)
        self.decoder_context = str(decoder_context).strip().lower()
        self.detach_o2o = True
        self.o2o_body_grad_scale = 0.35

        self.o2o = HybridStage0Branch(in_channels, self.nc, reg_width, cls_width)
        self.o2m = HybridStage0Branch(in_channels, self.nc, reg_width, cls_width)
        # Store module id (int, not an nn.Module attribute) to detect probes._swap_heads without aliasing state_dict.
        self._o2m_id: int | None = id(self.o2m)
        self.aux_obj = (
            nn.ModuleList(nn.Conv2d(cls_width, 1, 1) for _ in in_channels) if aux_objectness else None
        )
        self.o2o_groups = nn.ModuleList()

        # Compact per-level 1x1 projection to shared query_feat_dim (64ch) -> single gather on Top-300 -> Linear(64->128)
        self.feat_align = nn.ModuleList(Conv(c, int(query_feat_dim), 1) for c in in_channels)
        self.query_up = nn.Linear(int(query_feat_dim), self.decoder_dim)
        self.box_pe = nn.Sequential(
            nn.Linear(4, self.decoder_dim),
            nn.SiLU(inplace=False),
            nn.Linear(self.decoder_dim, self.decoder_dim),
        )
        self.query_norm = nn.LayerNorm(self.decoder_dim)
        self.decoder = nn.ModuleList(
            GeoSALayer(
                dim=self.decoder_dim, nc=self.nc, num_heads=4,
                hyper_dim=int(hyper_dim), context_mode=self.decoder_context,
            )
            for _ in range(self.decoder_layers)
        )

        # Training-only Contrastive Denoising (CDN) class embedding (nc + 1 for background); stripped at deploy
        self.cdn_label_emb = nn.Embedding(self.nc + 1, self.decoder_dim)
        self._grid_cache: dict[tuple, torch.Tensor] = {}
        self._tie_cache: dict[tuple, torch.Tensor] = {}
        self.init_biases()

    def _get_tie(self, n_total: int, device: torch.device) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch._dynamo.is_compiling():
            return (torch.arange(n_total, device=device, dtype=torch.float32) * 2e-6).unsqueeze(0)
        key = (n_total, device)
        tie = self._tie_cache.get(key)
        if tie is None:
            tie = (torch.arange(n_total, device=device, dtype=torch.float32) * 2e-6).unsqueeze(0)
            if len(self._tie_cache) >= 8:
                self._tie_cache.pop(next(iter(self._tie_cache)))
            self._tie_cache[key] = tie
        return tie

    def _get_grid(self, nx: int, ny: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if torch._dynamo.is_compiling():
            return DetectHead._grid(nx, ny, device, dtype)
        key = (nx, ny, device, dtype)
        grid = self._grid_cache.get(key)
        if grid is None:
            grid = DetectHead._grid(nx, ny, device, dtype)
            if len(self._grid_cache) >= 16:
                self._grid_cache.pop(next(iter(self._grid_cache)))
            self._grid_cache[key] = grid
        return grid

    @property
    def has_o2m(self) -> bool:
        return self.o2m is not None

    def strip_o2m_for_inference(self):
        self.o2m = None
        self._o2m_id = None
        self.aux_obj = None
        self.cdn_label_emb = None
        return self

    def init_biases(self):
        for branch in (self.o2o, self.o2m):
            if branch is None:
                continue
            for i, stride in enumerate(self.strides):
                nn.init.normal_(branch.cls_pred[i].weight, std=0.01)
                nn.init.constant_(branch.cls_pred[i].bias, math.log(5.0 / self.nc / (640.0 / stride) ** 2))
                nn.init.normal_(branch.reg_pred[i].weight, std=0.01)
                nn.init.constant_(branch.reg_pred[i].bias, 1.0)  # sELU(1.0) = 2.0 cells
        if self.aux_obj is not None:
            for layer, stride in zip(self.aux_obj, self.strides):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, math.log(5.0 / (640.0 / stride) ** 2))

    def _decode(self, cls_raw: torch.Tensor, ltrb: torch.Tensor, grid: torch.Tensor, stride: float, imgsz):
        batch = cls_raw.shape[0]
        logits = cls_raw.reshape(batch, self.nc, -1).permute(0, 2, 1)
        dist = ltrb.reshape(batch, 4, -1).permute(0, 2, 1).to(logits.dtype)
        return logits, DetectHead._dist2bbox(dist, grid, stride, imgsz)

    def _scale_body_grad(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if self.detach_o2o:
            return x.detach()
        s = float(self.o2o_body_grad_scale)
        if s >= 1.0:
            return x
        if s <= 0.0:
            return x.detach()
        # Straight-through gradient scaling: exact identity forward, s * grad backward
        return x * s + x.detach() * (1.0 - s)

    def forward_cdn(self, dn_boxes: torch.Tensor, dn_labels: torch.Tensor,
                    hyper_map: torch.Tensor, hyper_tokens: torch.Tensor,
                    attn_mask: torch.Tensor | None = None):
        """Run training-only Contrastive Denoising (CDN) queries through the shared GeoSA decoder."""
        if self.cdn_label_emb is None:
            return [], []
        amp_enabled = hyper_map.is_cuda and hyper_map.dtype in (torch.bfloat16, torch.float16)
        with torch.autocast(device_type=hyper_map.device.type, dtype=hyper_map.dtype, enabled=amp_enabled):
            box_dtype = hyper_map.dtype if amp_enabled else self.cdn_label_emb.weight.dtype
            q = self.query_norm(self.cdn_label_emb(dn_labels) + self.box_pe(dn_boxes.to(box_dtype)))
            # Init prior logits near background bias so residual cls_delta learns cleanly
            b, nq = dn_labels.shape
            prior_bias = math.log(0.01 / 0.99)
            logits = torch.full((b, nq, self.nc), prior_bias, device=q.device, dtype=q.dtype)
            boxes = dn_boxes.to(q.dtype)
            h_map = self._scale_body_grad(hyper_map)
            h_tok = self._scale_body_grad(hyper_tokens)
            if not amp_enabled:
                h_map = h_map.to(q.dtype)
                h_tok = h_tok.to(q.dtype)
            dec_logits, dec_boxes = [], []
            for layer in self.decoder:
                q, boxes, logits = layer(q, boxes, logits, h_map, h_tok, attn_mask=attn_mask)
                dec_logits.append(logits)
                dec_boxes.append(boxes)
        return dec_logits, dec_boxes

    def forward(self, feats, hyper_map: torch.Tensor, hyper_tokens: torch.Tensor,
                imgsz=None, use_o2m: bool = True):
        if imgsz is None:
            imgsz = float(self.imgsz)
        train = self.training
        # Compatibility hook for probes._swap_heads (swaps self.o2o and self.o2m during eval)
        probing_o2m = (not train) and (self.o2m is not None) and (id(self.o2o) == self._o2m_id)
        run_o2m = bool(train and use_o2m and self.o2m is not None)

        logits_o, boxes_o, logits_m, boxes_m = [], [], [], []
        anchors, stride_rows, objectness = [], [], []
        aligned_feats = []

        tracing = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        use_triton_s0 = (not train) and (not tracing) and (not probing_o2m) and triton_available(feats[0])
        if use_triton_s0:
            if isinstance(imgsz, (tuple, list)):
                imgsz_w, imgsz_h = float(imgsz[0]), float(imgsz[1])
            elif torch.is_tensor(imgsz):
                use_triton_s0 = False
                imgsz_w = imgsz_h = float(self.imgsz)
            else:
                imgsz_w = imgsz_h = float(imgsz)

        for i, feat in enumerate(feats):
            stride = float(self.strides[i])
            feat_o = self._scale_body_grad(feat)
            if use_triton_s0:
                cls_raw, reg_raw = self.o2o.forward_raw(i, feat_o)
                b_sz = cls_raw.shape[0]
                level_logits = cls_raw.reshape(b_sz, self.nc, -1).permute(0, 2, 1)
                level_boxes = fused_stage0_decode(reg_raw, stride, imgsz_w, imgsz_h, apply_selu=True)
                if level_boxes is None:
                    grid = self._get_grid(feat.shape[-1], feat.shape[-2], feat.device, feat.dtype)
                    level_logits, level_boxes = self._decode(cls_raw, F.elu(reg_raw, alpha=1.0) + 1.0, grid, stride, imgsz)
                else:
                    level_boxes = level_boxes.to(level_logits.dtype)
            else:
                grid = self._get_grid(feat.shape[-1], feat.shape[-2], feat.device, feat.dtype)
                cls_raw, ltrb, _ = self.o2o(i, feat_o)
                level_logits, level_boxes = self._decode(cls_raw, ltrb, grid, stride, imgsz)
            logits_o.append(level_logits)
            boxes_o.append(level_boxes)
            if not probing_o2m:
                af = self.feat_align[i](feat_o)
                aligned_feats.append(af.flatten(2).transpose(1, 2))  # [B, N_i, C_q]
            if not train:
                continue
            anchors.append(grid.squeeze(0))
            stride_rows.append(torch.full(
                (feat.shape[-2] * feat.shape[-1], 1), stride, device=feat.device, dtype=feat.dtype))
            if run_o2m:
                cls_m, ltrb_m, cls_hm = self.o2m(i, feat)
                ml, mb = self._decode(cls_m, ltrb_m, grid, stride, imgsz)
                logits_m.append(ml)
                boxes_m.append(mb)
                if self.aux_obj is not None:
                    objectness.append(self.aux_obj[i](cls_hm).flatten(1))

        pred_logits_s0 = torch.cat(logits_o, dim=1)     # [B, N, nc]
        pred_boxes_s0 = torch.cat(boxes_o, dim=1)       # [B, N, 4]

        if probing_o2m:
            return torch.cat((pred_logits_s0.clamp(-25.0, 25.0), pred_boxes_s0), dim=-1)

        # Stage 1: Select Top-K proposals on raw max logits BEFORE query projection & GeoSA
        b, n_total, _ = pred_logits_s0.shape
        k = self.num_queries if tracing else min(self.num_queries, n_total)
        if train:
            max_scores = pred_logits_s0.detach().max(dim=-1).values
        else:
            # Deterministic monotone tie-breaker (does not alter output logits) so TopK order is
            # bit-exact across PyTorch, ONNX Runtime, and TensorRT even on uniform/blank images.
            tie = self._get_tie(n_total, pred_logits_s0.device)
            max_scores = pred_logits_s0.max(dim=-1).values.float() - tie
        topk_idx = torch.topk(max_scores, k, dim=1, sorted=True).indices  # [B, K]

        all_aligned = torch.cat(aligned_feats, dim=1)                     # [B, N, C_q]
        cq = all_aligned.shape[-1]
        q_raw = all_aligned.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, cq))
        b_cur = pred_boxes_s0.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))
        l_cur = pred_logits_s0.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, self.nc))
        if train:
            # Detach coarse boxes entering box_pe for clean iterative refinement stability
            b_pe_in = b_cur.detach()
        else:
            b_pe_in = b_cur

        q = self.query_norm(self.query_up(q_raw) + self.box_pe(b_pe_in))
        h_map = self._scale_body_grad(hyper_map)
        h_tok = self._scale_body_grad(hyper_tokens)

        dec_logits, dec_boxes = [], []
        for layer in self.decoder:
            q, b_cur, l_cur = layer(q, b_cur, l_cur, h_map, h_tok, attn_mask=None)
            dec_logits.append(l_cur)
            dec_boxes.append(b_cur)

        if not train:
            return torch.cat((l_cur.clamp(-25.0, 25.0), b_cur), dim=-1)

        result = {
            "pred_logits_o2o": pred_logits_s0,
            "pred_boxes_o2o": pred_boxes_s0,
            "anchor_points": torch.cat(anchors, dim=0),
            "stride_tensor": torch.cat(stride_rows, dim=0),
            "topk_indices": topk_idx,
            "dec_logits": dec_logits,
            "dec_boxes": dec_boxes,
            "hyper_map": hyper_map,
            "hyper_tokens": hyper_tokens,
        }
        if logits_m:
            result["pred_logits_o2m"] = torch.cat(logits_m, dim=1)
            result["pred_boxes_o2m"] = torch.cat(boxes_m, dim=1)
        if objectness:
            result["pred_obj_o2m"] = torch.cat(objectness, dim=1)
        return result


class V2HybridDetector(nn.Module):
    """fotonet2n (Hi Bird) end-to-end NMS-free hybrid detector."""

    def __init__(self, model_id: str = "fotonet2n", nc: int = 80, imgsz: int = 640, *,
                 stem_channels=None, p2_channels=None, stages=None, junctions=None, head_hidden=None,
                 neck_attention=None, neck_down=None, taps=None, reg_max=None,
                 stage_attn=None, hyper_ace=None, decoder_dim=None, decoder_layers=None,
                 decoder_queries=None, decoder_context=None, mlp_ratio=None,
                 o2m_mult: float = 1.0, o2o_groups: int = 1, aux_objectness: bool = False):
        super().__init__()
        spec = get_spec(model_id)
        if reg_max is not None and int(reg_max) != 1:
            raise ValueError("v2 uses direct LTRB regression; reg_max must be 1.")
        raw = {
            "stem_channels": spec.stem_channels if stem_channels is None else stem_channels,
            "p2_channels": spec.p2_channels if p2_channels is None else p2_channels,
            "stages": spec.stages if stages is None else stages,
            "junctions": spec.junctions if junctions is None else junctions,
            "head_hidden": spec.head_hidden if head_hidden is None else head_hidden,
            "neck_attention": spec.neck_attention if neck_attention is None else neck_attention,
            "neck_down": spec.neck_down if neck_down is None else neck_down,
            "taps": spec.taps if taps is None else taps,
            "stage_attn": spec.stage_attn if stage_attn is None else stage_attn,
            "hyper_ace": spec.hyper_ace if hyper_ace is None else hyper_ace,
            "decoder_dim": spec.decoder_dim if decoder_dim is None else decoder_dim,
            "decoder_layers": spec.decoder_layers if decoder_layers is None else decoder_layers,
            "decoder_queries": spec.decoder_queries if decoder_queries is None else decoder_queries,
            "decoder_context": spec.decoder_context if decoder_context is None else decoder_context,
            "mlp_ratio": spec.mlp_ratio if mlp_ratio is None else mlp_ratio,
        }
        layout = normalize_layout(raw)
        validate_layout(nc=int(nc), reg_max=1, **layout)

        self.spec = spec
        self.model_id = spec.model_id
        self.graph_version = GRAPH_VERSION
        self.nc = int(nc)
        self.reg_max = 1
        self.imgsz = int(imgsz)
        self.taps = False

        self.backbone = V2HybridBackbone(
            layout["stem_channels"], layout["p2_channels"], layout["stages"],
            layout["stage_attn"], mlp_ratio=layout["mlp_ratio"],
        )
        self.neck = V2HybridNeck(
            self.backbone.out_channels, layout["junctions"],
            hyper_ace=layout["hyper_ace"], neck_attention=layout["neck_attention"],
            mlp_ratio=layout["mlp_ratio"],
        )
        self.head = V2HybridHead(
            self.neck.out_channels, nc=self.nc, hidden=layout["head_hidden"], imgsz=self.imgsz,
            hyper_dim=self.neck.hyper_dim, decoder_dim=layout["decoder_dim"],
            decoder_layers=layout["decoder_layers"], num_queries=layout["decoder_queries"],
            decoder_context=layout["decoder_context"], aux_objectness=bool(aux_objectness),
        )
        self.model_config = {
            "model_id": spec.model_id,
            "graph_version": GRAPH_VERSION,
            "nc": self.nc,
            "reg_max": self.reg_max,
            "imgsz": self.imgsz,
            **{
                key: [list(row) for row in value] if key in ("stages", "junctions", "stage_attn")
                else list(value) if isinstance(value, tuple) else value
                for key, value in layout.items()
            },
            "o2m_mult": 1.0,
            "o2o_groups": 1,
            "aux_objectness": bool(aux_objectness),
        }
        self.architecture_fingerprint = architecture_fingerprint(self.model_config)
        self.model_config["architecture_fingerprint"] = self.architecture_fingerprint

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps, module.momentum = 1e-3, 0.03
            elif isinstance(module, AreaAttention):
                nn.init.zeros_(module.proj.bn.weight)
            elif isinstance(module, RELANBlock):
                nn.init.zeros_(module.ffn[1].bn.weight)
        self.head.init_biases()
        self._fused = False
        self.use_cuda_graph = True
        self._cg_cache: dict[tuple, tuple] = {}

    def _clear_cuda_graphs(self) -> None:
        if getattr(self, "_cg_cache", None):
            self._cg_cache.clear()

    def _apply(self, fn):
        self._clear_cuda_graphs()
        return super()._apply(fn)

    def train(self, mode: bool = True):
        if mode:
            self._clear_cuda_graphs()
        return super().train(mode)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self._clear_cuda_graphs()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @property
    def strides(self):
        return self.head.strides

    def set_imgsz(self, imgsz):
        if isinstance(imgsz, (tuple, list)):
            if int(imgsz[0]) != int(imgsz[1]):
                raise ValueError("v2 training image size must be square.")
            side = int(imgsz[0])
        else:
            side = int(imgsz)
        if side % FEATURE_STRIDES[-1]:
            raise ValueError(f"v2 image size must be divisible by {FEATURE_STRIDES[-1]}, got {side}.")
        self.imgsz = side
        self.head.imgsz = side
        self.model_config["imgsz"] = side
        return self

    def _features(self, x: torch.Tensor):
        feats = self.backbone(x)
        return self.neck(feats)

    def _forward_eval_eager(self, x: torch.Tensor) -> torch.Tensor:
        stride = FEATURE_STRIDES[-1]
        height, width = int(x.shape[-2]), int(x.shape[-1])
        pad_h = (stride - height % stride) % stride
        pad_w = (stride - width % stride) % stride
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=114.0 / 255.0)
        padded_h, padded_w = int(x.shape[-2]), int(x.shape[-1])
        outs, hyper_map, hyper_tokens = self._features(x)
        raw = self.head(outs, hyper_map, hyper_tokens, imgsz=(float(padded_w), float(padded_h)), use_o2m=False)
        boxes = raw[..., self.nc:]
        if pad_h or pad_w:
            boxes = boxes * boxes.new_tensor([
                padded_w / width, padded_h / height, padded_w / width, padded_h / height,
            ])
        boxes = torch.nan_to_num(_clip_norm_xywh(boxes), nan=0.0, posinf=1.0, neginf=0.0)
        return torch.cat((raw[..., :self.nc], boxes), dim=-1)

    def _forward_cuda_graph(self, x: torch.Tensor) -> torch.Tensor:
        key = (tuple(x.shape), x.dtype, x.device.index)
        entry = self._cg_cache.get(key)
        if entry is None:
            if len(self._cg_cache) >= 4:
                self._cg_cache.clear()
            static_in = x.contiguous().clone()
            # Warmup on a side stream so cuDNN/Triton JIT and allocator pools settle before capture
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _ = self._forward_eval_eager(static_in)
            torch.cuda.current_stream(x.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self._forward_eval_eager(static_in)
            entry = (graph, static_in, static_out)
            self._cg_cache[key] = entry
        graph, static_in, static_out = entry
        static_in.copy_(x)
        graph.replay()
        return static_out.clone()

    def forward(self, x: torch.Tensor, use_o2m: bool = True):
        if self.training:
            height, width = int(x.shape[-2]), int(x.shape[-1])
            if height != width or height % FEATURE_STRIDES[-1]:
                raise RuntimeError(
                    f"v2 training expects a square image size divisible by {FEATURE_STRIDES[-1]}, "
                    f"got {height}x{width}.")
            outs, hyper_map, hyper_tokens = self._features(x)
            return self.head(outs, hyper_map, hyper_tokens, imgsz=(float(width), float(height)), use_o2m=use_o2m)

        stride = FEATURE_STRIDES[-1]
        tracing = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        if tracing:
            static_exp = bool(getattr(self, "_static_export", False))
            if static_exp and (int(x.shape[-2]) % stride == 0) and (int(x.shape[-1]) % stride == 0):
                h_s, w_s = int(x.shape[-2]), int(x.shape[-1])
                outs, hyper_map, hyper_tokens = self._features(x)
                raw = self.head(outs, hyper_map, hyper_tokens, imgsz=(float(w_s), float(h_s)), use_o2m=False)
                boxes = _clip_norm_xywh(raw[..., self.nc:])
                return torch.cat((raw[..., :self.nc], boxes), dim=-1)
            shape = torch._shape_as_tensor(x)
            pad_h = torch.remainder(-shape[2], stride)
            pad_w = torch.remainder(-shape[3], stride)
            x = F.pad(x, (0, pad_w, 0, pad_h), value=114.0 / 255.0)
            padded = torch._shape_as_tensor(x)
            orig_wh = shape[[3, 2]].to(dtype=x.dtype)
            padded_wh = padded[[3, 2]].to(dtype=x.dtype)
            outs, hyper_map, hyper_tokens = self._features(x)
            raw = self.head(outs, hyper_map, hyper_tokens, imgsz=padded_wh, use_o2m=False)
            scale = (padded_wh / orig_wh.clamp(min=1)).to(device=raw.device, dtype=raw.dtype)
            boxes = raw[..., self.nc:] * torch.cat((scale, scale))
            boxes = torch.nan_to_num(_clip_norm_xywh(boxes), nan=0.0, posinf=1.0, neginf=0.0)
            return torch.cat((raw[..., :self.nc], boxes), dim=-1)

        if (
            self._fused
            and self.use_cuda_graph
            and x.is_cuda
            and not torch.is_grad_enabled()
            and not torch.cuda.is_current_stream_capturing()
            and not torch._dynamo.is_compiling()
        ):
            return self._forward_cuda_graph(x)
        return self._forward_eval_eager(x)

    def strip_o2m_for_inference(self):
        self._clear_cuda_graphs()
        self.head.strip_o2m_for_inference()
        return self

    @staticmethod
    def _fold_scale_into_conv(conv: nn.Conv2d, scale: torch.Tensor) -> None:
        s = scale.detach().to(device=conv.weight.device, dtype=conv.weight.dtype).reshape(-1)
        with torch.no_grad():
            conv.weight.mul_(s.view(-1, 1, 1, 1))
            if conv.bias is not None:
                conv.bias.mul_(s)

    @staticmethod
    def _fold_scale_into_linear(lin: nn.Linear, scale: torch.Tensor) -> None:
        s = scale.detach().to(device=lin.weight.device, dtype=lin.weight.dtype).reshape(-1)
        with torch.no_grad():
            lin.weight.mul_(s.view(-1, 1))
            if lin.bias is not None:
                lin.bias.mul_(s)

    def fuse(self):
        """Fold every RepConv into one 3x3, every Conv+BN pair, and linear residual gates (0 BatchNorm2d)."""
        if self.training:
            raise RuntimeError("Call model.eval() before fusing Conv-BN modules.")
        if self._fused:
            return self
        self._clear_cuda_graphs()
        for module in self.modules():
            if isinstance(module, RepConv):
                module.reparameterize()
        for module in self.modules():
            conv = getattr(module, "conv", None)
            bn = getattr(module, "bn", None)
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                module.conv = fuse_conv_bn(conv, bn)
                module.bn = nn.Identity()
        # Exact mathematical folding of learnable residual gates into preceding activation-free linear/conv ops
        for module in self.modules():
            if isinstance(module, RELANBlock):
                if not module._gamma_attn_folded and isinstance(module.attn.proj.conv, nn.Conv2d):
                    self._fold_scale_into_conv(module.attn.proj.conv, module.gamma_attn)
                    module._gamma_attn_folded = True
                if not module._gamma_ffn_folded and isinstance(module.ffn[1].conv, nn.Conv2d):
                    self._fold_scale_into_conv(module.ffn[1].conv, module.gamma_ffn)
                    module._gamma_ffn_folded = True
            elif isinstance(module, HyperACE):
                if not module._alphas_folded:
                    self._fold_scale_into_conv(module.dist_p3.conv, module.alpha3)
                    self._fold_scale_into_conv(module.dist_p4.conv, module.alpha4)
                    self._fold_scale_into_conv(module.dist_p5.conv, module.alpha5)
                    module._alphas_folded = True
            elif isinstance(module, GeoSALayer):
                if not module._gamma_ctx_folded:
                    target_lin = module.edge_proj if module.context_mode == "edge_sample" else module.ctx_out
                    self._fold_scale_into_linear(target_lin, module.gamma_ctx)
                    module._gamma_ctx_folded = True
        self._fused = True
        return self


__all__ = [
    "AreaAttention",
    "GeoSALayer",
    "HYBRID_TRAINING_ONLY_PREFIXES",
    "HybridStage0Branch",
    "HyperACE",
    "RELANBlock",
    "UnifiedRELANStage",
    "V2HybridBackbone",
    "V2HybridDetector",
    "V2HybridHead",
    "V2HybridNeck",
]
