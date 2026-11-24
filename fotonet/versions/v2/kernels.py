"""Custom Triton CUDA inference kernels for fotonet2n (Hi Bird).

All kernels are inference-only, zero-accuracy-loss exact equivalents of the reference PyTorch ops
in ``fotonet.versions.v2.hybrid``:
  1. ``fused_geo_bias``       : pairwise 5-D box relation + 2-layer SiLU MLP -> [B, H, Q, Q]
  2. ``fused_edge_sample``    : 4-edge-midpoint border-padded bilinear sampling -> [B, Q, 4*C]
  3. ``fused_box_refine``     : scale-equivariant residual cxcywh refinement -> [B, Q, 4]
  4. ``fused_stage0_decode``  : shifted-ELU + anchor grid + dist2bbox decode -> [B, H*W, 4]
  5. ``fused_scale_res_add``  : x + gamma * y channel-scaled residual add -> [B, C, H, W]

Each helper automatically falls back cleanly when Triton is unavailable, inputs are on CPU,
autograd is enabled, or ONNX/TorchScript tracing is active.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only / no-triton hosts
    triton = None
    tl = None
    _HAS_TRITON = False


def triton_available(tensor: torch.Tensor) -> bool:
    return (
        _HAS_TRITON
        and tensor.is_cuda
        and not torch.is_grad_enabled()
        and not torch.onnx.is_in_onnx_export()
        and not torch.jit.is_tracing()
    )


if _HAS_TRITON:

    @triton.jit
    def _fused_geo_bias_kernel(
        boxes_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
        Q,
        stride_bb, stride_bq,
        stride_ob, stride_oh, stride_oi, stride_oj,
        H_MID: tl.constexpr,
        H_OUT: tl.constexpr,
        BLOCK_J: tl.constexpr,
    ):
        """Each program computes one (b, i) query against BLOCK_J key boxes j."""
        pid_j = tl.program_id(0)
        pid_i = tl.program_id(1)
        pid_b = tl.program_id(2)

        j_offs = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
        j_mask = j_offs < Q

        # Query box i (scalars in registers)
        base_i = pid_b * stride_bb + pid_i * stride_bq
        cxi = tl.load(boxes_ptr + base_i + 0).to(tl.float32)
        cyi = tl.load(boxes_ptr + base_i + 1).to(tl.float32)
        wi = tl.maximum(tl.load(boxes_ptr + base_i + 2).to(tl.float32), 1e-4)
        hi = tl.maximum(tl.load(boxes_ptr + base_i + 3).to(tl.float32), 1e-4)
        x1i = cxi - 0.5 * wi
        y1i = cyi - 0.5 * hi
        x2i = cxi + 0.5 * wi
        y2i = cyi + 0.5 * hi
        areai = wi * hi
        inv_wi = 1.0 / wi
        inv_hi = 1.0 / hi

        # Key boxes j (vector [BLOCK_J] in registers)
        base_j = pid_b * stride_bb + j_offs * stride_bq
        cxj = tl.load(boxes_ptr + base_j + 0, mask=j_mask, other=0.0).to(tl.float32)
        cyj = tl.load(boxes_ptr + base_j + 1, mask=j_mask, other=0.0).to(tl.float32)
        wj = tl.maximum(tl.load(boxes_ptr + base_j + 2, mask=j_mask, other=1e-4).to(tl.float32), 1e-4)
        hj = tl.maximum(tl.load(boxes_ptr + base_j + 3, mask=j_mask, other=1e-4).to(tl.float32), 1e-4)

        dx = tl.minimum(tl.maximum((cxj - cxi) * inv_wi, -10.0), 10.0)
        dy = tl.minimum(tl.maximum((cyj - cyi) * inv_hi, -10.0), 10.0)
        dw = tl.minimum(tl.maximum(tl.log(wj * inv_wi), -4.0), 4.0)
        dh = tl.minimum(tl.maximum(tl.log(hj * inv_hi), -4.0), 4.0)

        x1j = cxj - 0.5 * wj
        y1j = cyj - 0.5 * hj
        x2j = cxj + 0.5 * wj
        y2j = cyj + 0.5 * hj
        ix1 = tl.maximum(x1i, x1j)
        iy1 = tl.maximum(y1i, y1j)
        ix2 = tl.minimum(x2i, x2j)
        iy2 = tl.minimum(y2i, y2j)
        inter = tl.maximum(ix2 - ix1, 0.0) * tl.maximum(iy2 - iy1, 0.0)
        iou = inter / (areai + wj * hj - inter + 1e-6)

        # Layer 1: Linear(5 -> H_MID) + SiLU, kept purely in registers [BLOCK_J, H_MID]
        m_offs = tl.arange(0, H_MID)
        b1 = tl.load(b1_ptr + m_offs).to(tl.float32)
        w1_0 = tl.load(w1_ptr + m_offs * 5 + 0).to(tl.float32)
        w1_1 = tl.load(w1_ptr + m_offs * 5 + 1).to(tl.float32)
        w1_2 = tl.load(w1_ptr + m_offs * 5 + 2).to(tl.float32)
        w1_3 = tl.load(w1_ptr + m_offs * 5 + 3).to(tl.float32)
        w1_4 = tl.load(w1_ptr + m_offs * 5 + 4).to(tl.float32)

        h1 = (
            b1[None, :]
            + dx[:, None] * w1_0[None, :]
            + dy[:, None] * w1_1[None, :]
            + dw[:, None] * w1_2[None, :]
            + dh[:, None] * w1_3[None, :]
            + iou[:, None] * w1_4[None, :]
        )
        h1 = h1 * tl.sigmoid(h1)

        # Layer 2: Linear(H_MID -> H_OUT) and direct write to [B, H_OUT, Q, Q]
        out_base = pid_b * stride_ob + pid_i * stride_oi + j_offs * stride_oj
        for h in tl.static_range(0, H_OUT):
            w2_h = tl.load(w2_ptr + h * H_MID + m_offs).to(tl.float32)
            b2_h = tl.load(b2_ptr + h).to(tl.float32)
            val = tl.sum(h1 * w2_h[None, :], axis=1) + b2_h
            tl.store(out_ptr + out_base + h * stride_oh, val, mask=j_mask)


    @triton.jit
    def _fused_edge_sample_kernel(
        boxes_ptr, fmap_ptr, out_ptr,
        Q, H, W,
        stride_bb, stride_bq,
        stride_fb, stride_fc, stride_fh, stride_fw,
        stride_ob, stride_oq,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Each program samples 1 of 4 edge midpoints for (b, q) across all C channels."""
        pid_pt = tl.program_id(0)   # 0:top, 1:bottom, 2:left, 3:right
        pid_q = tl.program_id(1)
        pid_b = tl.program_id(2)

        base_b = pid_b * stride_bb + pid_q * stride_bq
        cx = tl.load(boxes_ptr + base_b + 0).to(tl.float32)
        cy = tl.load(boxes_ptr + base_b + 1).to(tl.float32)
        hw = 0.5 * tl.load(boxes_ptr + base_b + 2).to(tl.float32)
        hh = 0.5 * tl.load(boxes_ptr + base_b + 3).to(tl.float32)

        # Match _edge_sample: pts_x=(cx, cx, cx-hw, cx+hw)*2-1, pts_y=(cy-hh, cy+hh, cy, cy)*2-1
        nx = tl.where(pid_pt == 2, cx - hw, tl.where(pid_pt == 3, cx + hw, cx)) * 2.0 - 1.0
        ny = tl.where(pid_pt == 0, cy - hh, tl.where(pid_pt == 1, cy + hh, cy)) * 2.0 - 1.0
        nx = tl.minimum(tl.maximum(nx, -1.0), 1.0)
        ny = tl.minimum(tl.maximum(ny, -1.0), 1.0)

        # Exact PyTorch grid_sample(align_corners=False, padding_mode='border') unnormalization:
        # ix = ((nx + 1) * W - 1) * 0.5, then border-clamp each corner index to [0, W-1]
        fx = ((nx + 1.0) * W - 1.0) * 0.5
        fy = ((ny + 1.0) * H - 1.0) * 0.5
        ix0 = tl.floor(fx).to(tl.int32)
        iy0 = tl.floor(fy).to(tl.int32)
        wx1 = fx - ix0.to(tl.float32)
        wy1 = fy - iy0.to(tl.float32)
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        cx0 = tl.minimum(tl.maximum(ix0, 0), W - 1)
        cx1 = tl.minimum(tl.maximum(ix0 + 1, 0), W - 1)
        cy0 = tl.minimum(tl.maximum(iy0, 0), H - 1)
        cy1 = tl.minimum(tl.maximum(iy0 + 1, 0), H - 1)

        c_offs = tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        fbase = pid_b * stride_fb + c_offs * stride_fc

        v00 = tl.load(fmap_ptr + fbase + cy0 * stride_fh + cx0 * stride_fw, mask=c_mask, other=0.0).to(tl.float32)
        v01 = tl.load(fmap_ptr + fbase + cy0 * stride_fh + cx1 * stride_fw, mask=c_mask, other=0.0).to(tl.float32)
        v10 = tl.load(fmap_ptr + fbase + cy1 * stride_fh + cx0 * stride_fw, mask=c_mask, other=0.0).to(tl.float32)
        v11 = tl.load(fmap_ptr + fbase + cy1 * stride_fh + cx1 * stride_fw, mask=c_mask, other=0.0).to(tl.float32)

        val = (v00 * wx0 + v01 * wx1) * wy0 + (v10 * wx0 + v11 * wx1) * wy1
        obase = pid_b * stride_ob + pid_q * stride_oq + pid_pt * C + c_offs
        tl.store(out_ptr + obase, val, mask=c_mask)


    @triton.jit
    def _fused_box_refine_kernel(
        boxes_ptr, delta_ptr, out_ptr, N,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        base = offs * 4

        cx = tl.load(boxes_ptr + base + 0, mask=mask, other=0.0).to(tl.float32)
        cy = tl.load(boxes_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        w = tl.maximum(tl.load(boxes_ptr + base + 2, mask=mask, other=1e-4).to(tl.float32), 1e-4)
        h = tl.maximum(tl.load(boxes_ptr + base + 3, mask=mask, other=1e-4).to(tl.float32), 1e-4)

        d0 = tl.load(delta_ptr + base + 0, mask=mask, other=0.0).to(tl.float32)
        d1 = tl.load(delta_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        d2 = tl.minimum(tl.maximum(tl.load(delta_ptr + base + 2, mask=mask, other=0.0).to(tl.float32), -2.0), 2.0)
        d3 = tl.minimum(tl.maximum(tl.load(delta_ptr + base + 3, mask=mask, other=0.0).to(tl.float32), -2.0), 2.0)

        cx_n = tl.minimum(tl.maximum(cx + d0 * w, 0.0), 1.0)
        cy_n = tl.minimum(tl.maximum(cy + d1 * h, 0.0), 1.0)
        w_n = tl.minimum(tl.maximum(w * tl.exp(d2), 1e-4), 1.0)
        h_n = tl.minimum(tl.maximum(h * tl.exp(d3), 1e-4), 1.0)

        tl.store(out_ptr + base + 0, cx_n, mask=mask)
        tl.store(out_ptr + base + 1, cy_n, mask=mask)
        tl.store(out_ptr + base + 2, w_n, mask=mask)
        tl.store(out_ptr + base + 3, h_n, mask=mask)


    @triton.jit
    def _fused_stage0_decode_kernel(
        reg_ptr, out_ptr,
        H, W, HW,
        stride_rb, stride_rc,
        stride_ob,
        stride_val, inv_w, inv_h,
        APPLY_SELU: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Decode [B, 4, H, W] -> normalized cxcywh [B, H*W, 4] with optional inline sELU."""
        pid_n = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs = pid_n * BLOCK + tl.arange(0, BLOCK)
        mask = offs < HW

        ay = (offs // W).to(tl.float32) + 0.5
        ax = (offs % W).to(tl.float32) + 0.5

        rbase = pid_b * stride_rb + offs
        l = tl.load(reg_ptr + rbase + 0 * stride_rc, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(reg_ptr + rbase + 1 * stride_rc, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(reg_ptr + rbase + 2 * stride_rc, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(reg_ptr + rbase + 3 * stride_rc, mask=mask, other=0.0).to(tl.float32)

        if APPLY_SELU:
            l = tl.where(l > 0.0, l + 1.0, tl.exp(l))
            t = tl.where(t > 0.0, t + 1.0, tl.exp(t))
            r = tl.where(r > 0.0, r + 1.0, tl.exp(r))
            b = tl.where(b > 0.0, b + 1.0, tl.exp(b))

        x1 = (ax - l) * stride_val
        y1 = (ay - t) * stride_val
        x2 = (ax + r) * stride_val
        y2 = (ay + b) * stride_val

        cx = (x1 + x2) * (0.5 * inv_w)
        cy = (y1 + y2) * (0.5 * inv_h)
        bw = tl.maximum(x2 - x1, 0.0) * inv_w
        bh = tl.maximum(y2 - y1, 0.0) * inv_h

        obase = pid_b * stride_ob + offs * 4
        tl.store(out_ptr + obase + 0, cx, mask=mask)
        tl.store(out_ptr + obase + 1, cy, mask=mask)
        tl.store(out_ptr + obase + 2, bw, mask=mask)
        tl.store(out_ptr + obase + 3, bh, mask=mask)


    @triton.jit
    def _fused_scale_res_add_kernel(
        x_ptr, y_ptr, g_ptr, out_ptr,
        C, HW, N,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        c = (offs // HW) % C
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + c, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, x + g * y, mask=mask)


def fused_geo_bias(boxes: torch.Tensor, geo_mlp: torch.nn.Sequential) -> torch.Tensor | None:
    """Compute pairwise 5-D box relation + geo_mlp -> [B, H, Q, Q] in one Triton kernel."""
    if not triton_available(boxes):
        return None
    l1, l2 = geo_mlp[0], geo_mlp[2]
    h_mid = int(l1.out_features)
    h_out = int(l2.out_features)
    if h_mid != 16 or h_out not in (2, 4, 8):
        return None
    b, q, _ = boxes.shape
    boxes_c = boxes.contiguous()
    w1 = l1.weight.contiguous()
    b1 = l1.bias.contiguous()
    w2 = l2.weight.contiguous()
    b2 = l2.bias.contiguous()
    out = torch.empty((b, h_out, q, q), device=boxes.device, dtype=boxes.dtype)
    block_j = 64
    grid = (triton.cdiv(q, block_j), q, b)
    _fused_geo_bias_kernel[grid](
        boxes_c, w1, b1, w2, b2, out,
        q,
        boxes_c.stride(0), boxes_c.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H_MID=16, H_OUT=h_out, BLOCK_J=block_j,
    )
    return out


def fused_edge_sample(boxes: torch.Tensor, hyper_map: torch.Tensor) -> torch.Tensor | None:
    """Sample 4 box edge midpoints from hyper_map [B, C, H, W] -> [B, Q, 4*C] in one Triton kernel."""
    if not triton_available(boxes):
        return None
    b, c, h, w = hyper_map.shape
    q = boxes.shape[1]
    block_c = triton.next_power_of_2(c)
    if block_c > 256:
        return None
    boxes_c = boxes.contiguous()
    fmap_c = hyper_map.contiguous()
    out = torch.empty((b, q, 4 * c), device=boxes.device, dtype=hyper_map.dtype)
    grid = (4, q, b)
    _fused_edge_sample_kernel[grid](
        boxes_c, fmap_c, out,
        q, h, w,
        boxes_c.stride(0), boxes_c.stride(1),
        fmap_c.stride(0), fmap_c.stride(1), fmap_c.stride(2), fmap_c.stride(3),
        out.stride(0), out.stride(1),
        C=c, BLOCK_C=block_c,
    )
    return out


def fused_box_refine(boxes: torch.Tensor, delta: torch.Tensor) -> torch.Tensor | None:
    """Scale-equivariant box refinement [B, Q, 4] in one Triton kernel."""
    if not triton_available(boxes):
        return None
    boxes_c = boxes.contiguous()
    delta_c = delta.contiguous()
    n = boxes_c.numel() // 4
    out = torch.empty_like(boxes_c)
    block = 128
    _fused_box_refine_kernel[(triton.cdiv(n, block),)](boxes_c, delta_c, out, n, BLOCK=block)
    return out


def fused_stage0_decode(reg: torch.Tensor, stride: float, imgsz_w: float, imgsz_h: float,
                        apply_selu: bool = False) -> torch.Tensor | None:
    """Decode [B, 4, H, W] -> normalized cxcywh [B, H*W, 4] in one Triton kernel."""
    if not triton_available(reg):
        return None
    b, _, h, w = reg.shape
    hw = h * w
    reg_c = reg.contiguous()
    out = torch.empty((b, hw, 4), device=reg.device, dtype=reg.dtype)
    block = 256
    grid = (triton.cdiv(hw, block), b)
    _fused_stage0_decode_kernel[grid](
        reg_c, out,
        h, w, hw,
        reg_c.stride(0), reg_c.stride(1),
        out.stride(0),
        float(stride), 1.0 / float(imgsz_w), 1.0 / float(imgsz_h),
        APPLY_SELU=apply_selu, BLOCK=block,
    )
    return out


def fused_scale_res_add(x: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor | None:
    """Compute x + gamma * y for [B, C, H, W] tensors in one Triton kernel."""
    if not triton_available(x) or x.ndim != 4 or x.shape != y.shape:
        return None
    x_c = x.contiguous()
    y_c = y.contiguous()
    g_c = gamma.reshape(-1).contiguous()
    out = torch.empty_like(x_c)
    c = x_c.shape[1]
    hw = x_c.shape[2] * x_c.shape[3]
    n = x_c.numel()
    block = 512
    _fused_scale_res_add_kernel[(triton.cdiv(n, block),)](x_c, y_c, g_c, out, c, hw, n, BLOCK=block)
    return out


__all__ = [
    "fused_box_refine",
    "fused_edge_sample",
    "fused_geo_bias",
    "fused_scale_res_add",
    "fused_stage0_decode",
    "triton_available",
]
