"""GPU-side training augmentation (compiled, shape-static; first used by fotonet v1.1).

Moves the post-mosaic chain off the CPU workers onto the (idle) GPU:
random affine (rotate/scale/translate/shear, cv2-faithful matrix math),
HSV jitter, flips, and BGR swap — batched over the whole batch and
fused by torch.compile.  Randomness is sampled eagerly and passed in,
every tensor shape is static (boxes padded to ``mmax``), and there are
no data-dependent branches: one compile, no recompiles.

Workers keep only what genuinely needs the CPU: JPEG decode, resize,
mosaic paste + crop.

Semantic deltas vs the shared CPU path (deliberate, recorded):
- affine runs a single pass (no min-label-keep retry); the area-ratio
  keep filter (wh > 2px, area ratio > 0.1) is identical.
- HSV is computed in float instead of uint8 LUTs (sub-rounding noise).

Same distributions as the CPU path: the hue shift is in OpenCV hue units
(gain * hsv_h * 180 of a 0-180 scale, i.e. twice that in degrees), and x and
y shear are drawn independently.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _augment_core(imgs_u8, boxes_pad, valid, cos, sin, scale, shear, tx, ty,
                  hsv_gains, mh, mv, mb, po, H, W):
    """Deterministic, shape-static augmentation core (compiled)."""
    B = imgs_u8.shape[0]
    ones = torch.ones_like(cos)
    rs = torch.stack([cos * scale, -sin * scale,
                      sin * scale, cos * scale], -1).view(B, 2, 2)
    sh = torch.stack([ones, shear[:, 0], shear[:, 1], ones], -1).view(B, 2, 2)
    lin = sh @ rs                                     # [B,2,2] forward
    tr = torch.stack((tx, ty), -1)                    # [B,2]
    lin_inv = torch.linalg.inv(lin)

    # Center the affine about the image middle (external-review fix):
    # x' = A(x - c) + c + t  ==  A x + (t + c - A c).  Without this the
    # transform scaled/rotated about the TOP-LEFT corner — pixels and
    # boxes stayed mutually consistent, but the augmentation distribution
    # shifted ~180px at scale 1.5625 relative to the intended (CPU-path)
    # semantics: zoom-ins pushed content out the right/bottom edge and
    # zoom-outs piled it toward the top-left.
    ctr = torch.tensor([W * 0.5, H * 0.5], device=lin.device,
                       dtype=lin.dtype)                        # [2]
    tr = tr + ctr - torch.einsum("bij,j->bi", lin, ctr)      # [B,2]

    pi = torch.einsum("bij,pj->bpi", lin_inv, po) \
        - torch.einsum("bij,bj->bi", lin_inv, tr).unsqueeze(1)
    x = imgs_u8.float()
    # Sample through a 1px apron of 114 so border blends meet the fill color
    # instead of zero-padding (a dark 1px row/column seam used to appear
    # wherever the source window's edge crossed the canvas).  Border
    # replication beyond the apron reproduces cv2 warpAffine's constant-114
    # border semantics without any data-dependent branch.
    xp = F.pad(x, (1, 1, 1, 1), mode="constant", value=114.0)
    gx = (2.0 * (pi[..., 0] + 1.0) + 1.0) / (W + 2) - 1.0
    gy = (2.0 * (pi[..., 1] + 1.0) + 1.0) / (H + 2) - 1.0
    grid = torch.stack((gx, gy), -1).view(B, H, W, 2)
    warped = F.grid_sample(xp, grid, mode="bilinear",
                           padding_mode="border", align_corners=False)
    x = warped

    # ---- boxes through the FORWARD matrices ----
    cx, cy = boxes_pad[..., 0] * W, boxes_pad[..., 1] * H
    bw, bh = boxes_pad[..., 2] * W, boxes_pad[..., 3] * H
    corners = torch.stack([
        torch.stack([cx - bw * 0.5, cy - bh * 0.5], -1),
        torch.stack([cx + bw * 0.5, cy - bh * 0.5], -1),
        torch.stack([cx - bw * 0.5, cy + bh * 0.5], -1),
        torch.stack([cx + bw * 0.5, cy + bh * 0.5], -1),
    ], 2)                                              # [B,N,4,2]
    mapped = torch.einsum("bij,bnj->bni", lin,
                          corners.reshape(B, -1, 2)) + tr.unsqueeze(1)
    mapped = mapped.view(B, -1, 4, 2)
    x1 = mapped[..., 0].amin(-1).clamp(0, W)
    y1 = mapped[..., 1].amin(-1).clamp(0, H)
    x2 = mapped[..., 0].amax(-1).clamp(0, W)
    y2 = mapped[..., 1].amax(-1).clamp(0, H)
    nw = (x2 - x1).clamp(min=0)
    nh = (y2 - y1).clamp(min=0)
    keep = valid & (nw > 2.0) & (nh > 2.0) & \
        ((nw * nh) / (bw.clamp(min=1e-6) * bh.clamp(min=1e-6)) > 0.1)
    boxes_out = torch.stack([
        ((x1 + x2) * 0.5) / W,
        ((y1 + y2) * 0.5) / H,
        nw / W,
        nh / H,
    ], -1)

    # ---- HSV jitter (per-sample gains, fused) ----
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc = x.max(1).values
    minc = x.min(1).values
    v = maxc
    delta = maxc - minc + 1e-6
    s = delta / (maxc + 1e-6)
    sector = torch.where(
        maxc == r, ((g - b) / delta) % 6.0,
        torch.where(maxc == g, (b - r) / delta + 2.0,
                    (r - g) / delta + 4.0))
    # OpenCV hue units are 2 degrees: a shift of g*180 units is g*360 degrees.
    hue = (sector * 60.0 + hsv_gains[0] * 360.0) % 360.0
    s = ((s * 255.0) * (1.0 + hsv_gains[1])).clamp(0, 255) / 255.0
    v = (v * (1.0 + hsv_gains[2])).clamp(0, 255)
    c = v * s
    hp = hue / 60.0
    sext = hp.floor().clamp(0, 5)
    frac = hp - sext
    p = v - c
    q = v - c * frac
    tcol = v - c * (1.0 - frac)
    # standard HSV->RGB table: (V,T,P / Q,V,P / P,V,T / P,Q,V / T,P,V / V,P,Q)
    r = torch.where(sext == 0, v, torch.where(sext == 1, q,
        torch.where(sext == 2, p, torch.where(sext == 3, p,
        torch.where(sext == 4, tcol, v)))))
    g = torch.where(sext == 0, tcol, torch.where(sext == 1, v,
        torch.where(sext == 2, v, torch.where(sext == 3, q,
        torch.where(sext == 4, p, p)))))
    b = torch.where(sext == 0, p, torch.where(sext == 1, p,
        torch.where(sext == 2, tcol, torch.where(sext == 3, v,
        torch.where(sext == 4, v, q)))))
    x = torch.stack((r, g, b), 1)

    # ---- flips + BGR swap (branch-free, full-batch where) ----
    if4 = lambda m: m.view(B, 1, 1, 1)
    x = torch.where(if4(mh), x.flip(-1), x)
    boxes_out = torch.where(
        mh.view(B, 1, 1),
        torch.stack((1.0 - boxes_out[..., 0], boxes_out[..., 1],
                     boxes_out[..., 2], boxes_out[..., 3]), -1),
        boxes_out)
    x = torch.where(if4(mv), x.flip(-2), x)
    boxes_out = torch.where(
        mv.view(B, 1, 1),
        torch.stack((boxes_out[..., 0], 1.0 - boxes_out[..., 1],
                     boxes_out[..., 2], boxes_out[..., 3]), -1),
        boxes_out)
    if3 = mb.view(B, 1, 1, 1)
    x = torch.where(if3, x[:, [2, 1, 0]], x)

    x = x.div(255.0)
    boxes_out = boxes_out.clamp(0.0, 1.0)
    return x, boxes_out, keep


class GpuAugmenter:
    """Stateful batch augmenter matching the v1 recipe knobs."""

    def __init__(self, hyp, device, imgsz=640, mmax=48, compile_core=True):
        self.hyp = dict(hyp)
        self.device = torch.device(device)
        self.imgsz = int(imgsz)
        self.mmax = int(mmax)
        self.generator = torch.Generator(device=self.device)
        self._po = None
        # Reusable HOST-PINNED staging for GT (see __call__).  Allocating
        # [B, mmax, 4] plus masks on the CPU every step and then copying up
        # was re-introducing the host traffic this augmenter exists to remove.
        # It is a RING, not a single buffer: a non_blocking H2D reads the
        # pinned source asynchronously, so recycling a slot too soon would let
        # the CPU overwrite bytes the DMA engine has not read yet.  The ring
        # alone is not enough: a training step with no host sync lets the CPU
        # run more than STAGING_RING steps ahead of the GPU. Each slot
        # therefore records an event after its copies and waits on it before
        # being rewritten (a no-op unless that copy is genuinely pending).
        self._ring = []
        self._ring_events = []
        self._ring_i = 0
        self._slot = 0
        self._compiled = bool(compile_core)
        self._core = (torch.compile(_augment_core, dynamic=False)
                      if compile_core else _augment_core)

    STAGING_RING = 4

    def _staging(self, batch):
        """(boxes, labels, valid) pinned host views sized for ``batch``."""
        if not self._ring or self._ring[0][0].shape[0] < batch:
            cap = max(batch, self.mmax)
            self._ring = [
                (torch.zeros(cap, self.mmax, 4, dtype=torch.float32, pin_memory=True),
                 torch.zeros(cap, self.mmax, dtype=torch.long, pin_memory=True),
                 torch.zeros(cap, self.mmax, dtype=torch.bool, pin_memory=True))
                for _ in range(self.STAGING_RING)
            ]
            self._ring_events = [None] * self.STAGING_RING
            self._ring_i = 0
        self._slot = self._ring_i
        pending = self._ring_events[self._slot]
        if pending is not None:
            pending.synchronize()
        slot = self._ring[self._slot]
        self._ring_i = (self._ring_i + 1) % len(self._ring)
        return slot

    def _pixel_grid(self, H, W):
        if self._po is None or self._po.shape[0] != H * W:
            ys, xs = torch.meshgrid(
                torch.arange(H, device=self.device, dtype=torch.float32),
                torch.arange(W, device=self.device, dtype=torch.float32),
                indexing="ij")
            self._po = torch.stack((xs.reshape(-1), ys.reshape(-1)), -1)
        return self._po

    @torch.no_grad()
    def __call__(self, imgs_u8, targets, labels_pad=None):
        """imgs_u8 [B,3,H,W] uint8 (any device -> moved to self.device);
        targets = list of B dicts with 'boxes'/'labels'; labels_pad may be
        pre-supplied by the trainer.

        Returns (x [B,3,H,W] float in [0,1], boxes [B,mmax,4] xywh-norm,
        labels [B,mmax] long, valid [B,mmax] bool) — all on device.
        """
        dev = self.device
        B = imgs_u8.shape[0]
        M = self.mmax
        gen = self.generator
        hyp = self.hyp

        stage_boxes, stage_labels, stage_valid = self._staging(B)
        boxes_pad = stage_boxes[:B, :M]
        labels_pad_out = stage_labels[:B, :M]
        valid = stage_valid[:B, :M]
        boxes_pad.zero_()
        valid.zero_()
        if labels_pad is None:
            labels_pad_out.zero_()
        for b, t in enumerate(targets):
            n = min(t["boxes"].shape[0], M)
            if n:
                boxes_pad[b, :n].copy_(t["boxes"][:n])
                if labels_pad is None:
                    labels_pad_out[b, :n].copy_(t["labels"][:n])
                valid[b, :n] = True
        # One pinned->device DMA per tensor instead of a fresh CPU allocation
        # plus B host-side copies every step.
        boxes_pad = boxes_pad.to(dev, non_blocking=True)
        valid = valid.to(dev, non_blocking=True)
        if labels_pad is None:
            labels_pad = labels_pad_out.to(dev, non_blocking=True)
        else:
            labels_pad = torch.as_tensor(labels_pad).to(dev, non_blocking=True)
        if dev.type == "cuda":
            done = torch.cuda.Event(blocking=True)   # waits sleep, not spin
            done.record()
            self._ring_events[self._slot] = done
        imgs_u8 = imgs_u8.to(dev, non_blocking=True)

        def U(lo, hi):
            return lo + (hi - lo) * torch.rand(B, device=dev, generator=gen)

        deg = float(hyp.get("degrees", 0.0))
        trn = float(hyp.get("translate", 0.0))
        scl = float(hyp.get("scale", 0.0))
        shr = float(hyp.get("shear", 0.0))
        angle = U(-deg, deg) * math.pi / 180.0
        scale = U(max(1.0 - scl, 0.05), 1.0 + scl)
        shear = torch.stack((torch.tan(U(-shr, shr) * math.pi / 180.0),
                             torch.tan(U(-shr, shr) * math.pi / 180.0)), -1)
        tx = U(-trn, trn) * imgs_u8.shape[-1]
        ty = U(-trn, trn) * imgs_u8.shape[-2]
        hg = float(hyp.get("hsv_h", 0.0))
        sgn = float(hyp.get("hsv_s", 0.0))
        vgn = float(hyp.get("hsv_v", 0.0))
        hsv_gains = torch.rand(3, B, 1, 1, device=dev, generator=gen) * 2.0 - 1.0
        key = (hg, sgn, vgn, str(dev))
        if getattr(self, "_hsv_key", None) != key:
            # Built once, not every step: a host list to the GPU each call is
            # a small synchronous copy.
            self._hsv_scale = torch.tensor([hg, sgn, vgn], device=dev).view(3, 1, 1, 1)
            self._hsv_key = key
        hsv_gains = hsv_gains * self._hsv_scale
        mh = (torch.rand(B, device=dev, generator=gen)
              < float(hyp.get("fliplr", 0.0)))
        mv = (torch.rand(B, device=dev, generator=gen)
              < float(hyp.get("flipud", 0.0)))
        mb = (torch.rand(B, device=dev, generator=gen)
              < float(hyp.get("bgr", 0.0)))

        H, W = imgs_u8.shape[-2], imgs_u8.shape[-1]
        po = self._pixel_grid(H, W)
        args = (imgs_u8, boxes_pad, valid, torch.cos(angle), torch.sin(angle),
                scale, shear, tx, ty, hsv_gains, mh, mv, mb, po, H, W)
        try:
            x, boxes_out, keep = self._core(*args)
        except Exception as exc:
            if not self._compiled:
                raise
            # A compiler problem must not stop training: fall back to eager.
            import warnings
            warnings.warn(f"compiled GPU augmentation failed ({type(exc).__name__}: {exc}); using eager.")
            self._compiled = False
            self._core = _augment_core
            x, boxes_out, keep = self._core(*args)
        return x, boxes_out, labels_pad, keep


__all__ = ["GpuAugmenter"]
