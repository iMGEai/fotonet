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
import random

import torch
import torch.nn.functional as F


def _augment_core(imgs_u8, boxes_pad, valid, cos, sin, scale, shear, tx, ty,
                  hsv_gains, mh, mv, mb, po, H, W, channels_last=False, out_dtype=torch.float32):
    """Deterministic, shape-static augmentation core (compiled).

    Written so the compiler can produce the image in one pass: the flips are applied to the
    sampling grid instead of to the finished image, the channel max and min are elementwise, and
    the channel swap and the division by 255 happen where the output is written. ``channels_last``
    and ``out_dtype`` choose the layout and type of that output; a bfloat16 output holds the
    values autocast would make of the float32 one at the model's first convolution.
    """
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
    # Flipping the image after the warp equals sampling it through the flipped grid (the same
    # numbers land at the mirrored pixel), and needs no second pass over the image.
    grid = torch.where(mh.view(B, 1, 1, 1), grid.flip(2), grid)
    grid = torch.where(mv.view(B, 1, 1, 1), grid.flip(1), grid)
    warped = F.grid_sample(xp, grid, mode="bilinear",
                           padding_mode="border", align_corners=False)

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
    r, g, b = warped[:, 0], warped[:, 1], warped[:, 2]
    maxc = torch.maximum(torch.maximum(r, g), b)
    minc = torch.minimum(torch.minimum(r, g), b)
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

    # ---- flips (the image was flipped through its grid) + BGR swap, branch-free ----
    boxes_out = torch.where(
        mh.view(B, 1, 1),
        torch.stack((1.0 - boxes_out[..., 0], boxes_out[..., 1],
                     boxes_out[..., 2], boxes_out[..., 3]), -1),
        boxes_out)
    boxes_out = torch.where(
        mv.view(B, 1, 1),
        torch.stack((boxes_out[..., 0], 1.0 - boxes_out[..., 1],
                     boxes_out[..., 2], boxes_out[..., 3]), -1),
        boxes_out)
    swap = mb.view(B, 1, 1)
    channels = (torch.where(swap, b, r), g, torch.where(swap, r, b))
    if channels_last:
        x = torch.stack(channels, -1).div(255.0).permute(0, 3, 1, 2)
    else:
        x = torch.stack(channels, 1).div(255.0)
    boxes_out = boxes_out.clamp(0.0, 1.0)
    return x.to(out_dtype), boxes_out, keep


class GpuAugmenter:
    """Stateful batch augmenter matching the v1 recipe knobs."""

    def __init__(self, hyp, device, imgsz=640, mmax=48, compile_core=True, channels_last=False,
                 out_dtype=torch.float32):
        self.hyp = dict(hyp)
        # Layout and type of the output images (see _augment_core).
        self.channels_last = bool(channels_last)
        self.out_dtype = out_dtype
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

        Returns (x [B,3,H,W] in [0,1] (out_dtype, channels-last if asked), boxes [B,mmax,4] xywh-norm,
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
        pre = [bool(t.get("pre_zoom", False)) for t in targets]
        if any(pre):
            # aug.mosaic@1.2 already zoomed and shifted these mosaics on the whole canvas, in the worker.
            done = torch.tensor(pre, device=dev)
            scale = torch.where(done, torch.ones_like(scale), scale)
            tx = torch.where(done, torch.zeros_like(tx), tx)
            ty = torch.where(done, torch.zeros_like(ty), ty)
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
                scale, shear, tx, ty, hsv_gains, mh, mv, mb, po, H, W, self.channels_last, self.out_dtype)
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

        cp_prob = float(hyp.get("copy_paste", 0.0))
        if cp_prob > 0.0 and B > 1:
            x, boxes_out, labels_pad, keep = gpu_batch_copy_paste(
                x, boxes_out, labels_pad, keep,
                prob=cp_prob,
                max_boxes=M,
                generator=gen,
            )

        mixup_prob = float(hyp.get("mixup", 0.0))
        if mixup_prob > 0.0 and B > 1 and random.random() < mixup_prob:
            x, boxes_out, labels_pad, keep, _ = gpu_batch_mixup(
                x, boxes_out, labels_pad, keep,
                alpha=float(hyp.get("mixup_alpha", 32.0)),
                max_boxes=M,
            )
        return x, boxes_out, labels_pad, keep


def gpu_batch_mixup(images: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor,
                     alpha: float = 32.0, max_boxes: int | None = None):
    """Tensor-level GPU batch MixUp without CPU host roundtrips.

    images   [B, C, H, W] in [0, 1] (or uint8)
    boxes    [B, M, 4] normalized cxcywh
    labels   [B, M] class ids
    valid    [B, M] bool mask of valid boxes
    alpha    Beta distribution concentration parameter
    max_boxes optional maximum box count to keep (pads or truncates to max_boxes)

    Returns ``(images_mix, boxes_mix, labels_mix, valid_mix, weights_mix)``.
    """
    B = images.shape[0]
    if B <= 1:
        w_dtype = images.dtype if images.is_floating_point() else torch.float32
        ones_w = valid.to(w_dtype)
        return images, boxes, labels, valid, ones_w

    perm = torch.randperm(B, device=images.device)
    a = float(alpha)
    if a > 0:
        a_clamped = min(a, 1e4)
        lam = float(random.betavariate(a_clamped, a_clamped))
    else:
        lam = 0.5

    # Blend while preserving channels_last memory format if active
    is_cl = images.ndim == 4 and images.is_contiguous(memory_format=torch.channels_last)
    partner = images[perm]
    if is_cl and not partner.is_contiguous(memory_format=torch.channels_last):
        partner = partner.contiguous(memory_format=torch.channels_last)
    if images.is_floating_point():
        images_mix = torch.lerp(partner, images, lam)
    else:
        images_mix = torch.lerp(partner.float(), images.float(), lam).to(images.dtype)

    boxes_mix = torch.cat([boxes, boxes[perm]], dim=1)
    labels_mix = torch.cat([labels, labels[perm]], dim=1)
    valid_mix = torch.cat([valid, valid[perm]], dim=1)
    w_dtype = images.dtype if images.is_floating_point() else torch.float32
    weights_mix = torch.cat([lam * valid.to(w_dtype), (1.0 - lam) * valid[perm].to(w_dtype)], dim=1)

    if max_boxes is not None and boxes_mix.shape[1] > max_boxes:
        order = valid_mix.to(torch.uint8).argsort(dim=1, descending=True, stable=True)[:, :max_boxes]
        boxes_mix = boxes_mix.gather(1, order[..., None].expand(-1, -1, 4))
        labels_mix = labels_mix.gather(1, order)
        valid_mix = valid_mix.gather(1, order)
        weights_mix = weights_mix.gather(1, order)

    return images_mix, boxes_mix, labels_mix, valid_mix, weights_mix


def gpu_batch_copy_paste(images: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor,
                         prob: float = 0.15, max_paste: int = 3, max_area: float = 0.25,
                         max_overlap: float = 0.10, max_boxes: int | None = None,
                         generator: torch.Generator | None = None):
    """Zero-host-sync tensor-level GPU Box Copy-Paste.

    For each triggered image ``b`` in the batch, selects up to ``max_paste`` small/medium non-crowd
    objects from a distinct partner image ``perm[b]`` whose bounding boxes do not overlap any existing
    object in ``b`` (intersection / area < ``max_overlap``), pastes their exact pixel rectangles on
    GPU via a vectorized broadcast mask, and appends their boxes/labels while preserving static
    ``[B, max_boxes]`` shapes and ``channels_last`` layout.
    """
    B, _, H, W = images.shape
    M = boxes.shape[1]
    k_paste = min(int(max_paste), M)
    if B <= 1 or float(prob) <= 0.0 or k_paste <= 0:
        return images, boxes, labels, valid

    dev = images.device
    shift = random.randint(1, B - 1)
    perm = (torch.arange(B, device=dev) + shift) % B
    active = torch.rand(B, device=dev, generator=generator) < float(prob)

    p_boxes = boxes[perm]     # [B, M, 4] cxcywh norm
    p_labels = labels[perm]   # [B, M]
    p_valid = valid[perm]     # [B, M]

    # Candidate filter: valid, non-crowd (label < 10_000), small/medium area, >= 8px wide & tall
    pw = p_boxes[..., 2]
    ph = p_boxes[..., 3]
    p_area = pw * ph
    min_w = 8.0 / max(float(W), 1.0)
    min_h = 8.0 / max(float(H), 1.0)
    cand = p_valid & (p_labels < 10_000) & (p_area <= float(max_area)) & (pw >= min_w) & (ph >= min_h)

    # Check overlap against all existing valid boxes in destination image b (avoid occluding real GT)
    bx1 = boxes[..., 0] - 0.5 * boxes[..., 2]
    by1 = boxes[..., 1] - 0.5 * boxes[..., 3]
    bx2 = boxes[..., 0] + 0.5 * boxes[..., 2]
    by2 = boxes[..., 1] + 0.5 * boxes[..., 3]
    b_area = (boxes[..., 2] * boxes[..., 3]).clamp(min=1e-7)

    px1 = p_boxes[..., 0] - 0.5 * pw
    py1 = p_boxes[..., 1] - 0.5 * ph
    px2 = p_boxes[..., 0] + 0.5 * pw
    py2 = p_boxes[..., 1] + 0.5 * ph

    ix1 = torch.maximum(px1[:, :, None], bx1[:, None, :])
    iy1 = torch.maximum(py1[:, :, None], by1[:, None, :])
    ix2 = torch.minimum(px2[:, :, None], bx2[:, None, :])
    iy2 = torch.minimum(py2[:, :, None], by2[:, None, :])
    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)   # [B, M_p, M_b]
    occ_dst = inter / b_area[:, None, :]
    occ_src = inter / p_area.clamp(min=1e-7)[:, :, None]
    bad_overlap = ((occ_dst > float(max_overlap)) | (occ_src > float(max_overlap))) & valid[:, None, :]
    cand = cand & (~bad_overlap.any(dim=2)) & active[:, None]

    # Random priority slightly favoring smaller objects (boosts AP_s)
    rand_prio = torch.rand(B, M, device=dev, generator=generator) + 0.25 * (1.0 - p_area.clamp(0.0, 1.0))
    prio = torch.where(cand, rand_prio, rand_prio.new_full((), -1.0))
    top_val, top_idx = prio.topk(k_paste, dim=1)                      # [B, K]
    sel_valid = top_val > 0.0                                         # [B, K]
    sel_boxes = p_boxes.gather(1, top_idx[..., None].expand(-1, -1, 4))
    sel_labels = p_labels.gather(1, top_idx)

    # Vectorized rectangular pixel mask [B, 1, H, W] -> zero loops, zero host sync
    sx1 = ((sel_boxes[..., 0] - 0.5 * sel_boxes[..., 2]) * W).clamp(0.0, float(W))
    sy1 = ((sel_boxes[..., 1] - 0.5 * sel_boxes[..., 3]) * H).clamp(0.0, float(H))
    sx2 = ((sel_boxes[..., 0] + 0.5 * sel_boxes[..., 2]) * W).clamp(0.0, float(W))
    sy2 = ((sel_boxes[..., 1] + 0.5 * sel_boxes[..., 3]) * H).clamp(0.0, float(H))

    xs = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, 1, W) + 0.5
    ys = torch.arange(H, device=dev, dtype=torch.float32).view(1, 1, H, 1) + 0.5
    mx = (xs >= sx1[:, :, None, None]) & (xs < sx2[:, :, None, None])   # [B, K, 1, W]
    my = (ys >= sy1[:, :, None, None]) & (ys < sy2[:, :, None, None])   # [B, K, H, 1]
    paste_mask = ((mx & my) & sel_valid[:, :, None, None]).any(dim=1, keepdim=True)  # [B, 1, H, W]

    is_cl = images.ndim == 4 and images.is_contiguous(memory_format=torch.channels_last)
    partner_imgs = images[perm]
    out_imgs = torch.where(paste_mask, partner_imgs, images)
    if is_cl and not out_imgs.is_contiguous(memory_format=torch.channels_last):
        out_imgs = out_imgs.contiguous(memory_format=torch.channels_last)

    boxes_cat = torch.cat([boxes, sel_boxes], dim=1)
    labels_cat = torch.cat([labels, sel_labels], dim=1)
    valid_cat = torch.cat([valid, sel_valid], dim=1)

    if max_boxes is not None and boxes_cat.shape[1] > max_boxes:
        order = valid_cat.to(torch.uint8).argsort(dim=1, descending=True, stable=True)[:, :max_boxes]
        boxes_cat = boxes_cat.gather(1, order[..., None].expand(-1, -1, 4))
        labels_cat = labels_cat.gather(1, order)
        valid_cat = valid_cat.gather(1, order)

    return out_imgs, boxes_cat, labels_cat, valid_cat


__all__ = ["GpuAugmenter", "gpu_batch_copy_paste", "gpu_batch_mixup"]
