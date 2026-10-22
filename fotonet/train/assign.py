"""Assignment for dense one-to-one / one-to-many detectors (first used by fotonet v1.1).

O2M is task-aligned top-k with disjoint anchor ownership. Early in training
the score blends in a center prior so the first non-zero IoU does not
collapse the match. O2O picks its own top-k on the deployed head, with each
anchor owned by at most one GT, scored either by the deployed head alone
("self") or by the O2M alignment metric ("consistent"). Assignment runs over
every anchor (dense) or over the nearest legal candidates per GT (sparse).
"""
from __future__ import annotations

import torch

DIRECT_ASSIGN_CELLS = 16


def xywh_to_xyxy_t(x: torch.Tensor) -> torch.Tensor:
    """Convert [..., 4] cxcywh boxes to xyxy."""
    c, wh = x[..., :2], x[..., 2:]
    return torch.cat((c - wh * 0.5, c + wh * 0.5), -1)


def bbox_iou_t(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU. box1 [B, N, 4], box2 [B, M, 4], both xyxy. Returns [B, N, M]."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0:1], box1[..., 1:2], box1[..., 2:3], box1[..., 3:4]
    b2_x1 = box2[..., 0:1].transpose(1, 2)
    b2_y1 = box2[..., 1:2].transpose(1, 2)
    b2_x2 = box2[..., 2:3].transpose(1, 2)
    b2_y2 = box2[..., 3:4].transpose(1, 2)
    inter_x1 = torch.maximum(b1_x1, b2_x1)
    inter_y1 = torch.maximum(b1_y1, b2_y1)
    inter_x2 = torch.minimum(b1_x2, b2_x2)
    inter_y2 = torch.minimum(b1_y2, b2_y2)
    inter = (inter_x2 - inter_x1).clamp_min(0.0) * (inter_y2 - inter_y1).clamp_min(0.0)
    area1 = (b1_x2 - b1_x1).clamp_min(0.0) * (b1_y2 - b1_y1).clamp_min(0.0)
    area2 = (b2_x2 - b2_x1).clamp_min(0.0) * (b2_y2 - b2_y1).clamp_min(0.0)
    union = area1 + area2 - inter
    return (inter / union.clamp_min(eps)).clamp(0.0, 1.0)


def _candidate_mask(anchors, strides, gtmap, valid, level_strides, imgsz_h, imgsz_w,
                    slack_cells, stal):
    """Anchors whose centers lie inside each ground-truth box and inside the direct-regression range."""
    B, _, _ = gtmap.shape
    gx1 = (gtmap[..., 0] - gtmap[..., 2] * 0.5) * float(imgsz_w)
    gx2 = (gtmap[..., 0] + gtmap[..., 2] * 0.5) * float(imgsz_w)
    gy1 = (gtmap[..., 1] - gtmap[..., 3] * 0.5) * float(imgsz_h)
    gy2 = (gtmap[..., 1] + gtmap[..., 3] * 0.5) * float(imgsz_h)
    gx1, gx2, gy1, gy2 = (t.unsqueeze(1) for t in (gx1, gx2, gy1, gy2))

    cx = anchors[:, 0:1] * strides
    cy = anchors[:, 1:2] * strides
    if stal and level_strides is not None and len(level_strides) >= 2:
        s_min, s_big = float(level_strides[0]), float(level_strides[1])
        vmask = valid.unsqueeze(1)
        gw, gh = gx2 - gx1, gy2 - gy1
        ccx, ccy = (gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5
        ew = torch.where(vmask & (gw < s_min), torch.full_like(gw, s_big), gw)
        eh = torch.where(vmask & (gh < s_min), torch.full_like(gh, s_big), gh)
        sx1, sx2 = ccx - ew * 0.5, ccx + ew * 0.5
        sy1, sy2 = ccy - eh * 0.5, ccy + eh * 0.5
    else:
        sx1, sx2, sy1, sy2 = gx1, gx2, gy1, gy2

    inside = (cx >= sx1) & (cx <= sx2) & (cy >= sy1) & (cy <= sy2)
    cap = float(DIRECT_ASSIGN_CELLS) - 1.01
    dl = (cx - sx1) / strides
    dr = (sx2 - cx) / strides
    dt = (cy - sy1) / strides
    db = (sy2 - cy) / strides
    in_range = (dl <= cap) & (dr <= cap) & (dt <= cap) & (db <= cap)
    mask = (inside & in_range) & valid.unsqueeze(1)

    slack = float(slack_cells)
    if slack > 0.0:
        has_cand = mask.any(dim=1, keepdim=True)
        pad = slack * strides
        gx1e, gx2e = gx1 - pad, gx2 + pad
        gy1e, gy2e = gy1 - pad, gy2 + pad
        inside_fb = (cx >= gx1e) & (cx <= gx2e) & (cy >= gy1e) & (cy <= gy2e)
        dl2 = (cx - gx1e) / strides
        dr2 = (gx2e - cx) / strides
        dt2 = (cy - gy1e) / strides
        db2 = (gy2e - cy) / strides
        in_range_fb = (dl2 <= cap) & (dr2 <= cap) & (dt2 <= cap) & (db2 <= cap)
        mask = mask | ((inside_fb & in_range_fb) & (valid.unsqueeze(1) & ~has_cand))

    gcx = (gx1 + gx2) * 0.5
    gcy = (gy1 + gy2) * 0.5
    cd = (((cx - gcx) / strides).square() + ((cy - gcy) / strides).square()).sqrt()
    return mask, cd


def unique_topk(scores, valid, k, rounds=3, anchor_ids=None, num_anchors=None, taken=None):
    """Pick ``k`` candidates per ground truth with every anchor owned by at most one GT.

    ``scores`` is ``[B, K, M]`` (candidate x GT) and ``valid`` is ``[B, M]``.
    Candidates are anchors themselves (``anchor_ids`` None, K == N) or a
    per-GT candidate list whose anchor ids are ``anchor_ids`` ``[B, K, M]``.
    Each slot lets every pending GT bid on its best free anchor; when several
    GTs bid on the same anchor the highest score wins (lowest GT index on a
    tie) and the losers bid again on what is left, up to ``rounds`` times. A GT
    that still has no free candidate gets value 0 and is treated as unmatched.
    ``taken`` ``[B, N]`` pre-marks anchors owned elsewhere (earlier O2O groups).

    Shapes are static and nothing syncs to the host, so the matcher stays
    ``torch.compile``-friendly. Returns ``(values, anchors, positions, taken)``:
    values, anchor ids, and candidate positions ``[B, k, M]``, plus the updated
    ``taken``.
    """
    B, K, M = scores.shape
    N = int(num_anchors) if num_anchors is not None else K
    k = min(max(int(k), 1), K)
    device = scores.device
    gt_index = torch.arange(M, device=device).view(1, M).expand(B, M)
    taken = torch.zeros(B, N, dtype=torch.bool, device=device) if taken is None else taken.clone()
    values = scores.new_zeros(B, k, M)
    anchors = torch.zeros(B, k, M, dtype=torch.long, device=device)
    positions = torch.zeros(B, k, M, dtype=torch.long, device=device)
    for slot in range(k):
        pending = valid.clone()
        for _ in range(max(int(rounds), 1)):
            if anchor_ids is None:
                blocked = taken.unsqueeze(-1)
            else:
                blocked = taken.gather(1, anchor_ids.reshape(B, -1)).reshape(B, K, M)
            free = scores.masked_fill(blocked, 0.0)
            best_val, best_pos = free.max(dim=1)
            best_anchor = best_pos if anchor_ids is None else anchor_ids.gather(1, best_pos.unsqueeze(1)).squeeze(1)
            active = pending & (best_val > 0)
            bid = torch.where(active, best_val, torch.full_like(best_val, -1.0))
            top_bid = torch.full((B, N), -1.0, device=device, dtype=scores.dtype)
            top_bid.scatter_reduce_(1, best_anchor, bid, reduce="amax", include_self=True)
            win = active & (bid >= top_bid.gather(1, best_anchor))
            first = torch.full((B, N), M, device=device, dtype=torch.long)
            first.scatter_reduce_(
                1, best_anchor, torch.where(win, gt_index, torch.full_like(gt_index, M)),
                reduce="amin", include_self=True,
            )
            win = win & (gt_index == first.gather(1, best_anchor))
            values[:, slot] = torch.where(win, best_val, values[:, slot])
            anchors[:, slot] = torch.where(win, best_anchor, anchors[:, slot])
            positions[:, slot] = torch.where(win, best_pos, positions[:, slot])
            claimed = torch.zeros(B, N, dtype=torch.long, device=device)
            claimed.scatter_reduce_(1, best_anchor, win.long(), reduce="amax", include_self=True)
            taken = taken | claimed.bool()
            pending = pending & ~win
    return values, anchors, positions, taken


def _dense_targets(logits, indices, labels, values, valid_slots):
    """Scatter the best soft target onto [B, N, C]. indices is [B, K, M] anchor ids."""
    B, N, C = logits.shape
    target = torch.zeros(B, N * C, device=logits.device, dtype=logits.dtype)
    lab = labels.masked_fill(~valid_slots, 0)
    flat_index = (indices * C + lab.unsqueeze(1)).reshape(B, -1)
    target.scatter_reduce_(1, flat_index, values.reshape(B, -1), reduce="amax", include_self=True)
    target = target.view(B, N, C)
    return torch.where(target > 0, target.clamp(min=0.05), target)


class _Geometry:
    """Candidate set shared by the O2O and O2M assignment of one batch.

    Dense mode scores every anchor against every GT ([B, N, M]). Sparse mode
    keeps, per GT, the ``candidates`` anchors nearest its center among those
    that may legally regress it ([B, K, M]), so every later tensor is K/N the
    size. Anchors outside the legal mask never become positives either way.
    """

    def __init__(self, anchors, strides, gtmap, labels, valid, level_strides, imgsz_h, imgsz_w,
                 slack_cells, stal, candidates):
        self.B, self.M = gtmap.shape[:2]
        self.N = anchors.shape[0]
        self.valid = valid
        self.labels = labels
        self.lab0 = labels.masked_fill(~valid, 0)
        self.gxyxy = xywh_to_xyxy_t(gtmap)
        mask, cd = _candidate_mask(
            anchors, strides, gtmap, valid, level_strides, imgsz_h, imgsz_w, slack_cells, stal,
        )
        k = int(candidates or 0)
        if 0 < k < self.N:
            key = torch.where(mask, -cd, torch.full_like(cd, float("-inf")))
            _, cand = key.topk(k, dim=1)
            # Anchor order, so score ties break exactly as in dense mode.
            cand, _ = cand.sort(dim=1)
            self.cand = cand
            self.mask = mask.gather(1, cand)
            self.cd = cd.gather(1, cand)
        else:
            self.cand = None
            self.mask = mask
            self.cd = cd

    def anchors_of(self, positions):
        """Anchor ids for candidate positions [B, k, M]."""
        return positions if self.cand is None else self.cand.gather(1, positions)

    def iou(self, boxes_xywh):
        boxes = xywh_to_xyxy_t(boxes_xywh)
        if self.cand is None:
            return bbox_iou_t(boxes, self.gxyxy)
        B, K, M = self.cand.shape
        picked = boxes.gather(1, self.cand.reshape(B, K * M, 1).expand(-1, -1, 4)).reshape(B, K, M, 4)
        g = self.gxyxy.unsqueeze(1)
        ix = (torch.minimum(picked[..., 2], g[..., 2]) - torch.maximum(picked[..., 0], g[..., 0])).clamp_min(0.0)
        iy = (torch.minimum(picked[..., 3], g[..., 3]) - torch.maximum(picked[..., 1], g[..., 1])).clamp_min(0.0)
        inter = ix * iy
        area1 = (picked[..., 2] - picked[..., 0]).clamp_min(0.0) * (picked[..., 3] - picked[..., 1]).clamp_min(0.0)
        area2 = (g[..., 2] - g[..., 0]).clamp_min(0.0) * (g[..., 3] - g[..., 1]).clamp_min(0.0)
        return (inter / (area1 + area2 - inter).clamp_min(1e-7)).clamp(0.0, 1.0)

    def class_prob(self, logits):
        B, N, C = logits.shape
        if self.cand is None:
            return logits.sigmoid().gather(2, self.lab0.unsqueeze(1).expand(B, N, self.M))
        flat = (self.cand * C + self.lab0.unsqueeze(1)).reshape(B, -1)
        return logits.reshape(B, N * C).gather(1, flat).reshape(self.cand.shape).sigmoid()

    def pair_index(self, k):
        B, M = self.B, self.M
        device = self.valid.device
        sb = torch.arange(B, device=device).view(B, 1, 1).expand(B, k, M).reshape(-1)
        sg = torch.arange(M, device=device).view(1, 1, M).expand(B, k, M).reshape(-1)
        return sb, sg


def _o2o_targets(geo, logits, boxes, k, metric, alpha, beta, o2o_beta, o2o_center_prior,
                 quality_mix, gt_weight, taken=None, prior_mix=None, radius=2.5):
    """Unique one-to-one targets for one O2O predictor set."""
    iou = geo.iou(boxes)
    p = geo.class_prob(logits)
    mask_f = geo.mask.to(p.dtype)
    prior = (1.0 - geo.cd / max(float(o2o_center_prior), 1e-6)).clamp(0.05, 1.0)
    if str(metric) == "consistent":
        # YOLOv10: the same metric as O2M, scored on the O2O head, so every O2O
        # positive is also an O2M-quality positive. That includes O2M's
        # center-prior blend early in training; without it the two heads
        # picked anchors by different rules until the blend ended.
        base = p.clamp_min(1e-4).pow(float(alpha))
        score = base * iou.pow(float(beta)) * mask_f
        if prior_mix is not None:
            center = (1.0 - geo.cd / max(float(radius), 1e-6)).clamp_min(0.0)
            score = score * (1.0 - prior_mix) + base * center * mask_f * prior_mix
    else:
        score = p.clamp_min(1e-4) * iou.pow(float(o2o_beta)) * prior * mask_f
    fallback = p.clamp_min(1e-4) * prior * mask_f
    empty = score.amax(dim=1) <= 0
    score = torch.where(empty.unsqueeze(1), fallback, score)
    values, anchors, positions, taken = unique_topk(
        score, geo.valid, k, anchor_ids=geo.cand, num_anchors=geo.N, taken=taken,
    )
    kk = values.shape[1]
    matched = (values > 0) & geo.valid.unsqueeze(1)
    iou_sel = iou.gather(1, positions)
    target = 1.0 + (iou_sel - 1.0) * quality_mix
    target = torch.where(matched, target, torch.zeros_like(values)).clamp(0.0, 1.0)
    target = torch.where(matched, target.clamp(min=0.05), target)
    rw = torch.where(matched, iou_sel.clamp(0.05, 1.0), torch.zeros_like(iou_sel))
    if gt_weight is not None:
        target = target * gt_weight.unsqueeze(1)
        rw = rw * gt_weight.unsqueeze(1)
    tgt = _dense_targets(logits, anchors, geo.labels, target, geo.valid)
    wgt = (tgt.max(-1).values > 0).to(logits.dtype)
    sb, sg = geo.pair_index(kk)
    return tgt, wgt, sb, anchors.reshape(-1), sg, rw.reshape(-1), taken


def matcher_fn(
    logits_o2o: torch.Tensor,
    boxes_o2o: torch.Tensor,
    anchors: torch.Tensor,
    strides: torch.Tensor,
    gtmap: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    level_strides,
    imgsz_h: float,
    imgsz_w: float,
    logits_o2m: torch.Tensor | None = None,
    boxes_o2m: torch.Tensor | None = None,
    topk: int = 8,
    o2o_topk: int = 1,
    alpha: float = 0.5,
    beta: float = 6.0,
    o2o_beta: float = 2.0,
    radius: float = 2.5,
    o2o_center_prior: float = 5.0,
    slack_cells: float = 0.75,
    quality_mix: torch.Tensor | None = None,
    prior_mix: torch.Tensor | None = None,
    stal: bool = True,
    o2o_metric: str = "self",
    candidates: int = 0,
    gt_weight: torch.Tensor | None = None,
    return_state: bool = False,
):
    """Build O2O targets and, when the auxiliary head ran, O2M targets.

    ``o2o_metric`` is ``"self"`` (deployed-head score x IoU^o2o_beta x center
    prior) or ``"consistent"`` (the O2M alignment metric on the O2O head).
    ``prior_mix`` blends a center prior into the O2M score (1 = center prior,
    0 = IoU alignment). ``candidates`` > 0 switches to sparse assignment over
    that many nearest legal anchors per GT. ``gt_weight`` [B, M] scales every
    target and box weight of a GT (pseudo-labels use it).

    With ``return_state`` the 12-tuple is followed by a dict holding the
    geometry and the O2O-owned anchors, which ``o2o_group_targets`` uses.
    """
    with torch.no_grad():
        B, N, C = logits_o2o.shape
        if quality_mix is None:
            quality_mix = logits_o2o.new_zeros(())
        if prior_mix is None:
            prior_mix = logits_o2o.new_zeros(())
        geo = _Geometry(
            anchors, strides, gtmap, labels, valid, level_strides, imgsz_h, imgsz_w,
            slack_cells, stal, candidates,
        )
        tgt_o2o, wgt_o2o, sb_o, sa_o, sg_o, rw_o, taken = _o2o_targets(
            geo, logits_o2o, boxes_o2o, o2o_topk, o2o_metric, alpha, beta, o2o_beta,
            o2o_center_prior, quality_mix, gt_weight, prior_mix=prior_mix, radius=radius,
        )
        state = {"geometry": geo, "taken": taken, "prior_mix": prior_mix, "radius": radius}

        if logits_o2m is None or boxes_o2m is None:
            tgt_o2m = torch.zeros_like(tgt_o2o)
            wgt_o2m = torch.zeros(B, N, device=logits_o2o.device, dtype=logits_o2o.dtype)
            out = (
                tgt_o2m, tgt_o2o, wgt_o2m, wgt_o2o,
                sb_o[:0], sa_o[:0], sg_o[:0], rw_o[:0],
                sb_o, sa_o, sg_o, rw_o,
            )
            return (*out, state) if return_state else out

        iou_m = geo.iou(boxes_o2m)
        base_m = geo.class_prob(logits_o2m).clamp_min(1e-4).pow(float(alpha))
        mask_f = geo.mask.to(base_m.dtype)
        center_prior = (1.0 - geo.cd / max(float(radius), 1e-6)).clamp_min(0.0)
        s_align = base_m * iou_m.pow(float(beta)) * mask_f
        s_center = base_m * center_prior * mask_f
        s_o2m = s_align * (1.0 - prior_mix) + s_center * prior_mix
        empty_gt = s_o2m.amax(dim=1) <= 0
        s_o2m = torch.where(empty_gt.unsqueeze(1), s_center, s_o2m)

        tk = min(max(int(topk), 1), s_o2m.shape[1])
        tkv, tki = s_o2m.topk(tk, dim=1)
        anchor_m = geo.anchors_of(tki)
        iou_sel_m = iou_m.gather(1, tki)
        sel_ok = (tkv > 0) & valid.unsqueeze(1)
        # Disjoint ownership: an anchor claimed by several GTs goes to the one
        # it overlaps most (lowest GT index on a tie).
        gt_index = torch.arange(gtmap.shape[1], device=gtmap.device).view(1, 1, -1).expand_as(anchor_m)
        claim = torch.where(sel_ok, iou_sel_m, torch.full_like(iou_sel_m, -1.0))
        best = torch.full((B, N), -1.0, device=gtmap.device, dtype=claim.dtype)
        best.scatter_reduce_(1, anchor_m.reshape(B, -1), claim.reshape(B, -1), reduce="amax", include_self=True)
        own = sel_ok & (claim >= best.gather(1, anchor_m.reshape(B, -1)).reshape(anchor_m.shape))
        first = torch.full((B, N), gtmap.shape[1], device=gtmap.device, dtype=torch.long)
        first.scatter_reduce_(
            1, anchor_m.reshape(B, -1),
            torch.where(own, gt_index, torch.full_like(gt_index, gtmap.shape[1])).reshape(B, -1),
            reduce="amin", include_self=True,
        )
        sel_ok = own & (gt_index == first.gather(1, anchor_m.reshape(B, -1)).reshape(anchor_m.shape))
        mixed_m = 1.0 + (iou_sel_m - 1.0) * quality_mix
        soft_m = torch.where(sel_ok, mixed_m, torch.zeros_like(iou_sel_m))
        rw_m = torch.where(sel_ok, iou_sel_m.clamp(0.05, 1.0), torch.zeros_like(iou_sel_m))
        if gt_weight is not None:
            soft_m = soft_m * gt_weight.unsqueeze(1)
            rw_m = rw_m * gt_weight.unsqueeze(1)
        tgt_o2m = _dense_targets(logits_o2m, anchor_m, labels, soft_m, valid)
        wgt_o2m = (tgt_o2m.max(-1).values > 0).to(logits_o2m.dtype)
        sb_m, sg_m = geo.pair_index(tk)
        out = (
            tgt_o2m, tgt_o2o, wgt_o2m, wgt_o2o,
            sb_m, anchor_m.reshape(-1), sg_m, rw_m.reshape(-1),
            sb_o, sa_o, sg_o, rw_o,
        )
        return (*out, state) if return_state else out


def o2o_group_targets(state, logits, boxes, o2o_topk=1, o2o_metric="self", alpha=0.5, beta=6.0,
                      o2o_beta=2.0, o2o_center_prior=5.0, quality_mix=None, gt_weight=None):
    """Targets for an extra training-only O2O group (Group-DETR style).

    Each group is one-to-one on its own, and avoids the anchors earlier groups
    took, so groups add independent positives instead of duplicates. Updates
    ``state['taken']`` in place for the next group. Returns
    ``(tgt, wgt, sb, sa, sg, rw)``.
    """
    with torch.no_grad():
        if quality_mix is None:
            quality_mix = logits.new_zeros(())
        tgt, wgt, sb, sa, sg, rw, taken = _o2o_targets(
            state["geometry"], logits, boxes, o2o_topk, o2o_metric, alpha, beta, o2o_beta,
            o2o_center_prior, quality_mix, gt_weight, taken=state["taken"],
            prior_mix=state.get("prior_mix"), radius=state.get("radius", 2.5),
        )
        state["taken"] = taken
        return tgt, wgt, sb, sa, sg, rw


__all__ = [
    "matcher_fn", "o2o_group_targets", "unique_topk", "xywh_to_xyxy_t", "bbox_iou_t",
    "DIRECT_ASSIGN_CELLS",
]
