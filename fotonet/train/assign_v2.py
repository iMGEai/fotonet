"""Assignment v2 for dense one-to-one / one-to-many detectors (first used by the fotonet v1.2 trainer).

``assign.dual_o2o_o2m@1`` blends the O2M center prior into "consistent" O2O matching and fades it
together with the quality ramp (fix F8). The deployed head's positives then move from "nearest the
center" to "best IoU" exactly when its class targets finish changing, the most likely cause of r2's
post-ramp dip (recipe book, finding 2). Here O2O matching has a prior blend of its own,
``o2o_prior_mix``, which the trainer fades on its own schedule (``o2o_prior_epochs``); None means no
prior. The O2M head's positive targets can also be task-aligned (``o2m_target="tal"``: alignment over
the ground truth's best alignment, times its best IoU) instead of the plain IoU. Everything else is
``assign.dual_o2o_o2m@1``.
"""
from __future__ import annotations

import torch

from .assign import _Geometry, _dense_targets, _o2o_targets

O2M_TARGETS = ("iou", "tal")


def matcher_fn_v2(
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
    o2o_metric: str = "consistent",
    candidates: int = 0,
    gt_weight: torch.Tensor | None = None,
    return_state: bool = False,
    o2o_prior_mix: torch.Tensor | None = None,
    o2m_target: str = "iou",
):
    """``matcher_fn`` (v1) with a separate O2O prior blend and a choice of O2M target.

    ``prior_mix`` blends the center prior into O2M matching only; ``o2o_prior_mix`` does the same
    for consistent O2O matching (None: no prior there). The output layout is v1's.
    """
    if str(o2m_target) not in O2M_TARGETS:
        raise ValueError(f"o2m_target must be one of {', '.join(O2M_TARGETS)}, got {o2m_target!r}.")
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
            o2o_center_prior, quality_mix, gt_weight, prior_mix=o2o_prior_mix, radius=radius,
        )
        # Extra O2O groups (assign.o2o_groups) follow the O2O rule, so they get its prior.
        state = {"geometry": geo, "taken": taken, "prior_mix": o2o_prior_mix, "radius": radius}

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
        if str(o2m_target) == "tal":
            align = torch.where(sel_ok, tkv, torch.zeros_like(tkv))
            best_iou = torch.where(sel_ok, iou_sel_m, torch.zeros_like(iou_sel_m)).amax(dim=1, keepdim=True)
            quality = align / align.amax(dim=1, keepdim=True).clamp_min(1e-9) * best_iou
        else:
            quality = iou_sel_m
        mixed_m = 1.0 + (quality - 1.0) * quality_mix
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


__all__ = ["O2M_TARGETS", "matcher_fn_v2"]
