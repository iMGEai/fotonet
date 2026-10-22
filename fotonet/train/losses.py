"""Detection losses: varifocal, quality focal, IoU-family box losses (first used by fotonet v1.1).

The deployed head uses varifocal loss so its score learns to rank like IoU.
The auxiliary head uses quality focal loss on its soft targets. Box regression
is SIoU or CIoU, weighted by the matched IoU. Direct regression has no DFL
and no second coordinate loss. Consistency starts off: it is computed only
when ``w_cons`` is positive and the auxiliary head actually ran.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .assign import xywh_to_xyxy_t


def bbox_ciou(box1_xyxy, box2_xyxy, eps=1e-7):
    """CIoU between paired boxes. [N, 4] vs [N, 4]."""
    b1x1, b1y1, b1x2, b1y2 = box1_xyxy.unbind(-1)
    b2x1, b2y1, b2x2, b2y2 = box2_xyxy.unbind(-1)
    inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * \
            (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
    w1, h1 = (b1x2 - b1x1).clamp(min=eps), (b1y2 - b1y1).clamp(min=eps)
    w2, h2 = (b2x2 - b2x1).clamp(min=eps), (b2y2 - b2y1).clamp(min=eps)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = (torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)).clamp(min=eps)
    ch = (torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)).clamp(min=eps)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b1x1 + b1x2 - b2x1 - b2x2) ** 2 + (b1y1 + b1y2 - b2y1 - b2y2) ** 2) / 4
    v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)
    return iou - (rho2 / c2 + v * alpha)


def bbox_siou(box1_xyxy, box2_xyxy, eps=1e-7):
    """SIoU between paired boxes. [N, 4] vs [N, 4]."""
    b1x1, b1y1, b1x2, b1y2 = box1_xyxy.unbind(-1)
    b2x1, b2y1, b2x2, b2y2 = box2_xyxy.unbind(-1)
    inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * \
            (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
    w1, h1 = (b1x2 - b1x1).clamp(min=eps), (b1y2 - b1y1).clamp(min=eps)
    w2, h2 = (b2x2 - b2x1).clamp(min=eps), (b2y2 - b2y1).clamp(min=eps)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = (torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)).clamp(min=eps)
    ch = (torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)).clamp(min=eps)
    cw1, ch1 = (b1x1 + b1x2) * 0.5, (b1y1 + b1y2) * 0.5
    cw2, ch2 = (b2x1 + b2x2) * 0.5, (b2y1 + b2y2) * 0.5
    s_cw = (cw2 - cw1).abs()
    s_ch = (ch2 - ch1).abs()
    sigma = torch.sqrt(s_cw ** 2 + s_ch ** 2 + eps)
    sin_alpha = (s_ch / sigma).clamp(0.0, 1.0)
    sin_beta = (s_cw / sigma).clamp(0.0, 1.0)
    sin_angle = torch.where(sin_alpha > 0.70710678, sin_beta, sin_alpha)
    angle_cost = 1.0 - 2.0 * torch.sin(torch.asin(sin_angle.clamp(0.0, 0.9999)) - (math.pi / 4)) ** 2
    gamma = 2.0 - angle_cost
    rho_x = ((cw2 - cw1) / cw) ** 2
    rho_y = ((ch2 - ch1) / ch) ** 2
    dist_cost = 2.0 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)
    # Shape cost (SIoU, theta = 4): sum over w/h of (1 - exp(-omega)) ** theta,
    # omega = |w - w_gt| / max(w, w_gt). The power is applied once, outside.
    max_w = torch.max(w1, w2).clamp(min=eps)
    max_h = torch.max(h1, h2).clamp(min=eps)
    omega_w = (w1 - w2).abs() / max_w
    omega_h = (h1 - h2).abs() / max_h
    shape_cost = (1.0 - torch.exp(-omega_w)) ** 4 + (1.0 - torch.exp(-omega_h)) ** 4
    return iou - 0.5 * (dist_cost + shape_cost)


def _box_iou_loss(pred_xywh, gt_xywh, iou_type):
    pred = xywh_to_xyxy_t(pred_xywh)
    gt = xywh_to_xyxy_t(gt_xywh)
    if str(iou_type).lower() == "ciou":
        return 1.0 - bbox_ciou(pred, gt)
    return 1.0 - bbox_siou(pred, gt)


def _masked_box_loss(pred, gt, weight, iou_type):
    """IoU loss on positive pairs only. Invalid pairs are replaced with zeros before the kernel."""
    safe = weight > 0
    gate = safe.unsqueeze(-1)
    pred_use = torch.where(gate, pred, torch.zeros_like(pred))
    gt_use = torch.where(gate, gt, torch.zeros_like(gt))
    term = _box_iou_loss(pred_use, gt_use, iou_type)
    term = torch.where(safe, term, torch.zeros_like(term))
    denom = weight.sum().clamp(min=1.0)
    return (term * weight).sum() / denom


def varifocal_loss(logits, targets, alpha=0.75, gamma=2.0, neg_scale=None):
    """Varifocal loss. Positives are weighted by their IoU target. Negatives use p^gamma.

    ``neg_scale`` (broadcastable to ``logits``) multiplies the negative weight.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pred = logits.sigmoid().detach()
    negative = float(alpha) * pred.pow(float(gamma))
    if neg_scale is not None:
        negative = negative * neg_scale
    weight = torch.where(targets > 0, targets, negative)
    return weight * bce


def quality_focal_loss(logits, targets, gamma=2.0):
    """Quality focal loss for continuous soft targets."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    modulating = (targets - logits.sigmoid()).abs().pow(float(gamma))
    return modulating * bce


def o2o_branch_loss(logits, boxes, tgt, wgt, sb, sa, sg, rw, gtmap, w_box=7.5, gamma=2.0,
                    iou_type="siou"):
    """Varifocal + IoU loss of one one-to-one predictor set. Returns (total, cls, box)."""
    n_pos = (wgt > 0).sum().clamp(min=1.0)
    l_cls = varifocal_loss(logits, tgt, gamma=gamma).sum() / n_pos
    l_box = _masked_box_loss(boxes[sb, sa], gtmap[sb, sg], rw, iou_type)
    return l_cls + float(w_box) * l_box, l_cls.detach(), l_box.detach()


def loss_fn(
    preds_logits_o2o,
    preds_boxes_o2o,
    tgt_o2o,
    wgt_o2o,
    sb_o,
    sa_o,
    sg_o,
    rw_o,
    gtmap,
    preds_logits_o2m=None,
    preds_boxes_o2m=None,
    tgt_o2m=None,
    wgt_o2m=None,
    sb_m=None,
    sa_m=None,
    sg_m=None,
    rw_m=None,
    w_box=7.5,
    w_o2o=1.0,
    w_cons=0.0,
    gamma=2.0,
    iou_type="siou",
    preds_obj_o2m=None,
    w_obj=0.0,
    w_runner_up=1.0,
    o2o_neg_alpha=0.75,
    o2o_neg_gamma=None,
):
    """Fused O2O loss plus the optional O2M auxiliary loss and its objectness term.

    The objectness branch (training-only) learns "is any object here" with the
    O2M soft target of the anchor's best class, the same foreground/background
    signal YOLO27's small models train with. Returns
    ``(total, cls_o2o, box_o2o, cls_o2m, box_o2m, cons, obj)``.

    ``o2o_neg_alpha`` and ``o2o_neg_gamma`` set the deployed head's negative weight,
    ``alpha * p^gamma``; ``o2o_neg_gamma`` None keeps ``gamma``.
    """
    pos = wgt_o2o > 0
    n_pos = pos.sum().clamp(min=1.0)
    neg_scale = None
    if float(w_runner_up) != 1.0 and wgt_o2m is not None:
        # Anchors the O2M head treats as positive but O2O does not are the
        # neighbors that turn into duplicates; push their scores down harder.
        runner_up = (wgt_o2m > 0) & ~pos
        neg_scale = torch.where(runner_up, float(w_runner_up), 1.0).unsqueeze(-1).to(preds_logits_o2o.dtype)
    neg_gamma = gamma if o2o_neg_gamma is None else o2o_neg_gamma
    l_cls_o2o = varifocal_loss(
        preds_logits_o2o, tgt_o2o, alpha=o2o_neg_alpha, gamma=neg_gamma, neg_scale=neg_scale,
    ).sum() / n_pos
    boxes_o = preds_boxes_o2o[sb_o, sa_o]
    gt_o = gtmap[sb_o, sg_o]
    l_box_o2o = _masked_box_loss(boxes_o, gt_o, rw_o, iou_type)

    zero = l_cls_o2o.new_zeros(())
    ran_o2m = (
        preds_logits_o2m is not None and preds_boxes_o2m is not None
        and tgt_o2m is not None and rw_m is not None and rw_m.numel() > 0
    )
    if ran_o2m:
        pos_m = wgt_o2m > 0
        n_pos_m = pos_m.sum().clamp(min=1.0)
        l_cls_o2m = quality_focal_loss(preds_logits_o2m, tgt_o2m, gamma=gamma).sum() / n_pos_m
        boxes_m = preds_boxes_o2m[sb_m, sa_m]
        gt_m = gtmap[sb_m, sg_m]
        l_box_o2m = _masked_box_loss(boxes_m, gt_m, rw_m, iou_type)
    else:
        l_cls_o2m = zero
        l_box_o2m = zero

    if ran_o2m and preds_obj_o2m is not None and float(w_obj) > 0.0:
        obj_target = tgt_o2m.amax(dim=-1)
        l_obj = quality_focal_loss(preds_obj_o2m, obj_target, gamma=gamma).sum() / n_pos_m
    else:
        l_obj = zero

    if ran_o2m and float(w_cons) > 0.0:
        teacher_cls = preds_logits_o2m[sb_o, sa_o].detach().sigmoid().clamp(0.01, 0.99)
        cls_term = F.binary_cross_entropy_with_logits(
            preds_logits_o2o[sb_o, sa_o], teacher_cls, reduction="none",
        ).sum(-1)
        teacher_box = preds_boxes_o2m[sb_o, sa_o].detach()
        box_term = _box_iou_loss(
            torch.where((rw_o > 0).unsqueeze(-1), boxes_o, torch.zeros_like(boxes_o)),
            torch.where((rw_o > 0).unsqueeze(-1), teacher_box, torch.zeros_like(teacher_box)),
            iou_type,
        )
        denom = rw_o.sum().clamp(min=1.0)
        l_cons = ((0.25 * cls_term + 0.75 * box_term) * rw_o).sum() / denom
    else:
        l_cons = zero

    loss_o2o = l_cls_o2o + float(w_box) * l_box_o2o + float(w_cons) * l_cons
    loss_o2m = l_cls_o2m + float(w_box) * l_box_o2m + float(w_obj) * l_obj
    total = float(w_o2o) * loss_o2o + loss_o2m
    return (
        total,
        l_cls_o2o.detach(),
        l_box_o2o.detach(),
        l_cls_o2m.detach(),
        l_box_o2m.detach(),
        l_cons.detach(),
        l_obj.detach(),
    )


__all__ = [
    "loss_fn", "o2o_branch_loss", "varifocal_loss", "quality_focal_loss", "bbox_siou", "bbox_ciou",
]
