"""Detection loss v2 (first used by the fotonet v1.2 trainer).

Same structure as ``loss.dual_o2o_o2m@1`` (``losses.py``): the deployed one-to-one (O2O) head's
class and box terms, plus the one-to-many (O2M) head's quality focal, box and objectness terms. The
O2M side is unchanged. What changes is the O2O class term, ``o2o_cls``:

- ``"bce"`` (default): binary cross-entropy against the matched IoU target with every negative at
  full strength, as the one-to-one heads of YOLOv10 and YOLO26 train. Varifocal's ``0.75 * p^2``
  negative weight let fotonetn's deployed head keep about 62% duplicate boxes for 200 epochs
  (recipe book, finding 1).
- ``"mal"``: DEIM's matchability-aware loss, positive target ``q^gamma`` and negatives weighted by
  ``p^gamma``.
- ``"vfl"``: varifocal as in v1 (``o2o_neg_alpha``, ``o2o_neg_gamma``).

``o2o_q_gamma`` raises the positive target to a power (1 keeps the IoU; MAL uses 1.5), so a badly
placed match scores lower than a well placed one. ``w_runner_up`` still multiplies the negative term
on the anchors the O2M head treats as positive, the ones that turn into duplicates.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .losses import _masked_box_loss, _box_iou_loss, quality_focal_loss, varifocal_loss

O2O_CLS = ("bce", "mal", "vfl")


def o2o_cls_loss(logits, targets, kind="bce", q_gamma=1.0, neg_alpha=1.0, neg_gamma=1.0, neg_scale=None):
    """Element-wise class loss of a one-to-one head; ``targets`` hold the matched IoU, 0 for negatives."""
    kind = str(kind).lower()
    if kind == "vfl":
        return varifocal_loss(logits, targets, alpha=neg_alpha, gamma=neg_gamma, neg_scale=neg_scale)
    if kind not in O2O_CLS:
        raise ValueError(f"o2o_cls must be one of {', '.join(O2O_CLS)}, got {kind!r}.")
    pos = targets > 0
    q = targets.clamp(0.0, 1.0)
    if float(q_gamma) != 1.0:
        q = torch.where(pos, q.clamp_min(1e-6).pow(float(q_gamma)), q)
    bce = F.binary_cross_entropy_with_logits(logits, q, reduction="none")
    if kind == "bce":
        negative = torch.ones_like(bce)
    else:
        negative = logits.sigmoid().detach().pow(float(q_gamma))
    if neg_scale is not None:
        negative = negative * neg_scale
    return torch.where(pos, torch.ones_like(bce), negative) * bce


def o2o_branch_loss_v2(logits, boxes, tgt, wgt, sb, sa, sg, rw, gtmap, w_box=7.5, gamma=2.0,
                       iou_type="siou", o2o_cls="bce", o2o_q_gamma=1.0, o2o_neg_alpha=1.0, o2o_neg_gamma=1.0):
    """Class (``o2o_cls``) and IoU loss of one one-to-one predictor set. Returns (total, cls, box)."""
    n_pos = (wgt > 0).sum().clamp(min=1.0)
    l_cls = o2o_cls_loss(
        logits, tgt, kind=o2o_cls, q_gamma=o2o_q_gamma, neg_alpha=o2o_neg_alpha, neg_gamma=o2o_neg_gamma,
    ).sum() / n_pos
    l_box = _masked_box_loss(boxes[sb, sa], gtmap[sb, sg], rw, iou_type)
    return l_cls + float(w_box) * l_box, l_cls.detach(), l_box.detach()


def loss_fn_v2(
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
    o2o_cls="bce",
    o2o_q_gamma=1.0,
):
    """``loss.dual_o2o_o2m@1`` with the O2O class term chosen by ``o2o_cls``.

    Returns ``(total, cls_o2o, box_o2o, cls_o2m, box_o2m, cons, obj)``, as v1.
    """
    pos = wgt_o2o > 0
    n_pos = pos.sum().clamp(min=1.0)
    neg_scale = None
    if float(w_runner_up) != 1.0 and wgt_o2m is not None:
        runner_up = (wgt_o2m > 0) & ~pos
        neg_scale = torch.where(runner_up, float(w_runner_up), 1.0).unsqueeze(-1).to(preds_logits_o2o.dtype)
    neg_gamma = gamma if o2o_neg_gamma is None else o2o_neg_gamma
    l_cls_o2o = o2o_cls_loss(
        preds_logits_o2o, tgt_o2o, kind=o2o_cls, q_gamma=o2o_q_gamma, neg_alpha=o2o_neg_alpha,
        neg_gamma=neg_gamma, neg_scale=neg_scale,
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


__all__ = ["O2O_CLS", "loss_fn_v2", "o2o_branch_loss_v2", "o2o_cls_loss"]
